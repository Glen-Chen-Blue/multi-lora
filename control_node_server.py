import os
import time
import uuid
import asyncio
import httpx
import logging
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Deque, Any, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL: %(message)s")
logger = logging.getLogger("ControlNode")
logging.getLogger("httpx").setLevel(logging.WARNING)

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
CLUSTER_NAME = os.environ.get("CLUSTER_NAME", "cluster_1")
CLUSTER_ID = CLUSTER_NAME  # 保留向下相容
EFO_URL = os.environ.get("EFO_URL", "http://127.0.0.1:9100")
MY_URL = os.environ.get("CONTROL_NODE_URL", "http://127.0.0.1:9000") # 提供給 EFO 自己的位址

limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# Lyapunov, TTFT & Auto-Scaling Hyperparameters
# ============================================================
T_MAX = 5.5                 
EPSILON = 0.05               
PSI_DROP = 10.0              

# Scale Up/Down 參數設定
SCALE_UP_DROP_THRESHOLD = 5
SCALE_DOWN_SURPLUS_THRESHOLD = 15  # 一台機器的基準容量(12) + 緩衝(3)

Z_debt = 0.0
z_lock = asyncio.Lock()

switching_nodes: Set[str] = set()
recent_capacity_drops: Deque[float] = deque()

# 紀錄收到的 LoRA 請求次數
lora_request_stats: Dict[str, int] = defaultdict(int)
stats_lock = asyncio.Lock()

def record_capacity_drop():
    recent_capacity_drops.append(time.time())

def get_recent_capacity_drops_count(window: float = 6.0) -> int:
    now = time.time()
    while recent_capacity_drops and now - recent_capacity_drops[0] > window:
        recent_capacity_drops.popleft()
    return len(recent_capacity_drops)

# ============================================================
# 模擬 EFO 資訊表 (現在由 EFO 伺服器動態提供)
# ============================================================
LORA_METADATA_TABLE: Dict[str, Any] = {}
LOCAL_AVAILABLE_LORAS: Set[str] = set()

# 儲存由 EFO 廣播過來的全域路由表 (供後續 Offloading 決策使用)
global_routing_table: Dict[str, Any] = {}

# 本地接收 offloading 狀態控制，避免反覆廣播 Halt
can_accept_offload: bool = True

# ============================================================
# Global State (含待機與 Draining 機制)
# ============================================================
class NodeManager:
    def __init__(self): 
        self.nodes: Dict[str, Dict] = {}
        
    def register_node(self, url: str):
        if url not in self.nodes:
            has_active = any(d["status"] == "active" for d in self.nodes.values())
            status = "standby" if has_active else "active"
            self.nodes[url] = {"metrics": None, "last_seen": time.time(), "status": status}
            logger.info(f"✅ Registered Node: {url} | Assigned Status: {status}")
            
            # 主動通知 Compute Node 切換到指定狀態
            asyncio.create_task(client.post(f"{url}/set_status", json={"status": status}))
        else: 
            self.nodes[url]["last_seen"] = time.time()
            
    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()
            
            # 以 Compute Node 回報的真實狀態為主
            reported_status = metrics.get("status", "standby")
            if self.nodes[url]["status"] == "draining" and reported_status == "standby":
                logger.info(f"❄️ [Scale Down] 節點已完全排空，自動轉為 Standby: {url}")
            self.nodes[url]["status"] = reported_status
            
    async def scale_up_one_node(self) -> bool:
        for url, data in self.nodes.items():
            if data["status"] == "standby":
                try:
                    resp = await client.post(f"{url}/set_status", json={"status": "active"}, timeout=2.0)
                    if resp.status_code == 200:
                        self.nodes[url]["status"] = "active"
                        logger.info(f"🚀 [Scale Up] 成功喚醒待機節點: {url}")
                        return True
                except Exception as e:
                    logger.error(f"喚醒節點 {url} 失敗: {e}")
        logger.warning("⚠️ [Scale Up] 擴容失敗：沒有可用的待機節點！")
        return False

    async def trigger_drain_best_node(self, v_nodes: List['VirtualNodeState']) -> bool:
        active_nodes = [n for n in v_nodes if self.nodes[n.url]["status"] == "active"]
        if len(active_nodes) <= 1:
            return False # 保留至少一台
            
        global_lora_counts = defaultdict(int)
        for n in active_nodes:
            for lora in n.active_loras:
                global_lora_counts[lora] += 1
                
        current_unmerge_count = sum(1 for n in active_nodes if n.mode == "unmerge")
                
        def drain_score(n: VirtualNodeState):
            orphan_count = sum(1 for lora in n.active_loras if global_lora_counts[lora] == 1)
            is_last_unmerge = (n.mode == "unmerge" and current_unmerge_count <= 1)
            last_unmerge_penalty = 10000 if is_last_unmerge else 0
            mode_penalty = 100 if n.mode == "merge" else 0
            
            return (orphan_count * 1000) + last_unmerge_penalty + mode_penalty + n.running_batch
            
        best_node = sorted(active_nodes, key=drain_score)[0]
        
        try:
            resp = await client.post(f"{best_node.url}/drain", timeout=2.0)
            if resp.status_code == 200:
                self.nodes[best_node.url]["status"] = "draining"
                logger.info(f"🚰 [Scale Down] 節點進入 Draining: {best_node.url} (孤兒:{sum(1 for lora in best_node.active_loras if global_lora_counts[lora] == 1)}, 負載:{best_node.running_batch}, 模式:{best_node.mode})")
                return True
        except Exception as e:
            logger.error(f"Drain 節點 {best_node.url} 失敗: {e}")
            
        return False

node_mgr = NodeManager()
request_queues: Dict[str, Deque] = defaultdict(deque)
global_request_list: List[Dict] = [] 
stream_queues: Dict[str, Dict] = {} 

class VirtualNodeState:
    def __init__(self, url: str, metrics: Dict):
        self.url = url
        self.mode = metrics.get("mode", "unmerge")
        load_data = metrics.get("load", {})
        lora_data = metrics.get("lora_state", {})
        
        self.running_batch = load_data.get("running_batch", 0)
        self.merged_adapter = lora_data.get("merged_adapter")
        self.active_loras = set(lora_data.get("running_adapters", []))
        self.loaded_adapters = set(lora_data.get("loaded_adapters", []))
        self.request_set = metrics.get("request_set", [])
        
        self.lora_request_counts = defaultdict(int)
        for req in self.request_set:
            self.lora_request_counts[req["adapter_id"]] += 1

        self.capacity_merged = 15
        self.capacity_unmerged_base = 12

    def get_free_slots(self, target_lora: str) -> int:
        if self.mode == "merge":
            return max(0, self.capacity_merged - self.running_batch) if self.merged_adapter == target_lora else 0
        current_cost = self.running_batch + len(self.active_loras)
        margin = self.capacity_unmerged_base - current_cost
        if target_lora not in self.active_loras:
            return (margin - 1) if margin >= 2 else 0 
        return max(0, margin)

    def commit_request(self, target_lora: str):
        self.running_batch += 1
        self.active_loras.add(target_lora)
        self.loaded_adapters.add(target_lora)
        self.lora_request_counts[target_lora] += 1

    def rollback_request(self, target_lora: str):
        self.running_batch = max(0, self.running_batch - 1)
        if self.lora_request_counts[target_lora] > 0:
            self.lora_request_counts[target_lora] -= 1
            if self.lora_request_counts[target_lora] == 0:
                self.active_loras.discard(target_lora)

def predict_cluster_ttft(nodes: List[VirtualNodeState], target_lora: str, global_pending_ahead: int) -> float:
    SCHEDULER_OVERHEAD = 0.300 
    DECODE_P95_SPEED = 0.040   
    DISK_LOAD_DELAY = 0.200    

    node_scores = []
    total_free = 0
    cluster_concurrent_capacity = 0
    
    for node in nodes:
        if node.url in switching_nodes or node.mode == "switching": continue
        is_merge = (node.mode == "merge" and node.merged_adapter == target_lora)
        if node.mode == "merge" and not is_merge: continue 
        
        is_in_vram = (node.mode == "unmerge" and target_lora in node.active_loras)
        is_in_cpu = (node.mode == "unmerge" and target_lora in node.loaded_adapters)
        is_empty = (node.mode == "unmerge" and len(node.active_loras) == 0)
        
        free_slots = node.get_free_slots(target_lora)
        
        if is_merge:
            cluster_concurrent_capacity += node.capacity_merged
        elif node.mode == "unmerge":
            cluster_concurrent_capacity += max(0, node.capacity_unmerged_base - 1)

        if free_slots > 0:
            score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, free_slots)
            node_scores.append({"score": score, "free": free_slots})
            total_free += free_slots

    my_position = global_pending_ahead + 1

    if my_position <= total_free:
        node_scores.sort(key=lambda x: x["score"], reverse=True)
        allocated = 0
        landing_score = None
        take_at_landing = 0
        
        for ns in node_scores:
            take = min(ns["free"], my_position - allocated)
            allocated += take
            if allocated == my_position:
                landing_score = ns["score"]
                take_at_landing = take
                break
                
        is_merge = landing_score[0] == 1
        is_in_cpu = landing_score[2] == 1
        multiplier = 0.8 if is_merge else 1.0
        
        load_delay = 0.0 if (is_in_cpu or is_merge) else DISK_LOAD_DELAY
        prefill_time = 0.050 * take_at_landing * multiplier
        
        return SCHEDULER_OVERHEAD + load_delay + prefill_time
    else:
        needed_to_finish = my_position - total_free
        if cluster_concurrent_capacity == 0: cluster_concurrent_capacity = 12
        full_cycles = needed_to_finish // cluster_concurrent_capacity
        remainder = needed_to_finish % cluster_concurrent_capacity
        
        all_remains = []
        for node in nodes:
            if node.url in switching_nodes or node.mode == "switching": continue
            if (node.mode == "merge" and node.merged_adapter == target_lora) or (node.mode == "unmerge"):
                all_remains.extend([r.get("remaining_tokens", 256) for r in node.request_set])
                
        if not all_remains: 
            current_wait = 1.0
        else:
            all_remains.sort()
            idx = min(len(all_remains) - 1, max(0, remainder - 1))
            current_wait = all_remains[idx] * DECODE_P95_SPEED
            
        total_wait_time = current_wait + (full_cycles * 256 * DECODE_P95_SPEED)
        return SCHEDULER_OVERHEAD + total_wait_time + 0.050

async def handle_offload_or_drop(rid: str, is_local: bool, best_ttft: float, z_current: float, q: asyncio.Queue, force_offload: bool = False, force_drop_reason: str = None) -> bool:
    if force_drop_reason:
        logger.warning(f"🚫 [Drop] {rid[:8]} | Reason: {force_drop_reason}")
        await q.put({"type": "error", "message": force_drop_reason})
        await q.put(None)
        return True

    if force_offload:
        logger.warning(f"🌐 [Offload] Global LoRA {rid[:8]} | Reason: Artifact Missing")
        await q.put({"type": "error", "message": "Artifact Missing (Offload Not Implemented)"})
        await q.put(None)
        return True

    if z_current > PSI_DROP:
        record_capacity_drop()
        if is_local:
            logger.warning(f"🚫 [Drop] Local LoRA {rid[:8]} | Pred TTFT: {best_ttft:.1f}s | Z: {z_current:.2f}")
            await q.put({"type": "error", "message": "System Congested (Local Drop)"})
        else:
            logger.warning(f"🌐 [Offload] Global LoRA {rid[:8]} | Pred TTFT: {best_ttft:.1f}s | Z: {z_current:.2f}")
            await q.put({"type": "error", "message": "System Congested (Offload Not Implemented)"})
        await q.put(None)
        return True
    return False

async def broadcast_halt():
    """廣播緊急 Halt 訊號給其他的 Control Node"""
    logger.warning("🚨 [Backpressure] Capacity exhausted! Broadcasting HALT to other clusters.")
    async with httpx.AsyncClient(timeout=2.0) as client:
        tasks = []
        for cluster, data in global_routing_table.items():
            if cluster != CLUSTER_NAME:
                url = f"{data['ip']}/emergency_halt"
                tasks.append(client.post(url, json={"cluster_name": CLUSTER_NAME}))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.debug(f"Halt broadcast to one cluster failed: {res}")

async def safe_mode_switch(node_url: str, endpoint: str, payload: Dict):
    if node_url in switching_nodes: return
    switching_nodes.add(node_url)
    try:
        resp = await client.post(f"{node_url}{endpoint}", json=payload, timeout=5.0)
        if resp.status_code == 200:
            logger.info(f"✅ Mode Switch {endpoint} Success: {node_url}")
    except Exception as e:
        logger.error(f"❌ Mode Switch Error: {e}")
    finally:
        await asyncio.sleep(0.5)
        switching_nodes.discard(node_url)

async def dispatch_task(v_node_url: str, req_data: dict, v_node_ptr: Optional[VirtualNodeState] = None, target_lora: str = None):
    rid = req_data["request_id"]
    if not target_lora: target_lora = req_data["adapter_id"]
    user_data = stream_queues.get(rid)
    if not user_data: return
    user_q = user_data["q"]

    try:
        payload = {"prompt": req_data["prompt"], "adapter_id": target_lora, "max_new_tokens": req_data.get("max_new_tokens", 256)}
        async with client.stream("POST", f"{v_node_url}/add_request", json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                if v_node_ptr: v_node_ptr.rollback_request(target_lora)
                await user_q.put({"type": "error", "message": f"Node Error {resp.status_code}"})
                return
            async for line in resp.aiter_lines():
                if line.startswith("data:") and user_q:
                    content = line[len("data:"):].strip()
                    if content: await user_q.put(content)
    except Exception as e:
        if v_node_ptr: v_node_ptr.rollback_request(target_lora)
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None)

async def scheduler_loop():
    global global_request_list
    logger.info("⏳ SP2 Full-Function Scheduler loop started.")
    while True:
        try:
            all_node_urls = list(node_mgr.nodes.keys())
            if all_node_urls:
                tasks = [client.get(f"{u}/metrics", timeout=1.0) for u in all_node_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(responses):
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        node_mgr.update_metrics(all_node_urls[i], r.json())

            active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
            v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]
            
            if not v_nodes: await asyncio.sleep(0.1); continue

            total_pending = sum(len(q) for q in request_queues.values())
            unmerged_count = sum(1 for n in v_nodes if n.mode == "unmerge")

            for v in v_nodes:
                if v.url in switching_nodes: continue 
                if v.mode == "unmerge" and unmerged_count > 1 and v.running_batch >= 10 and len(v.active_loras) == 1:
                    aid = next(iter(v.active_loras))
                    if len(request_queues[aid]) > 0:
                        asyncio.create_task(safe_mode_switch(v.url, "/merge", {"adapter_id": aid, "force": False}))
                        v.mode = "switching"; unmerged_count -= 1
                elif v.mode == "merge" and v.running_batch < 9:
                    if len(request_queues[v.merged_adapter]) == 0:
                        asyncio.create_task(safe_mode_switch(v.url, "/unmerge", {"force": False}))
                        v.mode = "switching"; unmerged_count += 1

            dispatched_any = True
            while dispatched_any and global_request_list:
                dispatched_any = False
                for req_meta in list(global_request_list):
                    target_aid = req_meta["original_aid"]
                    meta = LORA_METADATA_TABLE.get(target_aid, {})
                    valid_aids = [target_aid] + [s for s in meta.get("substitutes", []) if s in LOCAL_AVAILABLE_LORAS]
                    valid_aids = [aid for aid in valid_aids if aid in LOCAL_AVAILABLE_LORAS]
                    if not valid_aids: valid_aids = [target_aid]

                    best_plan = None
                    for aid in valid_aids:
                        candidate_reqs = []
                        for q_aid, q_reqs in request_queues.items():
                            if aid == q_aid or aid in LORA_METADATA_TABLE.get(q_aid, {}).get("substitutes", []):
                                candidate_reqs.extend(q_reqs)
                        if not candidate_reqs: continue
                        candidate_reqs.sort(key=lambda x: x["arrival_time"])
                        
                        for v in v_nodes:
                            if v.url in switching_nodes or v.mode == "switching": continue
                            free_slots = v.get_free_slots(aid)
                            if free_slots <= 0: continue
                            
                            can_take = min(free_slots, len(candidate_reqs))
                            if can_take == 0: continue
                            
                            is_merge = (v.mode == "merge" and v.merged_adapter == aid)
                            is_in_vram = (v.mode == "unmerge" and aid in v.active_loras)
                            is_in_cpu = (v.mode == "unmerge" and aid in v.loaded_adapters)
                            is_empty = (v.mode == "unmerge" and len(v.active_loras) == 0)
                            
                            score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, can_take)
                            
                            if best_plan is None or score > best_plan["score"]:
                                best_plan = {"node": v, "lora": aid, "requests": candidate_reqs[:can_take], "score": score}
                    
                    if best_plan:
                        node, aid, reqs_to_dispatch = best_plan["node"], best_plan["lora"], best_plan["requests"]
                        for req in reqs_to_dispatch:
                            q_aid = req["original_aid"]
                            request_queues[q_aid] = deque([r for r in request_queues[q_aid] if r["request_id"] != req["request_id"]])
                            global_request_list = [r for r in global_request_list if r["request_id"] != req["request_id"]]
                            req["adapter_id"] = aid
                            node.commit_request(aid)
                            asyncio.create_task(dispatch_task(node.url, req, node, target_lora=aid))
                        dispatched_any = True
                        break 
        except Exception as e: logger.error(f"🔥 Scheduler Error: {e}")
        await asyncio.sleep(0.5)

# ============================================================
# 📈 Auto-Scaling Monitor (Scale Up + Draining)
# ============================================================
async def auto_scaling_monitor():
    logger.info("⚖️ Auto-Scaling Monitor started.")
    last_scale_action_time = time.time()
    surplus_duration = 0.0
    
    while True:
        await asyncio.sleep(1.0)
        now = time.time()
        
        # --- 1. Scale Up ---
        recent_valid_drops = get_recent_capacity_drops_count(6.0)
        if Z_debt > (PSI_DROP * 0.8) and recent_valid_drops > SCALE_UP_DROP_THRESHOLD:
            if now - last_scale_action_time > 6.0:
                logger.info(f"🚨 [Scale Up] Z={Z_debt:.2f}, Drops={recent_valid_drops}")
                await node_mgr.scale_up_one_node()
                last_scale_action_time = now
                surplus_duration = 0.0
                continue
                
        # --- 2. Scale Down (Draining) ---
        active_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
        if len(active_urls) > 1:
            v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_urls if node_mgr.nodes[u].get("metrics")]
            
            total_pending = len(global_request_list)
            total_free_slots = sum(
                max(0, n.capacity_merged - n.running_batch) if n.mode == "merge" 
                else max(0, n.capacity_unmerged_base - n.running_batch - len(n.active_loras))
                for n in v_nodes
            )
            
            is_surplus = (total_free_slots - total_pending) >= SCALE_DOWN_SURPLUS_THRESHOLD
            
            if is_surplus:
                surplus_duration += 1.0
            else:
                surplus_duration = 0.0
                
            if surplus_duration >= 6.0 and (now - last_scale_action_time > 6.0):
                if await node_mgr.trigger_drain_best_node(v_nodes):
                    last_scale_action_time = now
                    surplus_duration = 0.0

# ============================================================
# API Routes
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    
    logger.info(f"🔄 Registering with EFO at {EFO_URL}...")
    for _ in range(5):
        try:
            resp = await client.post(
                f"{EFO_URL}/register_cluster", 
                json={"cluster_name": CLUSTER_NAME, "control_node_url": MY_URL},
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                LORA_METADATA_TABLE.update(data.get("metadata", {}))
                logger.info(f"✅ Successfully registered to EFO. Loaded {len(LORA_METADATA_TABLE)} LoRA metadata entries.")
                break
            else:
                logger.warning(f"⚠️ EFO rejected registration: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ EFO unreachable ({e}), retrying in 2s...")
        await asyncio.sleep(2.0)
    else:
        logger.error("❌ Failed to register with EFO after maximum retries. Running without metadata!")

    asyncio.create_task(scheduler_loop())
    asyncio.create_task(auto_scaling_monitor()) 
    
    async def cleanup_streams():
        while True:
            now = time.time()
            expired = [rid for rid, d in stream_queues.items() if now - d["ts"] > 120]
            for rid in expired: stream_queues.pop(rid, None)
            await asyncio.sleep(60)
    asyncio.create_task(cleanup_streams())
    
    yield
    await client.aclose()

app = FastAPI(title=f"Control Node SP2 ({CLUSTER_NAME})", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 256

class UpdateLorasRequest(BaseModel):
    loras: List[str]

class UpdateRoutingRequest(BaseModel):
    routing_table: Dict[str, Any]

class HaltRequest(BaseModel):
    cluster_name: str

@app.post("/update_local_loras")
async def update_local_loras(req: UpdateLorasRequest):
    global LOCAL_AVAILABLE_LORAS
    LOCAL_AVAILABLE_LORAS = set(req.loras)
    logger.info(f"🔄 Updated LOCAL_AVAILABLE_LORAS from EFO: {LOCAL_AVAILABLE_LORAS}")
    return {"status": "ok"}

@app.post("/send_request")
async def send_request(req: AddRequest):
    global Z_debt, can_accept_offload
    rid = str(uuid.uuid4())
    stream_queues[rid] = {"q": asyncio.Queue(), "ts": time.time()}
    meta = LORA_METADATA_TABLE.get(req.adapter_id)
    is_local = (meta and meta.get("type") == "local")

    if not meta or (is_local and meta.get("cluster") != CLUSTER_ID):
        await handle_offload_or_drop(rid, is_local, 999.0, Z_debt, stream_queues[rid]["q"], force_drop_reason="Sovereignty Violation")
        return {"request_id": rid}

    async with stats_lock:
        lora_request_stats[req.adapter_id] += 1

    valid_subs = [req.adapter_id] + [s for s in meta.get("substitutes", []) if s in LOCAL_AVAILABLE_LORAS]
    actual_valid = [s for s in valid_subs if s in LOCAL_AVAILABLE_LORAS]
    
    # 無法處理的 LoRA (本地完全沒有支援檔案) - 不觸發 Halt 廣播，直接 force_offload=True
    if not actual_valid:
        await handle_offload_or_drop(rid, is_local, 999.0, Z_debt, stream_queues[rid]["q"], force_offload=True)
        return {"request_id": rid}

    active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
    nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]
    
    best_ttft = 999.0
    global_pending = len(global_request_list)
    if nodes:
        for aid in actual_valid:
            ttft = predict_cluster_ttft(nodes, aid, global_pending)
            if ttft < best_ttft: best_ttft = ttft
    
    s_eff = 1.0 if best_ttft <= T_MAX else -1.0

    # 當發現自己容量真的不足時，如果是可處理的 LoRA，代表系統正在擁塞，觸發廣播
    if s_eff < 0 and can_accept_offload:
        can_accept_offload = False
        asyncio.create_task(broadcast_halt())

    async with z_lock:
        if s_eff < 0:
            handled = await handle_offload_or_drop(rid, is_local, best_ttft, Z_debt, stream_queues[rid]["q"])
            if handled: return {"request_id": rid}
            Z_debt = max(0.0, Z_debt + 1.0 - EPSILON)
        else:
            Z_debt = max(0.0, Z_debt - EPSILON)

    req_obj = {
        "request_id": rid, "prompt": req.prompt, "adapter_id": req.adapter_id,
        "original_aid": req.adapter_id, "max_new_tokens": req.max_new_tokens, "arrival_time": time.time()
    }
    request_queues[req.adapter_id].append(req_obj)
    global_request_list.append({"request_id": rid, "original_aid": req.adapter_id, "arrival_time": req_obj["arrival_time"]})
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str):
    if request_id not in stream_queues: raise HTTPException(404)
    q = stream_queues[request_id]["q"]
    stream_queues[request_id]["ts"] = time.time() 
    async def event_generator():
        yield "event: open\ndata: connected\n\n"
        while True:
            data = await q.get()
            if data is None: yield "event: end\ndata: [DONE]\n\n"; break
            if isinstance(data, dict) and data.get("type") == "error":
                yield f"event: error\ndata: {json.dumps(data)}\n\n"; break
            yield f"data: {data}\n\n"
        stream_queues.pop(request_id, None)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/register_node")
async def register(data: dict): 
    node_mgr.register_node(data["url"])
    return {"ok": True}

@app.get("/status")
async def status(): 
    return {
        "active_nodes": len([n for n, d in node_mgr.nodes.items() if d["status"] == "active"]),
        "draining_nodes": len([n for n, d in node_mgr.nodes.items() if d["status"] == "draining"]),
        "standby_nodes": len([n for n, d in node_mgr.nodes.items() if d["status"] == "standby"]),
        "z_debt": round(Z_debt, 2),
        "global_pending": len(global_request_list),
        "recent_drops": get_recent_capacity_drops_count(),
        "can_accept_offload": can_accept_offload
    }

@app.get("/fetch_adapter/{adapter_id}")
async def fetch(adapter_id: str): 
    path = os.path.join(LORA_PATH, "LoRA_1", "adapter_model.safetensors")
    return FileResponse(path)

@app.get("/pop_lora_stats")
async def pop_lora_stats():
    global lora_request_stats
    async with stats_lock:
        current_stats = dict(lora_request_stats)
        lora_request_stats.clear()
    return {"cluster": CLUSTER_NAME, "stats": current_stats}

# ============================================================
# SP2: Offloading APIs
# ============================================================
@app.get("/offload_status")
async def get_offload_status():
    """供 EFO 每 10 秒抓取一次的狀態，包含可用槽位容量與各 LoRA 的部署狀態"""
    active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
    v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]

    total_pending = len(global_request_list)
    total_free_slots = sum(
        max(0, n.capacity_merged - n.running_batch) if n.mode == "merge" 
        else max(0, n.capacity_unmerged_base - n.running_batch - len(n.active_loras))
        for n in v_nodes
    )

    # 如果 Z_debt 已經來到危險水位，或者本地已經觸發過 capacity halt，主動回報 budget 為 0
    if Z_debt >= PSI_DROP * 0.9 or not can_accept_offload:
        budget = 0
    else:
        budget = total_free_slots - total_pending

    merged_loras = set()
    loaded_loras = set()

    for n in v_nodes:
        if n.mode == "merge" and n.merged_adapter:
            merged_loras.add(n.merged_adapter)
        elif n.mode == "unmerge":
            loaded_loras.update(n.loaded_adapters)

    unloaded_loras = LOCAL_AVAILABLE_LORAS - merged_loras - loaded_loras

    return {
        "budget": budget,
        "lora_status": {
            "merged": list(merged_loras),
            "loaded": list(loaded_loras),
            "unloaded": list(unloaded_loras)
        }
    }

@app.post("/update_global_routing")
async def update_global_routing(req: UpdateRoutingRequest):
    """接收由 EFO 統整過後廣播的全域路由表，並重置本地的 Halt 狀態"""
    global global_routing_table, can_accept_offload
    global_routing_table = req.routing_table
    my_status = global_routing_table.get(CLUSTER_NAME, {})
    if my_status.get("budget", 0) > 0:
        can_accept_offload = True
    else:
        can_accept_offload = False
    logger.info(f"🗺️ Received new global routing table for {len(global_routing_table)} clusters.")
    return {"status": "ok"}

@app.post("/emergency_halt")
async def emergency_halt(req: HaltRequest):
    """接收來自其他 Control Node 的緊急 Halt 廣播，立刻阻斷對該 Cluster 的卸載"""
    if req.cluster_name in global_routing_table:
        global_routing_table[req.cluster_name]["budget"] = 0
        logger.warning(f"🛑 [Routing] Received emergency HALT from {req.cluster_name}. Budget set to 0.")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port)
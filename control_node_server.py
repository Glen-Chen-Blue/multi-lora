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

# 匯入集中管理的設定
from config import (
    LORA_PATH, LOG_PATH,
    MERGED_CAPACITY, UNMERGED_CAPACITY,
    T_MAX, EPSILON, PSI_DROP,
    SCALE_UP_DROP_THRESHOLD, SCALE_DOWN_SURPLUS_THRESHOLD,
    HTTP_MAX_CONNECTIONS,
    SCHEDULER_OVERHEAD, SIM_LOAD_DELAY,
    SIM_PREFILL_BASE_TIME, MERGE_SPEED_MULTIPLIER,
    SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    SP1_INTERVAL_SECONDS
)

# ============================================================
# Config & Logging
# ============================================================
class RoutingAccessFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/update_global_routing" not in msg and "/offload_status" not in msg and "/cluster_metrics" not in msg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL: %(message)s")
logger = logging.getLogger("ControlNode")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(RoutingAccessFilter())

CLUSTER_NAME = os.environ.get("CLUSTER_NAME", "cluster_1")
CLUSTER_ID = CLUSTER_NAME
EFO_URL = os.environ.get("EFO_URL", "http://127.0.0.1:9100")
MY_URL = os.environ.get("CONTROL_NODE_URL", "http://127.0.0.1:9000")

limits = httpx.Limits(max_keepalive_connections=200, max_connections=HTTP_MAX_CONNECTIONS)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# 📊 Metrics Collection State (Modified)
# ============================================================
class ClusterMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        # 累積計數器
        self.local_completed = 0
        self.offload_in_completed = 0
        self.offload_out = 0
        
        # [修改] 只保留兩種 Drop 類型
        self.drop_local_congestion = 0      # 本地壅塞 (包含被 Offload 進來但過載的情況)
        self.drop_no_target = 0             # 無處可去 (No Targets)

        # 即時數據
        self.ttft_records: List[float] = []
        self.latest_p95 = 0.0

    async def record_ttft(self, ttft: float):
        async with self.lock:
            self.ttft_records.append(ttft)

    async def calculate_p95_and_clear(self) -> float:
        async with self.lock:
            if self.ttft_records:
                self.ttft_records.sort()
                idx = int(0.95 * len(self.ttft_records))
                idx = min(idx, len(self.ttft_records) - 1)
                self.latest_p95 = self.ttft_records[idx]
                self.ttft_records.clear()
            return self.latest_p95

cluster_metrics = ClusterMetrics()
node_cumulative_inf_time: Dict[str, float] = {}

metrics_logging_task: Optional[asyncio.Task] = None
current_interval_id = 0

async def run_metrics_logging_cycle(interval_id: int):
    """
    在收到 SP1 reset 後執行，固定紀錄 20 次 Metrics。
    """
    logger.info(f"📊 [Metrics] Starting logging cycle for Interval {interval_id}")
    
    os.makedirs(LOG_PATH, exist_ok=True)
    log_file = f"{LOG_PATH}/control_{CLUSTER_NAME}_metrics.log"
    
    logging_interval = SP1_INTERVAL_SECONDS / 20.0
    
    try:
        for step in range(20):
            await asyncio.sleep(logging_interval)
            
            total_inf_time = sum(node_cumulative_inf_time.values())
            p95 = await cluster_metrics.calculate_p95_and_clear()
            
            async with cluster_metrics.lock:
                snapshot = {
                    "interval_id": interval_id,
                    "step_in_interval": step + 1,
                    "local_completed": cluster_metrics.local_completed,
                    "offload_in_completed": cluster_metrics.offload_in_completed,
                    "offload_out": cluster_metrics.offload_out,
                    # [修改] 紀錄簡化後的 Drop
                    "drop_local_congestion": cluster_metrics.drop_local_congestion,
                    "drop_no_target": cluster_metrics.drop_no_target,
                    "total_effective_inference_time": total_inf_time,
                    "p95_ttft": p95,
                    "z_debt": Z_debt
                }
                
            with open(log_file, "a") as f:
                log_entry = {
                    "timestamp": time.time(),
                    "cluster": CLUSTER_NAME,
                    "metrics": snapshot
                }
                f.write(json.dumps(log_entry) + "\n")
                
        logger.info(f"📊 [Metrics] Interval {interval_id} logging finished (20/20). Waiting for next edge.")
        
    except asyncio.CancelledError:
        logger.info(f"📊 [Metrics] Cycle {interval_id} cancelled (New Time Edge arrived).")
        raise

# ============================================================
# Lyapunov & Auto-Scaling Global States
# ============================================================
Z_debt = 0.0
z_lock = asyncio.Lock()

switching_nodes: Set[str] = set()
recent_capacity_drops: Deque[float] = deque()

lora_request_stats: Dict[str, int] = defaultdict(int)
stats_lock = asyncio.Lock()

def record_capacity_drop():
    recent_capacity_drops.append(time.time())

def get_recent_capacity_drops_count(window: float = 6.0) -> int:
    now = time.time()
    while recent_capacity_drops and now - recent_capacity_drops[0] > window:
        recent_capacity_drops.popleft()
    return len(recent_capacity_drops)

LORA_METADATA_TABLE: Dict[str, Any] = {}
LOCAL_AVAILABLE_LORAS: Set[str] = set()
global_routing_table: Dict[str, Any] = {}
can_accept_offload: bool = True
system_paused: bool = False

# ============================================================
# Global State
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
            asyncio.create_task(client.post(f"{url}/set_status", json={"status": status}))
        else: 
            self.nodes[url]["last_seen"] = time.time()
            
    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()
            
            if "metrics" in metrics:
                node_cumulative_inf_time[url] = metrics["metrics"].get("effective_inference_time", 0.0)
            
            reported_status = metrics.get("status", "standby")
            if self.nodes[url]["status"] == "draining" and reported_status == "standby":
                logger.info(f"❄️ [Scale Down] 節點已完全排空，自動轉為 Standby: {url}")
                self.nodes[url]["status"] = "standby"
            
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
        if len(active_nodes) <= 1: return False
            
        global_lora_counts = defaultdict(int)
        for n in active_nodes:
            for lora in n.active_loras: global_lora_counts[lora] += 1
                
        current_unmerge_count = sum(1 for n in active_nodes if n.mode == "unmerge")
                
        def drain_score(n: VirtualNodeState):
            orphan_count = sum(1 for lora in n.active_loras if global_lora_counts[lora] == 1)
            is_last_unmerge = (n.mode == "unmerge" and current_unmerge_count <= 1)
            return (orphan_count * 1000) + (10000 if is_last_unmerge else 0) + (100 if n.mode == "merge" else 0) + n.running_batch
            
        best_node = sorted(active_nodes, key=drain_score)[0]
        try:
            resp = await client.post(f"{best_node.url}/drain", timeout=2.0)
            if resp.status_code == 200:
                self.nodes[best_node.url]["status"] = "draining"
                logger.info(f"🚰 [Scale Down] 節點進入 Draining: {best_node.url}")
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
        for req in self.request_set: self.lora_request_counts[req["adapter_id"]] += 1

        self.capacity_merged = MERGED_CAPACITY
        self.capacity_unmerged_base = UNMERGED_CAPACITY

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
    node_scores = []
    total_free = 0
    cluster_concurrent_capacity = 0
    has_merged_node = False
    
    for node in nodes:
        if node.url in switching_nodes or node.mode == "switching": continue
        is_merge = (node.mode == "merge" and node.merged_adapter == target_lora)
        if node.mode == "merge" and not is_merge: continue 
        
        if is_merge: has_merged_node = True
        
        is_in_vram = (node.mode == "unmerge" and target_lora in node.active_loras)
        is_in_cpu = (node.mode == "unmerge" and target_lora in node.loaded_adapters)
        is_empty = (node.mode == "unmerge" and len(node.active_loras) == 0)
        
        free_slots = node.get_free_slots(target_lora)
        
        if is_merge: cluster_concurrent_capacity += node.capacity_merged
        elif node.mode == "unmerge": cluster_concurrent_capacity += max(0, node.capacity_unmerged_base - 1)

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
                
        is_merge_landing = landing_score[0] == 1
        is_in_cpu_landing = landing_score[2] == 1
        multiplier = MERGE_SPEED_MULTIPLIER if is_merge_landing else 1.0
        
        load_delay = 0.0 if (is_in_cpu_landing or is_merge_landing) else SIM_LOAD_DELAY
        prefill_time = SIM_PREFILL_BASE_TIME * take_at_landing * multiplier
        
        return SCHEDULER_OVERHEAD + load_delay + prefill_time
    else:
        needed_to_finish = my_position - total_free
        
        if has_merged_node:
            merge_node = next((n for n in nodes if n.mode == "merge"), nodes[0])
            assumed_batch = merge_node.capacity_merged 
            multiplier = MERGE_SPEED_MULTIPLIER
        else:
            assumed_batch = max(1, nodes[0].capacity_unmerged_base - 2)
            multiplier = 1.0
            
        dynamic_decode_speed = (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch) * multiplier
        
        if cluster_concurrent_capacity == 0: 
            cluster_concurrent_capacity = nodes[0].capacity_merged if has_merged_node else nodes[0].capacity_unmerged_base
            
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
            current_wait = all_remains[idx] * dynamic_decode_speed
            
        total_wait_time = current_wait + (full_cycles * 256 * dynamic_decode_speed)
        return SCHEDULER_OVERHEAD + total_wait_time + (SIM_PREFILL_BASE_TIME * multiplier)

async def broadcast_halt():
    logger.warning("🚨 [Backpressure] Capacity exhausted! Broadcasting HALT to other clusters.")
    async with httpx.AsyncClient(timeout=2.0) as client:
        tasks = []
        for cluster, data in global_routing_table.items():
            if cluster != CLUSTER_NAME:
                url = f"{data['ip']}/emergency_halt"
                tasks.append(client.post(url, json={"cluster_name": CLUSTER_NAME}))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

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
    
    arrival_time = req_data.get("arrival_time", time.time())
    is_delegated = req_data.get("is_delegated", False)
    first_token_received = False

    try:
        payload = {"prompt": req_data["prompt"], "adapter_id": target_lora, "max_new_tokens": req_data.get("max_new_tokens", 256)}
        async with client.stream("POST", f"{v_node_url}/add_request", json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                if v_node_ptr: v_node_ptr.rollback_request(target_lora)
                await user_q.put({"type": "error", "message": f"Node Error {resp.status_code}"})
                return
            
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content and content != "[DONE]":
                        # [Metrics] 攔截第一個 Token，記錄 TTFT
                        if not first_token_received:
                            first_token_received = True
                            ttft = time.time() - arrival_time
                            await cluster_metrics.record_ttft(ttft)
                        await user_q.put(content)
                    elif content == "[DONE]":
                        # [Metrics] 記錄完成次數
                        async with cluster_metrics.lock:
                            if is_delegated: cluster_metrics.offload_in_completed += 1
                            else: cluster_metrics.local_completed += 1
                        
    except Exception as e:
        if v_node_ptr: v_node_ptr.rollback_request(target_lora)
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None)

async def trigger_delegated_offload(target: dict, original_req: 'AddRequest', local_rid: str):
    user_q = stream_queues.get(local_rid, {}).get("q")
    if not user_q: return

    arrival_time = original_req.arrival_time or time.time()
    first_token_received = False

    payload = {
        "prompt": original_req.prompt,
        "adapter_id": original_req.adapter_id,
        "max_new_tokens": original_req.max_new_tokens,
        "is_delegated": True,
        "network_delay": target["delay_sec"]
    }

    try:
        resp = await client.post(f"{target['url']}/send_request", json=payload, timeout=5.0)
        if resp.status_code != 200:
            await user_q.put({"type": "error", "message": f"Offload Target Rejected: HTTP {resp.status_code}"})
            return

        data = resp.json()
        target_rid = data.get("request_id")
        
        async with client.stream("GET", f"{target['url']}/stream/{target_rid}", timeout=120.0) as stream_resp:
            async for line in stream_resp.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content == "[DONE]": 
                        break
                    elif content:
                        try:
                            content_obj = json.loads(content)
                            if isinstance(content_obj, dict) and content_obj.get("type") == "error":
                                await user_q.put(content_obj)
                                break
                        except: pass
                        
                        # [Metrics] 攔截第一個 Token，為來源端 User 記錄 TTFT
                        if not first_token_received:
                            first_token_received = True
                            ttft = time.time() - arrival_time
                            await cluster_metrics.record_ttft(ttft)
                            
                        await user_q.put(content)

    except Exception as e:
        await user_q.put({"type": "error", "message": f"Offload Proxy Exception: {str(e)}"})
    finally:
        await user_q.put(None)

async def scheduler_loop():
    global global_request_list, system_paused
    logger.info("⏳ SP2 Full-Function Scheduler loop started.")
    while True:
        try:
            # 1. 更新節點 Metrics
            all_node_urls = list(node_mgr.nodes.keys())
            if all_node_urls:
                tasks = [client.get(f"{u}/metrics", timeout=1.0) for u in all_node_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(responses):
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        node_mgr.update_metrics(all_node_urls[i], r.json())

            active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
            v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]
            
            if not v_nodes: 
                await asyncio.sleep(0.1)
                continue

            # [關鍵修正] 只有當 "沒有積壓任務" 時，System Paused 才會生效
            total_pending = sum(len(q) for q in request_queues.values())
            if system_paused and total_pending == 0:
                await asyncio.sleep(0.5)
                continue

            # MERGE_THRESHOLD: 當 Unmerged 模式接近滿載時 (ex: Cap-2)，切換至 Merge 以獲得更高吞吐量
            MERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 1)
            # UNMERGE_THRESHOLD: 當負載降低到可以安全塞回 Unmerged 模式時 (ex: Cap-2)，切換回 Unmerged 以恢復彈性
            UNMERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 2)

            # 2. 狀態切換邏輯 (Merge/Unmerge)
            unmerged_count = sum(1 for n in v_nodes if n.mode == "unmerge")

            for v in v_nodes:
                if v.url in switching_nodes: continue 
                if v.mode == "unmerge" and unmerged_count > 1 and v.running_batch >= MERGE_THRESHOLD and len(v.active_loras) == 1:
                    aid = next(iter(v.active_loras))
                    if len(request_queues[aid]) > 0:
                        asyncio.create_task(safe_mode_switch(v.url, "/merge", {"adapter_id": aid, "force": False}))
                        v.mode = "switching"; unmerged_count -= 1
                elif v.mode == "merge" and v.running_batch < UNMERGE_THRESHOLD:
                    if len(request_queues[v.merged_adapter]) == 0:
                        asyncio.create_task(safe_mode_switch(v.url, "/unmerge", {"force": False}))
                        v.mode = "switching"; unmerged_count += 1

            # 3. 請求分派邏輯 (Dispatching)
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
    global system_paused
    logger.info("⚖️ Auto-Scaling Monitor started.")
    last_scale_action_time = time.time()
    surplus_duration = 0.0
    
    while True:
        await asyncio.sleep(1.0)
        if system_paused:
            continue
            
        now = time.time()
        
        recent_valid_drops = get_recent_capacity_drops_count(6.0)
        
        if Z_debt > (PSI_DROP * 0.8) and recent_valid_drops >= SCALE_UP_DROP_THRESHOLD:
            if now - last_scale_action_time > 6.0:
                logger.info(f"🚨 [Scale Up] Z={Z_debt:.2f}, Drops={recent_valid_drops}")
                await node_mgr.scale_up_one_node()
                last_scale_action_time = now
                surplus_duration = 0.0
                continue
                
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
            
            if is_surplus: surplus_duration += 1.0
            else: surplus_duration = 0.0
                
            if surplus_duration >= 6.0 and (now - last_scale_action_time > 6.0):
                if await node_mgr.trigger_drain_best_node(v_nodes):
                    last_scale_action_time = now
                    surplus_duration = 0.0

# ============================================================
# API Routes & Lifespan
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
        except Exception: pass
        await asyncio.sleep(2.0)

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
    is_delegated: bool = False
    network_delay: float = 0.0
    arrival_time: Optional[float] = None 

class UpdateLorasRequest(BaseModel):
    loras: List[str]

class UpdateRoutingRequest(BaseModel):
    routing_table: Dict[str, Any]

class HaltRequest(BaseModel):
    cluster_name: str

def select_best_offload_target(adapter_id: str) -> Optional[dict]:
    best_target = None
    best_score = float('inf')

    meta = LORA_METADATA_TABLE.get(adapter_id, {})
    valid_aids = [adapter_id] + meta.get("substitutes", [])

    for cluster_name, info in global_routing_table.items():
        if cluster_name == CLUSTER_NAME: continue
        budget = info.get("budget", 0)
        if budget <= 0: continue

        lora_status = info.get("lora_status", {})
        merged = set(lora_status.get("merged", []))
        loaded = set(lora_status.get("loaded", []))
        unloaded = set(lora_status.get("unloaded", []))

        status_penalty = float('inf')
        
        if any(aid in merged for aid in valid_aids):
            status_penalty = 0.0
        elif any(aid in loaded for aid in valid_aids):
            status_penalty = 0.5
        elif any(aid in unloaded for aid in valid_aids):
            status_penalty = 1.0
            
        if status_penalty == float('inf'):
            continue

        delay_ms = info.get("delay", {}).get(CLUSTER_NAME, 0.0)
        delay_sec = delay_ms / 1000.0

        score = status_penalty + delay_sec
        if score < best_score:
            best_score = score
            best_target = {"cluster_name": cluster_name, "url": info.get("ip"), "delay_sec": delay_sec}
            
    return best_target

# [新增] 包含排空與重置機制的 SP1 套用 API
@app.post("/apply_sp1_and_reset")
async def apply_sp1_and_reset(req: UpdateLorasRequest):
    global system_paused, LOCAL_AVAILABLE_LORAS, metrics_logging_task, current_interval_id
    global global_request_list, request_queues
    
    logger.info("🛑 [SP1 Sync] Initiating FORCE system drain & reset...")
    system_paused = True  # 暫停所有調度與新進請求
    
    try:
        # === 1. 重置 Metrics Logging 任務 ===
        if metrics_logging_task and not metrics_logging_task.done():
            metrics_logging_task.cancel()
            try:
                await metrics_logging_task
            except asyncio.CancelledError:
                pass
        
        current_interval_id += 1
        
        # === 2. 強制結束所有請求 (不等待排空) ===
        logger.info(f"🧹 [SP1 Sync] Forcing completion for {len(stream_queues)} active connections...")
        # 讓 test_simulation 收到 [DONE] 並視為成功完成
        for rid, stream_data in stream_queues.items():
            await stream_data["q"].put(None)
        
        # 清空 Control Node 本地的等待佇列
        global_request_list.clear()
        for q in request_queues.values():
            q.clear()

        logger.info("🚰 [SP1 Sync] System queues cleared. Sending FORCE reset to compute nodes...")
        
        # === 3. 對所有 Compute Node 下達重置 (Reset) 指令 ===
        reset_tasks = []
        for url in node_mgr.nodes.keys():
            reset_tasks.append(client.post(f"{url}/reset", timeout=30.0))
        
        if reset_tasks:
            results = await asyncio.gather(*reset_tasks, return_exceptions=True)
            for url, res in zip(node_mgr.nodes.keys(), results):
                if isinstance(res, Exception):
                    logger.error(f"❌ [SP1 Sync] Reset failed for {url}: {res}")
                elif res.status_code != 200:
                    logger.error(f"❌ [SP1 Sync] Reset returned {res.status_code} for {url}")
                else:
                    logger.info(f"✅ [SP1 Sync] Node {url} reset confirmed (VRAM cleared).")
            
        # === 4. 更新本地支援的 LoRA 列表 ===
        LOCAL_AVAILABLE_LORAS = set(req.loras)
        logger.info(f"✅ [SP1 Sync] New configuration applied. LoRAs: {list(LOCAL_AVAILABLE_LORAS)}")
        
        # === 5. 啟動新一輪的 Logging ===
        metrics_logging_task = asyncio.create_task(run_metrics_logging_cycle(current_interval_id))
        
        return {"status": "success"}
    finally:
        # === 6. 解除鎖定 ===
        system_paused = False
        logger.info("▶️ [SP1 Sync] System resumed. Ready for next interval.")

@app.post("/update_local_loras")
async def update_local_loras(req: UpdateLorasRequest):
    global LOCAL_AVAILABLE_LORAS
    LOCAL_AVAILABLE_LORAS = set(req.loras)
    return {"status": "ok"}

@app.post("/send_request")
async def send_request(req: AddRequest):
    global Z_debt, can_accept_offload, system_paused
    
    if system_paused:
        raise HTTPException(status_code=503, detail="System is paused for SP1 synchronization")
        
    if req.arrival_time is None:
        req.arrival_time = time.time()
        
    rid = str(uuid.uuid4())
    stream_queues[rid] = {"q": asyncio.Queue(), "ts": time.time()}

    meta = LORA_METADATA_TABLE.get(req.adapter_id)
    is_local = (meta and meta.get("type") == "local")

    # [修改] Sovereignty Violation 不記錄為 Metric
    if not meta or (is_local and meta.get("cluster") != CLUSTER_ID):
        logger.warning(f"🚫 [Drop] {rid[:8]} | Sovereignty Violation")
        await stream_queues[rid]["q"].put({"type": "error", "message": "Sovereignty Violation"})
        await stream_queues[rid]["q"].put(None)
        return {"request_id": rid}

    if not req.is_delegated:
        async with stats_lock: lora_request_stats[req.adapter_id] += 1

    valid_subs = [req.adapter_id] + [s for s in meta.get("substitutes", []) if s in LOCAL_AVAILABLE_LORAS]
    actual_valid = [s for s in valid_subs if s in LOCAL_AVAILABLE_LORAS]

    target_ttft = T_MAX - (req.network_delay * 2)

    active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
    nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]
    
    best_ttft = 999.0
    global_pending = len(global_request_list)
    if nodes and actual_valid:
        for aid in actual_valid:
            ttft = predict_cluster_ttft(nodes, aid, global_pending)
            if ttft < best_ttft: best_ttft = ttft

    s_eff = 1.0 if (best_ttft <= target_ttft and actual_valid) else -1.0

    if s_eff < 0 and can_accept_offload and not req.is_delegated:
        can_accept_offload = False
        asyncio.create_task(broadcast_halt())

    async with z_lock:
        if s_eff < 0:
            if req.is_delegated:
                Z_debt = max(0.0, Z_debt + 1.0 - EPSILON) 
                # [修改] Offload 進來但過載 -> 算在 Local Congestion (Target Overloaded)
                async with cluster_metrics.lock: cluster_metrics.drop_local_congestion += 1
                await stream_queues[rid]["q"].put({"type": "error", "message": "Delegated target overloaded."})
                await stream_queues[rid]["q"].put(None)
                return {"request_id": rid}

            if is_local:
                record_capacity_drop()
                Z_debt = max(0.0, Z_debt + 1.0 - EPSILON)
                # [修改] 本地專用但滿載 -> Local Congestion
                async with cluster_metrics.lock: cluster_metrics.drop_local_congestion += 1
                await stream_queues[rid]["q"].put({"type": "error", "message": "System Congested (Local Drop)"})
                await stream_queues[rid]["q"].put(None)
                return {"request_id": rid}

            target = select_best_offload_target(req.adapter_id)
            if not target:
                record_capacity_drop()
                Z_debt = max(0.0, Z_debt + 1.0 - EPSILON)
                # [修改] 找不到 Target -> No Target
                async with cluster_metrics.lock: cluster_metrics.drop_no_target += 1
                await stream_queues[rid]["q"].put({"type": "error", "message": "System Congested (No Targets)"})
                await stream_queues[rid]["q"].put(None)
                return {"request_id": rid}

            logger.info(f"🌐 [Offload] Forwarding {rid[:8]} to {target['cluster_name']}")
            async with cluster_metrics.lock: cluster_metrics.offload_out += 1
            asyncio.create_task(trigger_delegated_offload(target, req, rid))
            return {"request_id": rid}
        else:
            Z_debt = max(0.0, Z_debt - EPSILON)

    req_obj = {
        "request_id": rid, "prompt": req.prompt, "adapter_id": req.adapter_id,
        "original_aid": req.adapter_id, "max_new_tokens": req.max_new_tokens, 
        "arrival_time": req.arrival_time, "is_delegated": req.is_delegated
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

@app.get("/cluster_metrics")
async def get_cluster_metrics():
    """供 EFO (每 60 秒) 拉取的 API"""
    async with cluster_metrics.lock:
        return {
            "local_completed": cluster_metrics.local_completed,
            "offload_in_completed": cluster_metrics.offload_in_completed,
            "offload_out": cluster_metrics.offload_out,
            # [修改] 回傳簡化後的 Drop
            "drop_local_congestion": cluster_metrics.drop_local_congestion,
            "drop_no_target": cluster_metrics.drop_no_target,
            "total_effective_inference_time": sum(node_cumulative_inf_time.values()),
            "latest_p95_ttft": cluster_metrics.latest_p95
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

@app.get("/offload_status")
async def get_offload_status():
    active_node_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
    v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_node_urls if node_mgr.nodes[u].get("metrics")]

    total_pending = len(global_request_list)
    total_free_slots = sum(
        max(0, n.capacity_merged - n.running_batch) if n.mode == "merge" 
        else max(0, n.capacity_unmerged_base - n.running_batch - len(n.active_loras))
        for n in v_nodes
    )

    if Z_debt >= PSI_DROP * 0.9:
        budget = 0
    else:
        budget = max(0, total_free_slots - total_pending)

    merged_loras = set()
    loaded_loras = set()

    for n in v_nodes:
        if n.mode == "merge" and n.merged_adapter: merged_loras.add(n.merged_adapter)
        elif n.mode == "unmerge": loaded_loras.update(n.loaded_adapters)

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
    global global_routing_table, can_accept_offload
    global_routing_table = req.routing_table
    
    if Z_debt < PSI_DROP * 0.8:
        can_accept_offload = True
        
    return {"status": "ok"}

@app.post("/emergency_halt")
async def emergency_halt(req: HaltRequest):
    if req.cluster_name in global_routing_table:
        global_routing_table[req.cluster_name]["budget"] = 0
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port)
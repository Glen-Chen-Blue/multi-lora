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
limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# Lyapunov & TTFT Hyperparameters
# ============================================================
T_MAX = 5.5                 
EPSILON = 0.05               
PSI_DROP = 10.0              

Z_debt = 0.0
z_lock = asyncio.Lock()

switching_nodes: Set[str] = set()

# ============================================================
# 模擬 EFO 資訊表
# ============================================================
CLUSTER_ID = "cluster_1" 
LORA_METADATA_TABLE = {
    "LoRA_1": {"type": "global", "substitutes": ["LoRA_2", "LoRA_3"]},
    "LoRA_2": {"type": "global", "substitutes": ["LoRA_1", "LoRA_3"]},
    "LoRA_3": {"type": "global", "substitutes": ["LoRA_1", "LoRA_2"]},
    "LoRA_4": {"type": "local", "cluster": "cluster_1", "substitutes": ["LoRA_5"]},
    "LoRA_5": {"type": "local", "cluster": "cluster_1", "substitutes": ["LoRA_4"]},
    "LoRA_6": {"type": "local", "cluster": "cluster_2", "substitutes": []},
    "LoRA_7": {"type": "global", "substitutes": ["LoRA_8"]},
    "LoRA_8": {"type": "global", "substitutes": ["LoRA_7"]},
    "LoRA_9": {"type": "global", "substitutes": []},
    "LoRA_10": {"type": "global", "substitutes": []},
}
LOCAL_AVAILABLE_LORAS = {"LoRA_1", "LoRA_2", "LoRA_3", "LoRA_4", "LoRA_5", "LoRA_7", "LoRA_9"}

# ============================================================
# Global State
# ============================================================
class NodeManager:
    def __init__(self): self.nodes: Dict[str, Dict] = {}
    def register_node(self, url: str):
        if url not in self.nodes:
            self.nodes[url] = {"metrics": None, "last_seen": time.time()}
            logger.info(f"✅ Registered Node: {url}")
        else: self.nodes[url]["last_seen"] = time.time()
    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()

node_mgr = NodeManager()
request_queues: Dict[str, Deque] = defaultdict(deque)
global_request_list: List[Dict] = [] 
stream_queues: Dict[str, Dict] = {} 

# ============================================================
# 🔮 記憶體感知虛擬狀態機
# ============================================================
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

# ============================================================
# 🔮 完美 TTFT 預測 (基於全域排隊深度的多週期推演)
# ============================================================
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

# ============================================================
# 🚨 統一 Task Offloading & Drop 決策樞紐
# ============================================================
async def handle_offload_or_drop(
    rid: str, 
    is_local: bool, 
    best_ttft: float, 
    z_current: float, 
    q: asyncio.Queue, 
    force_offload: bool = False,
    force_drop_reason: str = None
) -> bool:
    """
    統一處理所有的 Drop 或 Offload。
    回傳 True 表示請求已被處理（丟棄或轉發），不需要加入本地 Queue。
    回傳 False 表示要留在本地排隊硬吞。
    """
    # 1. 絕對的 Drop 條件 (例如違反資料主權，別區的 Local LoRA)
    if force_drop_reason:
        logger.warning(f"🚫 [Drop] {rid[:8]} | Reason: {force_drop_reason}")
        await q.put({"type": "error", "message": force_drop_reason})
        await q.put(None)
        return True

    # 2. 強制卸載 (例如本地缺乏檔案的 Global LoRA)
    if force_offload:
        # TODO: 未來實作 HTTP 轉發給 EFO 或有檔案的 Cluster
        logger.warning(f"🌐 [Offload] Global LoRA {rid[:8]} | Reason: Artifact Missing")
        await q.put({"type": "error", "message": "Artifact Missing (Offload Not Implemented)"})
        await q.put(None)
        return True

    # 3. 算力不足造成的超載 (Z > PSI_DROP)
    if z_current > PSI_DROP:
        if is_local:
            # 本區的機密資料，算力不足只能捨棄
            logger.warning(f"🚫 [Drop] Local LoRA {rid[:8]} | Pred TTFT: {best_ttft:.1f}s | Z: {z_current:.2f}")
            await q.put({"type": "error", "message": "System Congested (Local Drop)"})
        else:
            # Global 資料，算力不足就丟給鄰居
            # TODO: 未來實作跨區轉發 (Task Offloading)
            logger.warning(f"🌐 [Offload] Global LoRA {rid[:8]} | Pred TTFT: {best_ttft:.1f}s | Z: {z_current:.2f}")
            await q.put({"type": "error", "message": "System Congested (Offload Not Implemented)"})
        
        await q.put(None)
        return True
        
    return False

# ============================================================
# Scheduler & Dispatch Core
# ============================================================
async def safe_mode_switch(node_url: str, endpoint: str, payload: Dict):
    if node_url in switching_nodes: return
    switching_nodes.add(node_url)
    try:
        resp = await client.post(f"{node_url}{endpoint}", json=payload, timeout=5.0)
        if resp.status_code == 200:
            logger.info(f"✅ Mode Switch {endpoint} Success: {node_url}")
        else:
            logger.warning(f"⚠️ Mode Switch {endpoint} Failed ({resp.status_code}): {node_url}")
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
            node_urls = list(node_mgr.nodes.keys())
            if node_urls:
                tasks = [client.get(f"{u}/metrics", timeout=1.0) for u in node_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(responses):
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        node_mgr.update_metrics(node_urls[i], r.json())

            v_nodes = [VirtualNodeState(u, d["metrics"]) for u, d in node_mgr.nodes.items() if d.get("metrics")]
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
                elif v.mode == "merge" and v.running_batch < 5:
                    if len(request_queues[v.merged_adapter]) == 0 and (total_pending > 0):
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
                            
                            # 🚀 5-Tier 終極派發評分
                            score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, can_take)
                            
                            if best_plan is None or score > best_plan["score"]:
                                best_plan = {
                                    "node": v,
                                    "lora": aid,
                                    "requests": candidate_reqs[:can_take],
                                    "score": score
                                }
                    
                    if best_plan:
                        node = best_plan["node"]
                        aid = best_plan["lora"]
                        reqs_to_dispatch = best_plan["requests"]
                        
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
# API Routes
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    asyncio.create_task(scheduler_loop())
    async def cleanup_streams():
        while True:
            now = time.time()
            expired = [rid for rid, d in stream_queues.items() if now - d["ts"] > 120]
            for rid in expired: stream_queues.pop(rid, None)
            await asyncio.sleep(60)
    asyncio.create_task(cleanup_streams())
    yield
    await client.aclose()

app = FastAPI(title="Control Node SP2 (Stable+Deterministic)", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 256

@app.post("/send_request")
async def send_request(req: AddRequest):
    global Z_debt
    rid = str(uuid.uuid4())
    stream_queues[rid] = {"q": asyncio.Queue(), "ts": time.time()}
    meta = LORA_METADATA_TABLE.get(req.adapter_id)
    
    is_local = (meta and meta.get("type") == "local")

    # 1. Rule 1: Data Sovereignty (資料主權審查：嚴禁處理別區的 Local LoRA)
    if not meta or (is_local and meta.get("cluster") != CLUSTER_ID):
        # 🚀 觸發強制 Drop，不列入算力債 Z_debt
        await handle_offload_or_drop(rid, is_local, 999.0, Z_debt, stream_queues[rid]["q"], force_drop_reason="Sovereignty Violation")
        return {"request_id": rid}

    # 2. 檢查本地是否有檔案 (或替代檔案)
    valid_subs = [req.adapter_id] + [s for s in meta.get("substitutes", []) if s in LOCAL_AVAILABLE_LORAS]
    actual_valid = [s for s in valid_subs if s in LOCAL_AVAILABLE_LORAS]
    
    if not actual_valid:
        # 🚀 觸發強制卸載 (Force Offload)，因為缺乏檔案而非算力不足，不列入算力債 Z_debt
        await handle_offload_or_drop(rid, is_local, 999.0, Z_debt, stream_queues[rid]["q"], force_offload=True)
        return {"request_id": rid}

    # 3. 進入真實物理算力評估 (Virtual Dry-Run)
    nodes = [VirtualNodeState(u, d["metrics"]) for u, d in node_mgr.nodes.items() if d.get("metrics")]
    best_ttft = 999.0
    global_pending = len(global_request_list)
    
    if nodes:
        for aid in actual_valid:
            ttft = predict_cluster_ttft(nodes, aid, global_pending)
            if ttft < best_ttft: 
                best_ttft = ttft
    
    s_eff = 1.0 if best_ttft <= T_MAX else -1.0

    async with z_lock:
        if s_eff < 0:
            # 算力不足，交給流控樞紐決策 (會檢查 Z_debt 是否 > PSI_DROP)
            handled = await handle_offload_or_drop(rid, is_local, best_ttft, Z_debt, stream_queues[rid]["q"])
            if handled:
                # 已被丟棄或轉發，結束生命週期
                return {"request_id": rid}
                
            # 選擇留在本地硬吞，增加延遲債務 Z(t)
            Z_debt = max(0.0, Z_debt + 1.0 - EPSILON)
        else:
            # 預期不超時，順利收下，降低延遲債務 Z(t)
            Z_debt = max(0.0, Z_debt - EPSILON)

    # 成功通過所有的審查與保護機制，正式排入全域隊列
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
async def register(data: dict): node_mgr.register_node(data["url"]); return {"ok": True}

@app.get("/status")
async def status(): 
    return {
        "active_nodes": len([n for n, d in node_mgr.nodes.items() if d.get("metrics")]), 
        "z_debt": round(Z_debt, 2),
        "global_pending": len(global_request_list)
    }

@app.get("/fetch_adapter/{adapter_id}")
async def fetch(adapter_id: str): 
    path = os.path.join(LORA_PATH, "LoRA_1", "adapter_model.safetensors")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
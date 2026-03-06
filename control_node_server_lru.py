import os
import time
import uuid
import asyncio
import httpx
import logging
import json
import random
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
    T_MAX,
    HTTP_MAX_CONNECTIONS,
    SCHEDULER_OVERHEAD, SIM_LOAD_DELAY,
    SIM_PREFILL_BASE_TIME, MERGE_SPEED_MULTIPLIER,
    SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    SP1_INTERVAL_SECONDS
)

# ============================================================
# LRU Baseline 特有設定
# ============================================================
# 模擬從網路下載一個 LoRA 到本地硬碟的延遲時間 (秒)
SIM_DOWNLOAD_DELAY = 3.0  

# ============================================================
# Config & Logging
# ============================================================
class RoutingAccessFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/update_global_routing" not in msg and "/offload_status" not in msg and "/cluster_metrics" not in msg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL(LRU): %(message)s")
logger = logging.getLogger("ControlNode_LRU")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(RoutingAccessFilter())

CLUSTER_NAME = os.environ.get("CLUSTER_NAME", "cluster_1")
CLUSTER_ID = CLUSTER_NAME
EFO_URL = os.environ.get("EFO_URL", "http://127.0.0.1:9100")
MY_URL = os.environ.get("CONTROL_NODE_URL", "http://127.0.0.1:9000")

limits = httpx.Limits(max_keepalive_connections=200, max_connections=HTTP_MAX_CONNECTIONS)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# 📊 Metrics Collection State
# ============================================================
class ClusterMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.local_completed = 0
        self.offload_in_completed = 0
        self.offload_out = 0
        
        self.drop_local_congestion = 0      # 因節點滿載或 TTFT 超標 (包含下載超時) 而 Drop
        self.drop_no_target = 0             # 無節點可用

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
                    "drop_local_congestion": cluster_metrics.drop_local_congestion,
                    "drop_no_target": cluster_metrics.drop_no_target,
                    "total_effective_inference_time": total_inf_time,
                    "p95_ttft": p95
                }
                
            with open(log_file, "a") as f:
                log_entry = {
                    "timestamp": time.time(),
                    "cluster": CLUSTER_NAME,
                    "metrics": snapshot
                }
                f.write(json.dumps(log_entry) + "\n")
        logger.info(f"📊 [Metrics] Interval {interval_id} logging finished.")
    except asyncio.CancelledError:
        pass

# ============================================================
# Global State
# ============================================================
LORA_METADATA_TABLE: Dict[str, Any] = {}
LOCAL_AVAILABLE_LORAS: Set[str] = set() # 動態快取名單
system_paused: bool = False

lora_request_stats: Dict[str, int] = defaultdict(int)
stats_lock = asyncio.Lock()

class NodeManager:
    def __init__(self): 
        self.nodes: Dict[str, Dict] = {}
        
    def register_node(self, url: str):
        if url not in self.nodes:
            self.nodes[url] = {"metrics": None, "last_seen": time.time(), "status": "active"}
            logger.info(f"✅ Registered Node: {url}")
            asyncio.create_task(client.post(f"{url}/set_status", json={"status": "active"}))
        else: 
            self.nodes[url]["last_seen"] = time.time()
            
    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()
            if "metrics" in metrics:
                node_cumulative_inf_time[url] = metrics["metrics"].get("effective_inference_time", 0.0)

node_mgr = NodeManager()
request_queues: Dict[str, Deque] = defaultdict(deque)
global_request_list: List[Dict] = [] 
stream_queues: Dict[str, Dict] = {} 

class VirtualNodeState:
    def __init__(self, url: str, metrics: Dict):
        self.url = url
        self.mode = "unmerge"
        
        load_data = metrics.get("load", {})
        lora_data = metrics.get("lora_state", {})
        
        self.running_batch = load_data.get("running_batch", 0)
        self.active_loras = set(lora_data.get("running_adapters", []))
        self.loaded_adapters = set(lora_data.get("loaded_adapters", []))
        self.request_set = metrics.get("request_set", [])
        
        self.capacity_unmerged_base = UNMERGED_CAPACITY
        self.capacity_merged = MERGED_CAPACITY 
        self.merged_adapter = None

    def get_free_slots(self, target_lora: str) -> int:
        current_cost = self.running_batch + len(self.active_loras)
        margin = self.capacity_unmerged_base - current_cost
        
        if target_lora not in self.active_loras:
            return (margin - 1) if margin >= 2 else 0 
        
        return max(0, margin)

    def commit_request(self, target_lora: str):
        self.running_batch += 1
        self.active_loras.add(target_lora)

# ============================================================
# Core Functions
# ============================================================
def predict_cluster_ttft(nodes: List[VirtualNodeState], target_lora: str, global_pending_ahead: int) -> float:
    if not nodes: return 999.0
    
    node = nodes[0]
    is_in_vram = (target_lora in node.active_loras)
    is_in_cpu = (target_lora in node.loaded_adapters)
    
    load_delay = 0.0 if (is_in_cpu or is_in_vram) else SIM_LOAD_DELAY
    assumed_batch = UNMERGED_CAPACITY - 1
    prefill_time = SIM_PREFILL_BASE_TIME
    
    dynamic_decode_speed = (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch)
    
    all_remains = [r.get("remaining_tokens", 128) for r in node.request_set]
    if not all_remains:
        decode_wait = 0.0
    else:
        avg_remain = sum(all_remains) / len(all_remains)
        decode_wait = avg_remain * dynamic_decode_speed
    
    return SCHEDULER_OVERHEAD + load_delay + decode_wait + prefill_time

async def dispatch_task(v_node_url: str, req_data: dict, v_node_ptr: Optional[VirtualNodeState] = None, target_lora: str = None):
    rid = req_data["request_id"]
    if not target_lora: target_lora = req_data["adapter_id"]
    user_data = stream_queues.get(rid)
    if not user_data: return
    user_q = user_data["q"]
    
    arrival_time = req_data.get("arrival_time", time.time())
    first_token_received = False

    try:
        payload = {"prompt": req_data["prompt"], "adapter_id": target_lora, "max_new_tokens": req_data.get("max_new_tokens", 256)}
        async with client.stream("POST", f"{v_node_url}/add_request", json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                await user_q.put({"type": "error", "message": f"Node Error {resp.status_code}"})
                return
            
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content and content != "[DONE]":
                        if not first_token_received:
                            first_token_received = True
                            ttft = time.time() - arrival_time
                            await cluster_metrics.record_ttft(ttft)
                        await user_q.put(content)
                    elif content == "[DONE]":
                        async with cluster_metrics.lock:
                            cluster_metrics.local_completed += 1
                        
    except Exception as e:
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None)

async def handle_drop(req_data: dict, reason: str):
    rid = req_data["request_id"]
    logger.warning(f"🚫 [Drop] {rid[:8]} | Reason: {reason}")
    
    async with cluster_metrics.lock:
        if "No Node" in reason or "System Full" in reason:
            cluster_metrics.drop_no_target += 1
        else:
            cluster_metrics.drop_local_congestion += 1
            
    user_data = stream_queues.get(rid)
    if user_data:
        await user_data["q"].put({"type": "error", "message": reason})
        await user_data["q"].put(None)

# ============================================================
# Scheduler Loop (LRU Cache Interception Core)
# ============================================================
async def scheduler_loop():
    global global_request_list, system_paused, LOCAL_AVAILABLE_LORAS
    
    while True:
        try:
            # 1. Update Metrics form Nodes
            all_node_urls = list(node_mgr.nodes.keys())
            if all_node_urls:
                tasks = [client.get(f"{u}/metrics", timeout=1.0) for u in all_node_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(responses):
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        node_mgr.update_metrics(all_node_urls[i], r.json())

            # 2. Snapshot Virtual Nodes
            active_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
            v_nodes = [VirtualNodeState(u, node_mgr.nodes[u]["metrics"]) for u in active_urls if node_mgr.nodes[u].get("metrics")]

            if not v_nodes or system_paused:
                await asyncio.sleep(0.5)
                continue

            # 3. Process Pending Requests
            pending_requests = list(global_request_list)
            
            if not pending_requests:
                await asyncio.sleep(0.5)
                continue
                
            for req in pending_requests:
                aid = req["adapter_id"]
                download_penalty = 0.0
                
                # ===================================================
                # 🚀 LRU 快取攔截與下載延遲計算
                # ===================================================
                if aid not in LOCAL_AVAILABLE_LORAS:
                    # 情況 A: Cache Miss (未命中) -> 需要去 EFO 下載
                    try:
                        resp = await client.post(
                            f"{EFO_URL}/fetch_and_evict_lora", 
                            json={"cluster_name": CLUSTER_NAME, "lora_id": aid},
                            timeout=2.0
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("downloaded"):
                                # 更新本地快取名單，並加上嚴重的下載時間懲罰
                                LOCAL_AVAILABLE_LORAS = set(data.get("current_cache", []))
                                download_penalty = SIM_DOWNLOAD_DELAY 
                    except Exception as e:
                        logger.error(f"LRU Fetch Error: {e}")
                else:
                    # 情況 B: Cache Hit (命中) -> 告訴 EFO 更新 LRU 時間戳
                    asyncio.create_task(client.post(
                        f"{EFO_URL}/access_lora", 
                        json={"cluster_name": CLUSTER_NAME, "lora_id": aid}
                    ))
                # ===================================================
                
                # Step A: Filter Candidates
                candidates = [n for n in v_nodes if n.get_free_slots(aid) > 0]
                
                target_node = None
                
                # Step B: Apply Strategy (Random Baseline)
                if not candidates:
                    await handle_drop(req, "System Full (No Capacity)")
                else:
                    target_node = random.choice(candidates)
                
                if target_node:
                    # Step C: TTFT Constraint Check
                    wait_time = time.time() - req["arrival_time"]
                    exec_time = predict_cluster_ttft([target_node], aid, 0)
                    
                    # ⚡ 將下載延遲加上去，計算總 TTFT
                    total_ttft = wait_time + download_penalty + exec_time
                    
                    if total_ttft > T_MAX:
                        # 若 Cache Miss 加上下載時間超過 SLO，直接 Drop 並記錄
                        await handle_drop(req, f"SLO Violation (Pred: {total_ttft:.2f}s > {T_MAX}s)")
                    else:
                        # Step D: Dispatch
                        target_node.commit_request(aid)
                        asyncio.create_task(dispatch_task(target_node.url, req, target_node))
                
                # Step E: Cleanup Queue
                if req in request_queues[aid]:
                    request_queues[aid].remove(req)
                global_request_list = [r for r in global_request_list if r["request_id"] != req["request_id"]]
        
        except Exception as e:
            logger.error(f"🔥 Scheduler Error: {e}", exc_info=True)
            
        await asyncio.sleep(0.5)

# ============================================================
# API Routes
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOCAL_AVAILABLE_LORAS
    
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
                
                # 取得 EFO 給予的初始快取白名單 (包含 Mandatory Local LoRAs)
                LOCAL_AVAILABLE_LORAS = set(data.get("initial_cache", []))
                logger.info(f"✅ Registered. Initial Cache: {list(LOCAL_AVAILABLE_LORAS)}")
                break
        except Exception: pass
        await asyncio.sleep(2.0)

    # 在背景啟動排程與清理指標
    asyncio.create_task(scheduler_loop())
    
    global metrics_logging_task
    metrics_logging_task = asyncio.create_task(run_metrics_logging_cycle(0))
    
    async def cleanup_streams():
        while True:
            now = time.time()
            expired = [rid for rid, d in stream_queues.items() if now - d["ts"] > 120]
            for rid in expired: stream_queues.pop(rid, None)
            await asyncio.sleep(60)
    asyncio.create_task(cleanup_streams())
    
    yield
    await client.aclose()

app = FastAPI(title=f"Control Node LRU Baseline ({CLUSTER_NAME})", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 256
    is_delegated: bool = False
    network_delay: float = 0.0
    arrival_time: Optional[float] = None 

# 保留這個端點供相容性使用，但在 LRU 實驗中 EFO 不會再呼叫這個強制重置
class UpdateLorasRequest(BaseModel):
    loras: List[str]

@app.post("/apply_sp1_and_reset")
async def apply_sp1_and_reset(req: UpdateLorasRequest):
    logger.info("ℹ️ [LRU Mode] Received /apply_sp1_and_reset, ignored in LRU baseline.")
    return {"status": "success", "msg": "Ignored in LRU mode"}

@app.post("/send_request")
async def send_request(req: AddRequest):
    if system_paused:
        raise HTTPException(status_code=503, detail="System Paused")
        
    if req.arrival_time is None:
        req.arrival_time = time.time()
        
    rid = str(uuid.uuid4())
    stream_queues[rid] = {"q": asyncio.Queue(), "ts": time.time()}

    meta = LORA_METADATA_TABLE.get(req.adapter_id)
    is_local = (meta and meta.get("type") == "local")

    # Sovereignty Check
    if not meta or (is_local and meta.get("cluster") != CLUSTER_ID):
        await handle_drop({"request_id": rid}, "Sovereignty Violation")
        return {"request_id": rid}

    async with stats_lock: lora_request_stats[req.adapter_id] += 1

    req_obj = {
        "request_id": rid, 
        "prompt": req.prompt, 
        "adapter_id": req.adapter_id,
        "max_new_tokens": req.max_new_tokens, 
        "arrival_time": req.arrival_time
    }
    request_queues[req.adapter_id].append(req_obj)
    global_request_list.append(req_obj)
    
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

@app.get("/cluster_metrics")
async def get_cluster_metrics():
    async with cluster_metrics.lock:
        return {
            "local_completed": cluster_metrics.local_completed,
            "offload_in_completed": cluster_metrics.offload_in_completed,
            "offload_out": cluster_metrics.offload_out,
            "drop_local_congestion": cluster_metrics.drop_local_congestion,
            "drop_no_target": cluster_metrics.drop_no_target,
            "total_effective_inference_time": sum(node_cumulative_inf_time.values()),
            "latest_p95_ttft": cluster_metrics.latest_p95
        }

# EFO compatibility endpoints
@app.get("/offload_status")
async def get_offload_status():
    return {"budget": 0, "lora_status": {"merged": [], "loaded": [], "unloaded": []}}

@app.post("/update_global_routing")
async def update_global_routing(req: dict):
    return {"status": "ok"}

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port)
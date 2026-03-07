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
    T_MAX, FIXED_OUTPUT_LEN,
    HTTP_MAX_CONNECTIONS,
    SCHEDULER_OVERHEAD, SIM_LOAD_DELAY,
    SIM_PREFILL_BASE_TIME, MERGE_SPEED_MULTIPLIER,
    SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    SP1_INTERVAL_SECONDS
)

# ============================================================
# dLoRA Baseline 特有設定
# ============================================================
# 模擬從網路下載一個 LoRA 到本地硬碟的延遲時間 (秒)
SIM_DOWNLOAD_DELAY = 3.0  

# dLoRA 動態批次切換門檻 (論文數值)
DLORA_MERGE_RIGHT_THRESHOLD = 1.0    # 進入 Merge 模式：需要單一模型佔比達 100%
DLORA_MERGE_LEFT_THRESHOLD = 0.555   # 退出 Merge 模式：該模型佔比低於 55.5% 才退回 Unmerged 共享

# ============================================================
# Config & Logging
# ============================================================
class RoutingAccessFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/update_global_routing" not in msg and "/offload_status" not in msg and "/cluster_metrics" not in msg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL(dLoRA): %(message)s")
logger = logging.getLogger("ControlNode_dLoRA")
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
        
        self.drop_local_congestion = 0      # dLoRA 理論上盡量不 Drop，但硬體完全無容量時紀錄
        self.drop_no_target = 0             

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
        self.mode = metrics.get("mode", "unmerge")
        
        load_data = metrics.get("load", {})
        lora_data = metrics.get("lora_state", {})
        
        self.running_batch = load_data.get("running_batch", 0)
        self.waiting_queue = load_data.get("waiting_queue", 0)
        self.active_loras = set(lora_data.get("running_adapters", []))
        self.loaded_adapters = set(lora_data.get("loaded_adapters", []))
        self.request_set = metrics.get("request_set", [])

# ============================================================
# Core Functions
# ============================================================
def calculate_expected_pending_time(node: VirtualNodeState, target_lora: str) -> float:
    """dLoRA 專用的 Greedy 預期等待時間估算"""
    is_in_cpu = (target_lora in node.loaded_adapters)
    load_delay = 0.0 if is_in_cpu else SIM_LOAD_DELAY
    
    # 簡單估計每個 Request 的平均執行時間
    avg_time_per_req = SIM_DECODE_BASE_TIME * FIXED_OUTPUT_LEN 
    queue_length = len(node.request_set)
    
    # 預估等待時間 = 前方排隊總耗時 + 本次模型可能需要的載入時間
    return (queue_length * avg_time_per_req) + load_delay

async def dispatch_task(v_node_url: str, req_data: dict, target_lora: str = None):
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
# dLoRA Dynamic Batching Controller (Remote Control)
# ============================================================
async def dlora_batching_controller():
    """
    全知視角的微觀排程遙控器，精準實作 dLoRA 論文的遲滯雙門檻機制 (已修復死鎖)
    """
    logger.info("🎮 [dLoRA Controller] Dynamic Batching Remote Controller Started.")
    while True:
        try:
            active_urls = [u for u, d in node_mgr.nodes.items() if d["status"] == "active"]
            for url in active_urls:
                metrics = node_mgr.nodes[url].get("metrics")
                if not metrics: continue
                
                current_mode = metrics.get("mode")
                merged_adapter = metrics.get("lora_state", {}).get("merged_adapter")
                req_set = metrics.get("request_set", [])
                
                # 修復 1: 當佇列空了，如果還在 Merge 模式，必須主動退回 Unmerge，釋放 GPU 鎖定
                if not req_set: 
                    if current_mode == "merge":
                        asyncio.create_task(client.post(f"{url}/unmerge", json={"force": True}))
                        logger.info(f"🔄 [dLoRA] Node {url} -> UNMERGE (Queue empty, releasing lock)")
                    continue
                
                # 計算當前 Queue 中的 LoRA 佔比
                counts = defaultdict(int)
                for req in req_set:
                    counts[req["adapter_id"]] += 1
                    
                l_max = max(counts, key=counts.get)
                ratio = counts[l_max] / len(req_set)
                
                # 修復 2: Exit Merge Mode 邏輯強化
                if current_mode == "merge":
                    # 如果當前 Merge 的模型，在佇列中已經連一個都沒有了，或者比例太低，必須強制退出！
                    merged_count = counts.get(merged_adapter, 0)
                    merged_ratio = merged_count / len(req_set)
                    
                    if merged_ratio < DLORA_MERGE_LEFT_THRESHOLD:
                        asyncio.create_task(client.post(f"{url}/unmerge", json={"force": True}))
                        logger.info(f"🔄 [dLoRA] Node {url} -> UNMERGE (Target '{merged_adapter}' ratio={merged_ratio:.2f} < {DLORA_MERGE_LEFT_THRESHOLD})")
                        continue # 退出後等下一個週期再決定是否要 Merge 別的

                # [Condition 1] Enter Merge Mode (Threshold = 1.0)
                if ratio >= DLORA_MERGE_RIGHT_THRESHOLD:
                    if current_mode != "merge" or merged_adapter != l_max:
                        asyncio.create_task(client.post(f"{url}/merge", json={"adapter_id": l_max, "force": True}))
                        logger.info(f"🔄 [dLoRA] Node {url} -> MERGE ({l_max}, ratio={ratio:.2f} >= {DLORA_MERGE_RIGHT_THRESHOLD})")
                        
        except Exception as e:
            logger.error(f"Batching controller error: {e}")
            
        await asyncio.sleep(0.5)

# ============================================================
# Scheduler Loop (dLoRA Greedy Dispatcher)
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
                
                # 🚀 LFU (歷史頻率) 快取攔截與下載延遲計算
                if aid not in LOCAL_AVAILABLE_LORAS:
                    try:
                        resp = await client.post(
                            f"{EFO_URL}/fetch_and_evict_lora", 
                            json={"cluster_name": CLUSTER_NAME, "lora_id": aid},
                            timeout=2.0
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("downloaded"):
                                LOCAL_AVAILABLE_LORAS = set(data.get("current_cache", []))
                                download_penalty = SIM_DOWNLOAD_DELAY 
                    except Exception as e:
                        logger.error(f"LFU Fetch Error: {e}")
                else:
                    asyncio.create_task(client.post(
                        f"{EFO_URL}/access_lora", 
                        json={"cluster_name": CLUSTER_NAME, "lora_id": aid}
                    ))
                
                # dLoRA Baseline: 純貪婪最短時間派發 (不考慮語意替換與 Lyapunov)
                best_node = None
                min_pending_time = float('inf')
                
                for node in v_nodes:
                    pt = calculate_expected_pending_time(node, aid)
                    if pt < min_pending_time:
                        min_pending_time = pt
                        best_node = node
                
                if not best_node:
                    await handle_drop(req, "System Full (No Nodes Available)")
                else:
                    # 即使超過 T_MAX (SLO) dLoRA 也是硬接，這會反映在指標上的 P95 惡化
                    total_expected_ttft = (time.time() - req["arrival_time"]) + download_penalty + min_pending_time
                    
                    if total_expected_ttft > T_MAX * 5.0: # 極端防呆，避免記憶體爆掉
                        await handle_drop(req, f"Extreme Congestion (Pred: {total_expected_ttft:.2f}s)")
                    else:
                        asyncio.create_task(dispatch_task(best_node.url, req, target_lora=aid))
                
                # 清除佇列
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
                LOCAL_AVAILABLE_LORAS = set(data.get("initial_cache", []))
                logger.info(f"✅ Registered. Initial Cache: {list(LOCAL_AVAILABLE_LORAS)}")
                break
        except Exception: pass
        await asyncio.sleep(2.0)

    # 啟動雙重背景任務： Greedy 排程器 + dLoRA 動態切換遙控器
    asyncio.create_task(scheduler_loop())
    asyncio.create_task(dlora_batching_controller())
    
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

app = FastAPI(title=f"Control Node dLoRA Baseline ({CLUSTER_NAME})", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 256
    is_delegated: bool = False
    network_delay: float = 0.0
    arrival_time: Optional[float] = None 

class UpdateLorasRequest(BaseModel):
    loras: List[str]

@app.post("/apply_sp1_and_reset")
async def apply_sp1_and_reset(req: UpdateLorasRequest):
    logger.info("ℹ️ [dLoRA Mode] Received /apply_sp1_and_reset, ignored in baseline.")
    return {"status": "success", "msg": "Ignored in dLoRA mode"}

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

    # Sovereignty Check (資料主權檢查：即便是 dLoRA 也要遵守硬性法規)
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
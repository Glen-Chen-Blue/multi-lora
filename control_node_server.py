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
T_MAX = 4.0                  
EPSILON = 0.05               
PSI_DROP = 1.0              

Z_debt = 0.0
z_lock = asyncio.Lock()

# 🚀 用於追蹤正在切換狀態的節點，防止重複發送 POST /merge
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
# 🔮 精細化 TTFT 預測與虛擬狀態
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
        self.request_set = metrics.get("request_set", [])
        self.capacity_merged = 15
        self.capacity_unmerged_base = 12

    def get_free_slots(self, target_lora: str) -> int:
        if self.mode == "merge":
            return max(0, self.capacity_merged - self.running_batch) if self.merged_adapter == target_lora else 0
        current_cost = self.running_batch + len(self.active_loras)
        margin = self.capacity_unmerged_base - current_cost
        if target_lora not in self.active_loras:
            return (margin - 1) if margin >= 2 else 0 # 1 Context slot + 1 Req slot
        return max(0, margin)

    def commit_request(self, target_lora: str, is_new_lora: bool = True):
        """🚀 修正：區分是否為新模型，避免 Bundling 時重複扣除 Context Slot"""
        self.running_batch += 1
        if is_new_lora:
            self.active_loras.add(target_lora)

    def rollback_request(self, is_new_lora: bool = True):
        self.running_batch = max(0, self.running_batch - 1)

def predict_ttft_for_node(node: VirtualNodeState, target_lora: str, total_pending: int, num_nodes: int) -> float:
    """🚀 核心改進：使用每個 Request 的剩餘 Token 進行確定性預測"""
    if node.mode == "merge" and node.merged_adapter != target_lora: return 999.0
    loaded = {node.merged_adapter} if node.mode == "merge" else node.active_loras
    is_hit = (target_lora in loaded) or any(target_lora in LORA_METADATA_TABLE.get(l, {}).get("substitutes", []) for l in loaded)
    
    multiplier = 0.8 if node.mode == "merge" else 1.0
    load_delay = 0.200 if not is_hit else 0.0 
    my_prefill = 0.050 * multiplier           
    
    # 計算前面排隊造成的 Prefill 延遲
    avg_pending_ahead = total_pending / num_nodes
    queue_prefill_delay = avg_pending_ahead * 0.050 * multiplier
    
    # 預測等待 Slot 釋放的時間
    current_free = node.get_free_slots(target_lora)
    needed_to_finish = int(max(0, avg_pending_ahead - current_free)) + 1
    
    wait_decode_time = 0.0
    if needed_to_finish > 0:
        if node.request_set:
            # 🚀 排序剩餘 Token，找出第 N 個釋放的 Slot
            sorted_remains = sorted([r.get("remaining_tokens", 256) for r in node.request_set])
            idx = min(len(sorted_remains) - 1, needed_to_finish - 1)
            # 速度預估使用 Batch+1 以求逼真
            v_token = (0.025 + 0.0012 * (node.running_batch + 1)) * multiplier
            wait_decode_time = sorted_remains[idx] * v_token
        else:
            # 節點目前無執行中請求但沒位子(通常在切換中)
            wait_decode_time = 1.0 if node.url in switching_nodes else 0.0
            
    return load_delay + my_prefill + queue_prefill_delay + wait_decode_time

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

async def dispatch_task(v_node_url: str, req_data: dict, v_node_ptr: Optional[VirtualNodeState] = None, is_new_lora: bool = True):
    rid = req_data["request_id"]
    target_lora = req_data["adapter_id"]
    user_data = stream_queues.get(rid)
    if not user_data: return
    user_q = user_data["q"]

    try:
        payload = {"prompt": req_data["prompt"], "adapter_id": target_lora, "max_new_tokens": req_data.get("max_new_tokens", 256)}
        async with client.stream("POST", f"{v_node_url}/add_request", json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                if v_node_ptr: v_node_ptr.rollback_request(is_new_lora)
                await user_q.put({"type": "error", "message": f"Node Error {resp.status_code}"})
                return
            async for line in resp.aiter_lines():
                if line.startswith("data:") and user_q:
                    content = line[len("data:"):].strip()
                    if content: await user_q.put(content)
    except Exception as e:
        if v_node_ptr: v_node_ptr.rollback_request(is_new_lora)
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None)

async def scheduler_loop():
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

            # 2. 自動模式切換 (🚀 補強：檢查該 LoRA 是否有待處理請求)
            for v in v_nodes:
                if v.url in switching_nodes: continue 
                
                if v.mode == "unmerge" and unmerged_count > 1 and v.running_batch >= 10:
                    aid = next(iter(v.active_loras)) if v.active_loras else None
                    # 🚀 修正：必須 Pending Queue > 0 才切換，防止飢餓模式切換
                    if aid and len(request_queues[aid]) > 0:
                        asyncio.create_task(safe_mode_switch(v.url, "/merge", {"adapter_id": aid, "force": False}))
                        v.mode = "switching"; unmerged_count -= 1
                elif v.mode == "merge" and v.running_batch < 5:
                    if len(request_queues[v.merged_adapter]) == 0 and (total_pending > 0):
                        asyncio.create_task(safe_mode_switch(v.url, "/unmerge", {"force": False}))
                        v.mode = "switching"; unmerged_count += 1

            # 3. 嚴謹 FIFO 公平分派
            global_request_list.sort(key=lambda x: x["arrival_time"])
            for req_meta in list(global_request_list):
                aid = req_meta["original_aid"]
                if not request_queues[aid]:
                    if req_meta in global_request_list: global_request_list.remove(req_meta)
                    continue
                
                req = request_queues[aid][0] 
                target = None

                # 1. Hot Hit (Exact/Sub)
                for v in v_nodes:
                    if v.url in switching_nodes or v.mode == "switching": continue
                    loaded = {v.merged_adapter} if v.mode=="merge" else v.active_loras
                    is_hit = (aid in loaded) or any(aid in LORA_METADATA_TABLE.get(l,{}).get("substitutes",[]) for l in loaded)
                    if is_hit:
                        actual_id = aid if aid in loaded else next(l for l in loaded if aid in LORA_METADATA_TABLE.get(l,{}).get("substitutes",[]))
                        if v.get_free_slots(actual_id) > 0:
                            req["adapter_id"] = actual_id; target = v; break
                
                # 2. Cold Start (🚀 修正：精確的 Bundling 記帳)
                if not target:
                    cands = [v for v in v_nodes if v.mode=="unmerge" and v.url not in switching_nodes and aid not in v.active_loras and v.get_free_slots(aid) > 0]
                    if cands:
                        cands.sort(key=lambda x: (len(x.active_loras), x.get_free_slots(aid)), reverse=True)
                        target = cands[0]
                        subs = [aid] + LORA_METADATA_TABLE.get(aid, {}).get("substitutes", [])
                        for s_aid in subs:
                            while request_queues[s_aid] and target.get_free_slots(aid) > 0:
                                if s_aid != aid:
                                    sub_req = request_queues[s_aid].popleft()
                                    global_request_list[:] = [x for x in global_request_list if x["request_id"] != sub_req["request_id"]]
                                    sub_req["adapter_id"] = aid
                                    # 🚀 修正：此處僅增加 batch，不應視為「新 LoRA」重複扣除 Context Slot
                                    target.commit_request(aid, is_new_lora=False)
                                    asyncio.create_task(dispatch_task(target.url, sub_req, target, is_new_lora=False))
                                else: break

                if target:
                    request_queues[aid].popleft()
                    if req_meta in global_request_list: global_request_list.remove(req_meta)
                    target.commit_request(req.get("adapter_id", aid), is_new_lora=True)
                    asyncio.create_task(dispatch_task(target.url, req, target, is_new_lora=True))

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
    
    if not meta or (meta["type"]=="local" and meta.get("cluster")!=CLUSTER_ID):
        await stream_queues[rid]["q"].put({"type": "error", "message": "Sovereignty Violation"})
        await stream_queues[rid]["q"].put(None); return {"request_id": rid}

    valid_subs = [req.adapter_id] + [s for s in meta.get("substitutes", []) if s in LOCAL_AVAILABLE_LORAS]
    actual_valid = [s for s in valid_subs if s in LOCAL_AVAILABLE_LORAS]
    if not actual_valid:
        await stream_queues[rid]["q"].put({"type": "error", "message": "LoRA unavailable locally"}); await stream_queues[rid]["q"].put(None)
        return {"request_id": rid}

    # 🚀 執行確定性預測 (使用剩餘 Token)
    nodes = [VirtualNodeState(u, d["metrics"]) for u, d in node_mgr.nodes.items() if d.get("metrics")]
    best_ttft = 999.0
    total_pending = sum(len(q) for q in request_queues.values())
    if nodes:
        for aid in actual_valid:
            for node in nodes:
                ttft = predict_ttft_for_node(node, aid, total_pending, len(nodes))
                if ttft < best_ttft: best_ttft = ttft
    
    s_eff = 1.0 if best_ttft <= T_MAX else -1.0

    async with z_lock:
        if s_eff < 0 and Z_debt > PSI_DROP:
            logger.warning(f"🚫 [Drop] {rid[:8]} | Pred TTFT: {best_ttft:.1f}s | Z: {Z_debt:.2f}")
            await stream_queues[rid]["q"].put({"type": "error", "message": "System Congested"}); await stream_queues[rid]["q"].put(None)
            return {"request_id": rid}
        if s_eff > 0: Z_debt = max(0.0, Z_debt - EPSILON)
        else: Z_debt = max(0.0, Z_debt + 1.0 - EPSILON)

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
async def status(): return {"active_nodes": len([n for n, d in node_mgr.nodes.items() if d.get("metrics")]), "z_debt": round(Z_debt, 2)}
@app.get("/fetch_adapter/{adapter_id}")
async def fetch(adapter_id: str): 
    # 正確模擬權重路徑
    path = os.path.join(LORA_PATH, "LoRA_1", "adapter_model.safetensors")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
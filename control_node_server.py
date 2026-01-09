import os
import time
import uuid
import threading
import asyncio
import httpx
from queue import Queue, Empty
from typing import Dict, List, Deque, Optional, Tuple, Any
from collections import deque, defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

# ============================================================
# Logging & Config
# ============================================================
class EndpointFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "GET /metrics" not in msg and "GET /status" not in msg

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

logger = logging.getLogger("ControlNode")

MY_NODE_URL = os.environ.get("MY_NODE_URL", "http://localhost:9000")
EFO_URL = os.environ.get("EFO_URL", "http://localhost:9090")

# 初始候選節點
ALL_CANDIDATES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",")]

# Auto-Scaling Config
SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD", "4"))     
SCALE_COOLDOWN_SEC = float(os.environ.get("SCALE_COOLDOWN_SEC", "5.0"))
MIN_NODES = 1

# Merge Trigger Config
QMIN_MULT = int(os.environ.get("QMIN_MULT", "4")) # Queue > 4 * NodeCount 時觸發 Merge

# ============================================================
# State
# ============================================================
app = FastAPI(title="Control Node")
lock = threading.Lock()

# Resource Pools
active_node_urls = []   
standby_node_urls = []  

if ALL_CANDIDATES:
    active_node_urls.append(ALL_CANDIDATES[0])        
    standby_node_urls.extend(ALL_CANDIDATES[1:])      

# Nodes Management
# nodes[url] = {
#    "metrics": dict, 
#    "mode": "NORMAL" | "DRAINING" | "MERGED", 
#    "target": adapter_id (for draining/merged),
#    "last_seen": timestamp
# }
nodes: Dict[str, Dict[str, Any]] = {}

adapter_queues = defaultdict(deque) 
stream_queues = {}
merged_assignment = {} # {adapter_id: node_url} 用於快速查找

my_allowed_adapters = []

client = httpx.AsyncClient(timeout=None)
wakeup = threading.Event()
last_scale_action_ts = 0.0

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

# ============================================================
# Helpers
# ============================================================
def _ensure_stream(rid):
    with lock:
        if rid not in stream_queues:
            stream_queues[rid] = (Queue(), time.time())

def _push_data(rid, data):
    with lock:
        if rid in stream_queues: stream_queues[rid][0].put(data)

def _finish_stream(rid):
    with lock:
        if rid in stream_queues: stream_queues[rid][0].put(None)

def _http_post_bg(url, path, payload):
    """背景發送 HTTP POST，不等待回應，避免卡住 Scheduler"""
    def run():
        try:
            httpx.post(f"{url}{path}", json=payload, timeout=5.0)
        except Exception as e:
            logger.error(f"Failed to post to {url}{path}: {e}")
    threading.Thread(target=run, daemon=True).start()

# ============================================================
# Node State Helpers
# ============================================================
def _update_node_metrics(url, metrics):
    with lock:
        if url not in nodes:
            nodes[url] = {"mode": "NORMAL", "target": None, "metrics": None, "last_seen": 0}
        nodes[url]["metrics"] = metrics
        nodes[url]["last_seen"] = time.time()

def _get_healthy_active_nodes():
    now = time.time()
    res = []
    with lock:
        for url in active_node_urls:
            info = nodes.get(url)
            if info and info.get("metrics") and (now - info["last_seen"] < 5.0):
                res.append(url)
    return res

def _node_can_accept(url, adapter_id):
    """檢查節點是否能接收該 Adapter 的請求 (考慮 Mode 和 Load)"""
    with lock:
        info = nodes.get(url)
        if not info or not info.get("metrics"): return False
        
        mode = info["mode"]
        target = info["target"]
        m = info["metrics"]
        
        # 1. Load Check
        running = m["load"]["running_batch"]
        max_bs = m["capacity"]["max_batch_size"]
        if running >= max_bs: return False
        
        # 2. Mode Check
        if mode == "DRAINING": return False
        
        # 3. Merged Check
        # 檢查實際 GPU 狀態
        actual_merged = m["lora_state"]["merged_adapter"]
        if actual_merged and actual_merged != adapter_id: return False
        
        # 檢查 Control Node 狀態
        if mode == "MERGED" and target != adapter_id: return False
        
        return True

# ============================================================
# Scaling & Merging Logic
# ============================================================
def _check_autoscaling():
    global last_scale_action_ts
    now = time.time()
    if (now - last_scale_action_ts) < SCALE_COOLDOWN_SEC: return

    with lock:
        q_total = sum(len(q) for q in adapter_queues.values())
        n_active = len(active_node_urls)
        n_standby = len(standby_node_urls)
    
    # Scale UP
    if n_standby > 0 and q_total > (SCALE_UP_THRESHOLD * n_active):
        with lock:
            if standby_node_urls:
                new_node = standby_node_urls.pop(0)
                active_node_urls.append(new_node)
                last_scale_action_ts = now
                logger.info(f"🚀 [AutoScaler] Scale UP! Activated: {new_node}")
        return

    # Scale DOWN
    if n_active > MIN_NODES and q_total == 0:
        candidate = None
        with lock:
            # 從後面找閒置且模式為 NORMAL 的節點
            for i in range(len(active_node_urls) - 1, MIN_NODES - 1, -1):
                url = active_node_urls[i]
                info = nodes.get(url)
                if info and info.get("mode") == "NORMAL" and info["metrics"].get("idle") is True:
                     candidate = url
                     del active_node_urls[i]
                     standby_node_urls.insert(0, url)
                     break
        if candidate:
            last_scale_action_ts = now
            logger.info(f"💤 [AutoScaler] Scale DOWN! Deactivated: {candidate}")

def _maybe_trigger_merge():
    healthy_urls = _get_healthy_active_nodes()
    N = len(healthy_urls)
    if N == 0: return

    with lock:
        # Snapshot queue counts
        counts = {a: len(q) for a, q in adapter_queues.items() if len(q) > 0}
    
    Q = sum(counts.values())
    if Q < (QMIN_MULT * N): return # 排隊數量不夠多，不觸發 Merge

    # 找出最熱門且尚未被 Assign 的 Adapter
    hot_candidates = []
    with lock:
        assigned_adapters = set(merged_assignment.keys())
        for a, c in counts.items():
            if a not in assigned_adapters:
                hot_candidates.append((c, a))
    
    if not hot_candidates: return
    hot_candidates.sort(reverse=True)
    _, target_adapter = hot_candidates[0]

    # 找一個適合的節點來 Drain (通常是 NORMAL 模式且 Loading 較輕)
    target_node = None
    with lock:
        best_score = -1
        for url in healthy_urls:
            info = nodes.get(url)
            if info["mode"] != "NORMAL": continue
            
            m = info["metrics"]
            # 評分標準：如果已經有載入該 adapter 最好，否則找 loading 低的
            running_adapters = m["lora_state"]["running_adapters"]
            has_adapter = 1 if target_adapter in running_adapters else 0
            load = m["load"]["running_batch"]
            
            # 分數越高越適合：有 Adapter +100，Load 越低越好
            score = (has_adapter * 100) - load
            if score > best_score:
                best_score = score
                target_node = url
        
        if target_node:
            nodes[target_node]["mode"] = "DRAINING"
            nodes[target_node]["target"] = target_adapter
            logger.info(f"🔒 [Merge] Locking {target_node} to DRAIN for {target_adapter}")

def _maybe_finalize_drains():
    # 檢查處於 DRAINING 的節點，如果 Idle 了就發送 Merge
    with lock:
        candidates = []
        for url, info in nodes.items():
            if info.get("mode") == "DRAINING" and info.get("metrics"):
                candidates.append((url, info["target"], info["metrics"]["idle"]))
    
    for url, target, is_idle in candidates:
        if is_idle:
            logger.info(f"🔗 [Merge] Node {url} is idle. Sending MERGE {target}...")
            # 同步發送以確保狀態更新
            try:
                # 先 Unmerge 再 Merge (雖然 API 支援 Force，但這樣更保險)
                httpx.post(f"{url}/unmerge", json={"force": True}, timeout=2)
                httpx.post(f"{url}/merge", json={"adapter_id": target, "force": True}, timeout=2)
                
                with lock:
                    nodes[url]["mode"] = "MERGED"
                    merged_assignment[target] = url
                logger.info(f"✅ [Merge] Node {url} is now MERGED for {target}")
            except Exception as e:
                logger.error(f"❌ [Merge] Failed to finalize merge on {url}: {e}")

def _maybe_revert_merges():
    # 如果 Merged 節點閒置且沒有對應 Queue，解除 Merge
    with lock:
        revert_list = []
        for adapter, url in merged_assignment.items():
            # 檢查 Queue 是否為空
            if len(adapter_queues[adapter]) > 0: continue
            
            info = nodes.get(url)
            if info and info.get("metrics") and info["metrics"]["idle"]:
                revert_list.append((adapter, url))
    
    for adapter, url in revert_list:
        logger.info(f"🔓 [Merge] Reverting MERGE for {adapter} on {url} (Idle)")
        _http_post_json_bg(url, "/unmerge", {"force": False})
        with lock:
            if url in nodes:
                nodes[url]["mode"] = "NORMAL"
                nodes[url]["target"] = None
            if merged_assignment.get(adapter) == url:
                del merged_assignment[adapter]

def _http_post_json_bg(url, path, json_data):
    threading.Thread(target=lambda: httpx.post(f"{url}{path}", json=json_data), daemon=True).start()


# ============================================================
# Background Tasks
# ============================================================
def efo_heartbeat():
    global my_allowed_adapters
    while True:
        try:
            r = httpx.post(f"{EFO_URL}/register_node", json={"control_node_url": MY_NODE_URL}, timeout=2)
            if r.status_code == 200:
                with lock: my_allowed_adapters = r.json().get("assigned_adapters", [])
        except: pass
        time.sleep(10)

def compute_poller():
    while True:
        with lock: targets = list(active_node_urls)
        for url in targets:
            try:
                r = httpx.get(f"{url}/metrics", timeout=1)
                _update_node_metrics(url, r.json())
            except: 
                pass # 失敗不清除狀態，避免閃爍，靠 last_seen 判斷
        wakeup.set()
        time.sleep(0.5)

def scheduler():
    while True:
        wakeup.wait()
        
        # 1. 執行策略檢查
        _check_autoscaling()
        _maybe_trigger_merge()
        _maybe_finalize_drains()
        _maybe_revert_merges()

        with lock:
            # 優先處理 Merged 的 Queue
            merged_queues = [a for a in merged_assignment.keys() if adapter_queues[a]]
            normal_queues = [a for a in adapter_queues if adapter_queues[a] and a not in merged_assignment]
            
        did_work = False

        # 2a. Dispatch Merged Queues
        for aid in merged_queues:
            target_node = None
            with lock: target_node = merged_assignment.get(aid)
            
            if target_node and _node_can_accept(target_node, aid):
                req = None
                with lock:
                    if adapter_queues[aid]: req = adapter_queues[aid].popleft()
                
                if req:
                    _dispatch_to_compute(target_node, req)
                    did_work = True

        # 2b. Dispatch Normal Queues
        for aid in normal_queues:
            # 找任意可用節點 (NORMAL 模式)
            target_node = None
            healthy = _get_healthy_active_nodes()
            
            for url in healthy:
                if _node_can_accept(url, aid):
                    target_node = url
                    break
            
            if target_node:
                req = None
                with lock:
                    if adapter_queues[aid]: req = adapter_queues[aid].popleft()
                if req:
                    _dispatch_to_compute(target_node, req)
                    did_work = True

        if not did_work:
             time.sleep(0.02)

        with lock:
            if not any(adapter_queues.values()): wakeup.clear()

def reaper():
    while True:
        time.sleep(5)
        now = time.time()
        with lock:
            to_del = [rid for rid, (q, ts) in stream_queues.items() if now - ts > 60]
            for rid in to_del: del stream_queues[rid]

threading.Thread(target=efo_heartbeat, daemon=True).start()
threading.Thread(target=compute_poller, daemon=True).start()
threading.Thread(target=scheduler, daemon=True).start()
threading.Thread(target=reaper, daemon=True).start()

# ============================================================
# Proxy Helper
# ============================================================
def _proxy_to_efo(req_id, prompt, adapter, tokens):
    async def run():
        try:
            async with client.stream("POST", f"{EFO_URL}/relay_request", 
                                     json={"prompt": prompt, "adapter_id": adapter, "max_new_tokens": tokens}) as r:
                async for line in r.aiter_lines():
                    if line: 
                        if line.startswith("data:"):
                            content = line[len("data:"):].strip()
                            if content: _push_data(req_id, content)
        except Exception as e:
            _push_data(req_id, f"[Error: {e}]")
        finally:
            _finish_stream(req_id)
    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()

def _dispatch_to_compute(url, req):
    async def run():
        try:
            async with client.stream("POST", f"{url}/add_request", 
                                     json={"prompt": req["prompt"], "adapter_id": req["adapter_id"], "max_new_tokens": req["max_new_tokens"]}) as r:
                async for line in r.aiter_lines():
                    if line and line.startswith("data:"):
                        content = line[len("data:"):].strip()
                        if content and content != "[DONE]": _push_data(req["rid"], content)
        except Exception: pass
        finally:
            _finish_stream(req["rid"])
    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()

# ============================================================
# API
# ============================================================
@app.post("/send_request")
def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    _ensure_stream(rid)
    
    is_local = False
    with lock:
        if not my_allowed_adapters or req.adapter_id in my_allowed_adapters:
            is_local = True
    
    if is_local:
        with lock:
            adapter_queues[req.adapter_id].append({
                "rid": rid, "prompt": req.prompt, "adapter_id": req.adapter_id, "max_new_tokens": req.max_new_tokens
            })
            wakeup.set()
    else:
        _proxy_to_efo(rid, req.prompt, req.adapter_id, req.max_new_tokens)
        
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str, request: Request):
    with lock: item = stream_queues.get(request_id)
    if not item: raise HTTPException(404, "Not found")
    q, _ = item
    
    async def gen():
        yield "event: open\ndata: ok\n\n"
        while True:
            if await request.is_disconnected(): break
            try:
                data = q.get_nowait()
                if data is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                yield f"data: {data}\n\n"
            except Empty:
                await asyncio.sleep(0.01)
        with lock: stream_queues.pop(request_id, None)
    
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/status")
def status():
    # 方便 Debug
    with lock:
        return {
            "node_type": "CONTROL_NODE",
            "allowed": my_allowed_adapters,
            "active_nodes": active_node_urls,
            "merged": merged_assignment,
            "queues": {k: len(v) for k, v in adapter_queues.items()},
            "node_details": {u: {"mode": i.get("mode"), "target": i.get("target"), "load": i.get("metrics", {}).get("load")} for u, i in nodes.items()}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
import os
import time
import uuid
import threading
import asyncio
import httpx
import json
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
app = FastAPI(title="Control Node with Affinity")
lock = threading.Lock()

# Resource Pools
active_node_urls = []   
standby_node_urls = []  

if ALL_CANDIDATES:
    active_node_urls.append(ALL_CANDIDATES[0])        
    standby_node_urls.extend(ALL_CANDIDATES[1:])      

# Nodes Management
nodes: Dict[str, Dict[str, Any]] = {}

adapter_queues = defaultdict(deque) 
stream_queues = {}
merged_assignment = {} 

my_allowed_adapters = []

# [New] Affinity & Minimal Set from EFO
affinity_table = {} 
minimal_set = []

# 注意：這裡不建立全域 AsyncClient 用於背景任務，避免 Event Loop 衝突
wakeup = threading.Event()
last_scale_action_ts = 0.0

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

class ConfigUpdate(BaseModel):
    assigned_adapters: List[str]
    affinity_table: Dict[str, List[str]]
    minimal_set: List[str]

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
    """
    [MODIFIED] 加入重試機制，解決 Compute Node 啟動較慢導致連線被拒的問題。
    """
    def run():
        # 嘗試 30 次，每次間隔 2 秒，共等待 60 秒
        max_retries = 30
        for i in range(max_retries):
            try:
                r = httpx.post(f"{url}{path}", json=payload, timeout=5.0)
                if r.status_code == 200:
                    logger.info(f"✅ Successfully synced to {url}")
                    return
                else:
                    logger.warning(f"⚠️ Sync to {url} returned {r.status_code}. Retrying...")
            except Exception as e:
                # 只有在前幾次失敗時印出 Log，避免洗版
                if i < 3 or i % 10 == 0:
                    logger.info(f"⏳ Waiting for {url} to become available... (Attempt {i+1}/{max_retries})")
            
            time.sleep(2.0)
        
        logger.error(f"❌ Failed to post to {url}{path} after {max_retries} retries.")

    threading.Thread(target=run, daemon=True).start()

def sync_compute_nodes_adapters():
    """
    [NEW] 將當前的 my_allowed_adapters 同步給所有活躍的 Compute Nodes
    """
    logger.info(f"🔄 Syncing adapters to Compute Nodes: {my_allowed_adapters}")
    with lock:
        targets = list(active_node_urls)
    
    for url in targets:
        _http_post_bg(url, "/sync_adapters", {"adapters": my_allowed_adapters})

# ============================================================
# Node State Helpers (Affinity Aware)
# ============================================================
def _update_node_metrics(url, metrics):
    with lock:
        if url not in nodes:
            nodes[url] = {"mode": "NORMAL", "target": None, "metrics": None, "last_seen": 0, "merged_at": 0}
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
    """
    擴展原有的檢查邏輯，加入語意親和力判斷 (Fuzzy Matching)。
    """
    with lock:
        info = nodes.get(url)
        if not info or not info.get("metrics"): return False
        
        mode = info["mode"]
        target = info["target"]
        m = info["metrics"]
        
        running = m["load"]["running_batch"]
        max_bs = m["capacity"]["max_batch_size"]
        if running >= max_bs: return False
        
        if mode == "DRAINING": return False
        
        actual_merged = m["lora_state"]["merged_adapter"]
        
        # 0. 如果節點被鎖定在 MERGED 模式但還沒完成 Merge，檢查 Target
        if mode == "MERGED" and target != adapter_id:
             substitutes = affinity_table.get(adapter_id, [])
             if target not in substitutes:
                 return False

        # 1. 精確匹配 (Exact Match)
        if actual_merged == adapter_id: return True
        
        # 2. 語意親和力匹配 (Fuzzy Match)
        substitutes = affinity_table.get(adapter_id, [])
        if actual_merged and (actual_merged in substitutes):
            return True
            
        # 3. 節點未 Merge 且為 Normal 模式 (可以自由加載)
        if not actual_merged and mode == "NORMAL":
             return True
        
        return False

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
    
    if n_standby > 0 and q_total > (SCALE_UP_THRESHOLD * n_active):
        with lock:
            if standby_node_urls:
                new_node = standby_node_urls.pop(0)
                active_node_urls.append(new_node)
                last_scale_action_ts = now
                logger.info(f"🚀 [AutoScaler] Scale UP! Activated: {new_node}")
                
                # [ADDED] 新節點加入時，立刻同步 Adapter 允許清單
                _http_post_bg(new_node, "/sync_adapters", {"adapters": my_allowed_adapters})
        return

    if n_active > MIN_NODES and q_total == 0:
        candidate = None
        with lock:
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
        # 因為我們在 send_request 就已經合併了 ID，這裡的 counts 就是已經歸類過的
        counts = {a: len(q) for a, q in adapter_queues.items() if len(q) > 0}
    
    Q = sum(counts.values())
    if Q < (QMIN_MULT * N): return 

    demand_threshold = Q / N

    hot_candidates = []
    with lock:
        assigned_adapters = set(merged_assignment.keys())
        for a, c in counts.items():
            if a not in assigned_adapters:
                if c > demand_threshold:
                    hot_candidates.append((c, a))
    
    if not hot_candidates: return
    hot_candidates.sort(reverse=True)
    _, target_adapter = hot_candidates[0]

    target_node = None
    with lock:
        best_score = -1
        for url in healthy_urls:
            info = nodes.get(url)
            if info["mode"] != "NORMAL": continue
            
            m = info["metrics"]
            running_adapters = m["lora_state"]["running_adapters"]
            has_adapter = 1 if target_adapter in running_adapters else 0
            load = m["load"]["running_batch"]
            
            score = (has_adapter * 100) - load
            if score > best_score:
                best_score = score
                target_node = url
        
        if target_node:
            nodes[target_node]["mode"] = "DRAINING"
            nodes[target_node]["target"] = target_adapter
            logger.info(f"🔒 [Merge] Locking {target_node} to DRAIN for {target_adapter} (Queue: {counts[target_adapter]}, Threshold: {demand_threshold:.1f})")

def _maybe_finalize_drains():
    with lock:
        candidates = []
        for url, info in nodes.items():
            if info.get("mode") == "DRAINING" and info.get("metrics"):
                candidates.append((url, info["target"], info["metrics"]["idle"]))
    
    for url, target, is_idle in candidates:
        if is_idle:
            logger.info(f"🔗 [Merge] Node {url} is idle. Sending MERGE {target}...")
            try:
                httpx.post(f"{url}/unmerge", json={"force": True}, timeout=2)
                httpx.post(f"{url}/merge", json={"adapter_id": target, "force": True}, timeout=2)
                
                with lock:
                    nodes[url]["mode"] = "MERGED"
                    nodes[url]["merged_at"] = time.time()
                    merged_assignment[target] = url
                logger.info(f"✅ [Merge] Node {url} is now MERGED for {target}")
            except Exception as e:
                logger.error(f"❌ [Merge] Failed to finalize merge on {url}: {e}")

def _maybe_revert_merges():
    with lock:
        revert_list = []
        for adapter, url in merged_assignment.items():
            if len(adapter_queues[adapter]) > 0: continue
            
            info = nodes.get(url)
            if info and info.get("metrics") and info["metrics"]["idle"]:
                merged_at = info.get("merged_at", 0)
                if time.time() - merged_at < 30.0:
                    continue 

                revert_list.append((adapter, url))
    
    for adapter, url in revert_list:
        logger.info(f"🔓 [Merge] Reverting MERGE for {adapter} on {url} (Idle)")
        _http_post_json_bg(url, "/unmerge", {"force": False})
        with lock:
            if url in nodes:
                nodes[url]["mode"] = "NORMAL"
                nodes[url]["target"] = None
                nodes[url]["merged_at"] = 0
            if merged_assignment.get(adapter) == url:
                del merged_assignment[adapter]

def _http_post_json_bg(url, path, json_data):
    threading.Thread(target=lambda: httpx.post(f"{url}{path}", json=json_data), daemon=True).start()

# ============================================================
# Background Tasks
# ============================================================
def efo_heartbeat():
    """
    [Modified] Heartbeat now just pings EFO. Config comes via Push (/update_config).
    """
    # 1. Loop until registered
    while True:
        try:
            logger.info("Registering to EFO...")
            r = httpx.post(f"{EFO_URL}/register_node", json={"control_node_url": MY_NODE_URL}, timeout=5)
            if r.status_code == 200:
                logger.info("✅ Registered to EFO.")
                break
        except Exception as e:
            logger.warning(f"EFO registration failed: {e}. Retrying...")
            time.sleep(5)
    
    # 2. Simple Heartbeat
    while True:
        time.sleep(10)
        try:
            httpx.post(f"{EFO_URL}/heartbeat", json={"control_node_url": MY_NODE_URL}, timeout=2)
        except Exception:
            pass

def compute_poller():
    while True:
        with lock: targets = list(active_node_urls)
        for url in targets:
            try:
                r = httpx.get(f"{url}/metrics", timeout=1)
                _update_node_metrics(url, r.json())
            except: 
                pass 
        wakeup.set()
        time.sleep(0.5)

def scheduler():
    while True:
        wakeup.wait()
        
        _check_autoscaling()
        _maybe_trigger_merge()
        _maybe_finalize_drains()
        _maybe_revert_merges()

        with lock:
            merged_queues = [a for a in merged_assignment.keys() if adapter_queues[a]]
            normal_queues = [a for a in adapter_queues if adapter_queues[a] and a not in merged_assignment]
            
        did_work = False

        # 2a. Dispatch Merged Queues (Priority)
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
                    # 注意: 這裡已經在 send_request 做過一次 ID Rewrite
                    # 所以 req['adapter_id'] 已經是 Compute Node 擁有的 ID (例如 '1')
                    # 但如果目標節點剛好是被 Merge 在某個相容的 adapter 上 (例如也 Merge 成了 '1' 或者是 '5'?)
                    # 一般情況下直接送出即可。
                    
                    # 再次確認: 如果目標節點是被鎖定在某個 merged adapter，且與當前 req 相容，
                    # 我們要確保送出的 ID 是那個 merged ID。
                    
                    target_adapter_to_use = req["adapter_id"]
                    
                    with lock:
                        info = nodes.get(target_node)
                        if info and info.get("metrics"):
                            merged = info["metrics"]["lora_state"].get("merged_adapter")
                            if merged and merged != req["adapter_id"]:
                                substitutes = affinity_table.get(req["adapter_id"], [])
                                if merged in substitutes:
                                    logger.info(f"🔄 [Scheduler] Swapping {req['adapter_id']} -> {merged} for dispatch to {target_node}")
                                    target_adapter_to_use = merged
                    
                    req_to_send = req.copy()
                    req_to_send["adapter_id"] = target_adapter_to_use
                    
                    _dispatch_to_compute(target_node, req_to_send)
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
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{EFO_URL}/relay_request", 
                                         json={"prompt": prompt, "adapter_id": adapter, "max_new_tokens": tokens}) as r:
                    async for line in r.aiter_lines():
                        if line and line.startswith("data:"):
                            content = line[len("data:"):].rstrip("\n")
                            if content: _push_data(req_id, content)
            except Exception as e:
                _push_data(req_id, json.dumps(f"[Error: {e}]"))
            finally:
                _finish_stream(req_id)
    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()

def _dispatch_to_compute(url, req):
    async def run():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                payload = {
                    "prompt": req["prompt"], 
                    "adapter_id": req["adapter_id"],
                    "max_new_tokens": req["max_new_tokens"]
                }
                async with client.stream("POST", f"{url}/add_request", json=payload) as r:
                    if r.status_code != 200:
                        logger.error(f"Compute node {url} rejected request {req['rid']} with {r.status_code}")
                        _push_data(req["rid"], json.dumps(f"[ERROR] Compute node returned {r.status_code}"))
                        return

                    async for line in r.aiter_lines():
                        if line and line.startswith("data:"):
                            content = line[len("data:"):].rstrip("\n")
                            if content and content != "[DONE]": 
                                _push_data(req["rid"], content)
            except Exception as e:
                logger.error(f"Dispatch error to {url}: {e}")
                _push_data(req["rid"], json.dumps(f"[ERROR] Dispatch failed: {e}"))
            finally:
                _finish_stream(req["rid"])
    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()

# ============================================================
# API
# ============================================================
@app.post("/update_config")
def update_config(cfg: ConfigUpdate):
    """
    [NEW] 接收 EFO 的廣播配置
    """
    global my_allowed_adapters, affinity_table, minimal_set
    
    changed = (set(my_allowed_adapters) != set(cfg.assigned_adapters))
    
    with lock:
        my_allowed_adapters = cfg.assigned_adapters
        affinity_table = cfg.affinity_table
        minimal_set = cfg.minimal_set
    
    logger.info(f"📥 Received config update from EFO. Assigned: {len(my_allowed_adapters)} adapters.")
    
    # 如果分配的 Adapter 變了，通知 Compute Nodes 重新加載
    if changed:
        sync_compute_nodes_adapters()

    return {"status": "updated"}

@app.post("/send_request")
def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    _ensure_stream(rid)
    
    is_local = False
    # [NEW] 用來存儲最終要使用的 ID (可能是原始 ID，也可能是替代品 ID)
    final_adapter_id = req.adapter_id 

    with lock:
        # Check 1: 本地直接有 (Exact Match)
        if not my_allowed_adapters or req.adapter_id in my_allowed_adapters:
            is_local = True
        
        # Check 2: 本地有替代品 (Affinity Match in Allowed List)
        # 如果我沒有這個 Adapter，但我有它的 Expert (替代品) 且 Expert 在允許清單中 -> 我可以處理
        if not is_local:
             substitutes = affinity_table.get(req.adapter_id, [])
             for sub in substitutes:
                 if sub in my_allowed_adapters:
                     is_local = True
                     final_adapter_id = sub # [REWRITE] 改寫為替代品 ID
                     break
        
        # Check 3: Affinity Match in Merged State
        # 檢查是否有節點已經 Merge 了某個替代品
        if not is_local:
             substitutes = affinity_table.get(req.adapter_id, [])
             for url in active_node_urls:
                 info = nodes.get(url)
                 if info and info.get("metrics"):
                     merged = info["metrics"]["lora_state"]["merged_adapter"]
                     # 如果某個節點 Merge 了我的替代品，那也可以送過去
                     if merged and merged in substitutes:
                         is_local = True
                         # 注意：這裡不改寫 final_adapter_id，
                         # 因為 scheduler 會再做一次針對 Merged Node 的檢查並改寫
                         # 或者我們也可以在這裡改寫，但為了邏輯一致性，讓 scheduler 處理動態的 merged 狀態比較好
                         break
    
    if is_local:
        with lock:
            # [MODIFIED] 使用 final_adapter_id 入隊列
            # 這樣給 5 的請求就會進入 '1' 的隊列，計數會合併，Merge 也會正確觸發 '1'
            adapter_queues[final_adapter_id].append({
                "rid": rid, 
                "prompt": req.prompt, 
                "adapter_id": final_adapter_id, # [IMPORTANT] 使用改寫後的 ID
                "max_new_tokens": req.max_new_tokens
            })
            wakeup.set()
    else:
        # Proxy 邏輯維持原樣 (送原始 ID 給 EFO 重新分配)
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
    with lock:
        return {
            "node_type": "CONTROL_NODE",
            "allowed": my_allowed_adapters,
            "affinity_data": {
                "table_size": len(affinity_table),
                "minimal_set": minimal_set
            },
            "active_nodes": active_node_urls,
            "merged": merged_assignment,
            "queues": {k: len(v) for k, v in adapter_queues.items()},
            "node_details": {u: {"mode": i.get("mode"), "target": i.get("target"), "load": i.get("metrics", {}).get("load")} for u, i in nodes.items()}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
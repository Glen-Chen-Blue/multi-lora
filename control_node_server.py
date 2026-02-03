import os
import time
import uuid
import threading
import asyncio
import httpx
import json
import logging
import random
from queue import Queue, Empty
from typing import Dict, List, Deque, Optional, Any
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# ============================================================
# Experiment Flags (實驗開關)
# ============================================================
ENABLE_SEMANTIC = os.environ.get("ENABLE_SEMANTIC", "true").lower() == "true"
DISPATCH_MODE = os.environ.get("DISPATCH_MODE", "smart") 
ENABLE_AUTOSCALE = os.environ.get("ENABLE_AUTOSCALE", "true").lower() == "true"
INITIAL_NODES = os.environ.get("INITIAL_NODES", "one")

# ============================================================
# Config & Logging
# ============================================================
class EndpointFilter(logging.Filter):
    def filter(self, record): return "GET /metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ControlNode")

MY_NODE_URL = os.environ.get("MY_NODE_URL", "http://localhost:9000")
EFO_URL = os.environ.get("EFO_URL", "http://localhost:9090")
LORA_PATH = os.environ.get("LORA_PATH", "./lora_repo/control")
ALL_CANDIDATES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",")]
AREA_ID = os.environ.get("AREA_ID", "1")

SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD", "4"))     
SCALE_COOLDOWN_SEC = float(os.environ.get("SCALE_COOLDOWN_SEC", "5.0"))
MIN_NODES = 1
TTFT_THRESHOLD = 6.0  
EST_PREFILL_TIME = 0.12 
DOMINANCE_THRESHOLD = 0.6 

# ============================================================
# Node State Manager
# ============================================================
class NodeManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active_urls: List[str] = []
        self.standby_urls: List[str] = []
        self.nodes: Dict[str, Dict[str, Any]] = {} 
        
        # [實驗邏輯] 根據 INITIAL_NODES 決定初始狀態
        if ALL_CANDIDATES:
            if INITIAL_NODES == "all":
                logger.info("🧪 Experiment: Starting ALL nodes immediately.")
                self.active_urls.extend(ALL_CANDIDATES)
            else:
                self.active_urls.append(ALL_CANDIDATES[0])
                self.standby_urls.extend(ALL_CANDIDATES[1:])

        self.allowed_adapters: List[str] = []
        self.affinity_table: Dict[str, List[str]] = {}
        self.minimal_set: List[str] = []
        self.merged_assignment: Dict[str, str] = {} 
        self.lora_types: Dict[str, str] = {} 
        self.config_version: int = 0

    def update_metrics(self, url: str, metrics: Dict):
        with self.lock:
            if url not in self.nodes:
                self.nodes[url] = {"mode": "NORMAL", "target": None, "last_seen": 0, "merged_at": 0}
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()

    def set_mode(self, url: str, mode: str, target: Optional[str] = None):
        with self.lock:
            if url in self.nodes:
                if self.nodes[url]["mode"] != mode:
                    self.nodes[url]["mode"] = mode
                    self.nodes[url]["target"] = target
                    logger.info(f"🔄 Node {url} state -> {mode} (Target: {target})")

    def get_healthy_active_nodes(self) -> List[str]:
        now = time.time()
        res = []
        with self.lock:
            for url in self.active_urls:
                info = self.nodes.get(url)
                if info and info.get("metrics") and (now - info["last_seen"] < 10.0):
                    res.append(url)
        return res

    def can_node_accept(self, url: str, adapter_id: str) -> bool:
        with self.lock:
            info = self.nodes.get(url)
            if not info or not info.get("metrics"): return False
            mode = info["mode"]
            target = info["target"]
            m = info["metrics"]
            
            if m["load"]["running_batch"] >= m["capacity"]["max_batch_size"]: return False
            if mode == "SWITCHING": return False 
            
            merged_on_node = m["lora_state"]["merged_adapter"]
            
            # [實驗修改] No-Semantic 模式下，substitutes 為空
            substitutes = []
            if ENABLE_SEMANTIC:
                substitutes = self.affinity_table.get(adapter_id, [])

            if mode == "PRE_MERGE" or mode == "MERGED": 
                return (target == adapter_id or target in substitutes)

            if merged_on_node == adapter_id: return True
            if merged_on_node and merged_on_node in substitutes: return True
            if not merged_on_node: return True
            return False

# ============================================================
# Global Objects
# ============================================================
node_mgr = NodeManager()
adapter_queues: Dict[str, Deque] = defaultdict(deque)
stream_queues: Dict[str, Any] = {} 

scheduler_wakeup = threading.Event()
limits = httpx.Limits(max_keepalive_connections=1000, max_connections=2000)
client = httpx.AsyncClient(limits=limits, timeout=60.0) 
download_lock = asyncio.Lock()

# ============================================================
# Background Tasks
# ============================================================
async def sync_adapter_config(target_url: str, adapters: List[str], version_id: int) -> bool:
    payload = {"adapters": adapters, "version_id": version_id}
    for i in range(10):
        try:
            resp = await client.post(f"{target_url}/sync_adapters", json=payload, timeout=30.0)
            if resp.status_code == 200: return True
        except: pass
        await asyncio.sleep(2)
    return False

def trigger_sync_all(version_id: int):
    with node_mgr.lock:
        targets = list(node_mgr.active_urls)
        adapters = list(node_mgr.allowed_adapters)
    for url in targets:
        asyncio.create_task(sync_adapter_config(url, adapters, version_id))

async def activate_node_task(node_url: str, adapters: List[str], version_id: int):
    logger.info(f"⏳ Provisioning {node_url}...")
    if await sync_adapter_config(node_url, adapters, version_id):
        with node_mgr.lock: node_mgr.active_urls.append(node_url)
        logger.info(f"🚀 Node {node_url} is now ACTIVE.")
    else:
        with node_mgr.lock: node_mgr.standby_urls.append(node_url)

# ============================================================
# Scaling & Scheduler Logic
# ============================================================
last_scale_ts = 0.0

def auto_scaler():
    if not ENABLE_AUTOSCALE: return
    global last_scale_ts
    now = time.time()
    if now - last_scale_ts < SCALE_COOLDOWN_SEC: return

    with node_mgr.lock:
        q_total = sum(len(q) for q in adapter_queues.values())
        n_standby = len(node_mgr.standby_urls)
        n_active = len(node_mgr.active_urls)
        
        total_capacity = 0
        for url in node_mgr.active_urls:
            info = node_mgr.nodes.get(url)
            if info and info.get("metrics"):
                total_capacity += info["metrics"]["capacity"]["max_batch_size"]
        if total_capacity == 0: total_capacity = 32 
        
        threshold = total_capacity * 0.5
        
        if n_standby > 0 and q_total > threshold:
            new_node = node_mgr.standby_urls.pop(0)
            last_scale_ts = now
            logger.info(f"🚀 Scale UP: {new_node}")
            asyncio.create_task(activate_node_task(new_node, list(node_mgr.allowed_adapters), node_mgr.config_version))
            return

def check_merges_optimized():
    if not ENABLE_AUTOSCALE: return 
    healthy = node_mgr.get_healthy_active_nodes()
    if not healthy: return

    with node_mgr.lock:
        queues = {a: list(q) for a, q in adapter_queues.items()}
    
    total_reqs = sum(len(q) for q in queues.values())
    if total_reqs == 0:
        check_unmerges_optimized()
        return

    merged_adapters = set(node_mgr.merged_assignment.keys())
    candidate_adapter = None
    
    for aid, reqs in queues.items():
        if aid in merged_adapters: continue 
        q_len = len(reqs)
        dominance = q_len / total_reqs
        est_wait_time = (q_len * EST_PREFILL_TIME) / 4 
        
        if est_wait_time > TTFT_THRESHOLD or (dominance > DOMINANCE_THRESHOLD and q_len > 10):
            logger.info(f"🔥 Hotspot: {aid} (Q:{q_len})")
            candidate_adapter = aid
            break
    
    if not candidate_adapter:
        check_unmerges_optimized()
        return

    target_node = select_node_for_merge(candidate_adapter, healthy)
    if target_node:
        logger.info(f"🛡️ Strategy: MERGE {candidate_adapter} on {target_node}")
        node_mgr.set_mode(target_node, "PRE_MERGE", candidate_adapter)

def select_node_for_merge(adapter_id, healthy_nodes):
    best_node = None
    min_cost = float('inf')
    with node_mgr.lock:
        for url in healthy_nodes:
            info = node_mgr.nodes.get(url)
            if info["mode"] != "NORMAL": continue 
            m = info.get("metrics", {})
            running_list = [r["adapter_id"] for r in m.get("lora_state", {}).get("running_adapters_detail", [])]
            if not running_list: running_list = m.get("lora_state", {}).get("running_adapters", [])
            target_count = running_list.count(adapter_id)
            load = m.get("load", {}).get("running_batch", 0)
            score = target_count * 100 
            if m.get("idle", False): score += 50 
            else: score -= load 
            cost = -score
            if cost < min_cost:
                min_cost = cost
                best_node = url
    return best_node

def check_unmerges_optimized():
    to_revert = []
    with node_mgr.lock:
        for adapter, url in list(node_mgr.merged_assignment.items()):
            q_len = len(adapter_queues[adapter])
            info = node_mgr.nodes.get(url)
            if q_len == 0 and info and info.get("metrics", {}).get("idle"):
                to_revert.append(url)
    for url in to_revert:
        logger.info(f"❄️ Unmerge {url}")
        asyncio.create_task(do_unmerge_node(url))

async def process_transitions():
    tasks = []
    with node_mgr.lock:
        for url, info in list(node_mgr.nodes.items()):
            if info["mode"] == "PRE_MERGE":
                target = info["target"]
                m = info.get("metrics")
                if m:
                    running = m["lora_state"]["running_adapters"]
                    others = [x for x in running if x != target]
                    if not others:
                        node_mgr.set_mode(url, "SWITCHING", target) 
                        tasks.append(do_merge_node(url, target))
    if tasks: await asyncio.gather(*tasks)

async def do_merge_node(url: str, adapter_id: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": True})
        await client.post(f"{url}/merge", json={"adapter_id": adapter_id, "force": True})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.set_mode(url, "MERGED", adapter_id)
                node_mgr.merged_assignment[adapter_id] = url
    except: node_mgr.set_mode(url, "NORMAL", None)

async def do_unmerge_node(url: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": True})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.set_mode(url, "NORMAL", None)
                for k, v in list(node_mgr.merged_assignment.items()):
                    if v == url: del node_mgr.merged_assignment[k]
    except: pass

async def dispatch_request(url: str, req: Dict):
    try:
        async with client.stream("POST", f"{url}/add_request", json={
            "prompt": req["prompt"], "adapter_id": req["adapter_id"], "max_new_tokens": req["max_new_tokens"]
        }) as r:
            if r.status_code != 200:
                _push_stream(req["rid"], json.dumps({"type": "error", "message": f"Node Error: {r.status_code}"}))
                return
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content and content != "[DONE]": _push_stream(req["rid"], content)
    except Exception as e: _push_stream(req["rid"], json.dumps({"type": "error", "message": str(e)}))
    finally: _finish_stream(req["rid"])

def _push_stream(rid, data):
    if rid in stream_queues: stream_queues[rid][0].put(data)
def _finish_stream(rid):
    if rid in stream_queues: stream_queues[rid][0].put(None)

async def scheduler_loop():
    logger.info(f"📅 Scheduler started. Mode: {DISPATCH_MODE}")
    while True:
        await asyncio.to_thread(scheduler_wakeup.wait) 
        auto_scaler()
        check_merges_optimized() 
        await process_transitions()
        healthy_nodes = node_mgr.get_healthy_active_nodes()
        if not healthy_nodes:
            scheduler_wakeup.clear()
            await asyncio.sleep(1)
            continue

        did_work = False
        if DISPATCH_MODE == "smart":
            # Smart Dispatch Logic
            merged_map = node_mgr.merged_assignment.copy()
            for aid, dedicated_node in merged_map.items():
                if len(adapter_queues[aid]) > 0:
                    dispatched = False
                    info = node_mgr.nodes.get(dedicated_node)
                    if info and info.get("metrics"):
                        if info["metrics"]["load"]["running_batch"] < info["metrics"]["capacity"]["max_batch_size"]:
                            req = adapter_queues[aid].popleft()
                            asyncio.create_task(dispatch_request(dedicated_node, req))
                            did_work = True
                            dispatched = True
                    if not dispatched:
                        # Spillover
                        for url in healthy_nodes:
                            if url == dedicated_node: continue
                            if node_mgr.can_node_accept(url, aid):
                                req = adapter_queues[aid].popleft()
                                asyncio.create_task(dispatch_request(url, req))
                                did_work = True
                                break

            pending = [a for a, q in adapter_queues.items() if len(q) > 0]
            for aid in pending:
                if aid in merged_map and len(adapter_queues[aid]) == 0: continue
                for url in healthy_nodes:
                    if node_mgr.can_node_accept(url, aid):
                        req = adapter_queues[aid].popleft()
                        asyncio.create_task(dispatch_request(url, req))
                        did_work = True
                        break
        
        if not did_work:
            scheduler_wakeup.clear()
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0)

# ============================================================
# Background Workers
# ============================================================
async def poller_task():
    while True:
        targets = []
        with node_mgr.lock: targets = list(node_mgr.active_urls)
        for url in targets:
            try:
                r = await client.get(f"{url}/metrics", timeout=1.0)
                node_mgr.update_metrics(url, r.json())
            except: pass
        scheduler_wakeup.set()
        await asyncio.sleep(0.1)

async def heartbeat_task():
    while True:
        try: await client.post(f"{EFO_URL}/heartbeat", json={"control_node_url": MY_NODE_URL}, timeout=3.0)
        except: pass
        await asyncio.sleep(30.0)

async def reaper_task():
    while True:
        now = time.time()
        to_del = [rid for rid, (_, ts) in stream_queues.items() if now - ts > 60]
        for rid in to_del: del stream_queues[rid]
        await asyncio.sleep(10)

async def ensure_local_adapter(adapter_id: str):
    target_dir = os.path.join(LORA_PATH, adapter_id)
    target_file = os.path.join(target_dir, "adapter_model.safetensors")
    if os.path.exists(target_file): return
    async with download_lock:
        if os.path.exists(target_file): return
        os.makedirs(target_dir, exist_ok=True)
        try:
            async with client.stream("GET", f"{EFO_URL}/fetch_adapter/{adapter_id}") as resp:
                if resp.status_code == 200:
                    with open(target_file, "wb") as f:
                        async for chunk in resp.aiter_bytes(): f.write(chunk)
        except: 
            if os.path.exists(target_file): os.remove(target_file)

# ============================================================
# API
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    asyncio.create_task(scheduler_loop())
    asyncio.create_task(poller_task())
    asyncio.create_task(heartbeat_task())
    asyncio.create_task(reaper_task())
    asyncio.create_task(client.post(f"{EFO_URL}/register_node", json={"control_node_url": MY_NODE_URL, "area_id": AREA_ID}))
    yield
    await client.aclose()

app = FastAPI(title="Control Node", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

class ConfigUpdate(BaseModel):
    assigned_adapters: List[str]
    affinity_table: Dict[str, List[str]]
    minimal_set: List[str]
    lora_types: Dict[str, str] = {}
    version_id: int 

@app.post("/update_config")
async def update_config(cfg: ConfigUpdate):
    with node_mgr.lock:
        if cfg.version_id < node_mgr.config_version: return {"status": "ignored"}
        node_mgr.config_version = cfg.version_id
        node_mgr.allowed_adapters = cfg.assigned_adapters
        node_mgr.affinity_table = cfg.affinity_table
        node_mgr.minimal_set = cfg.minimal_set
        node_mgr.lora_types = cfg.lora_types
    tasks = [ensure_local_adapter(aid) for aid in cfg.assigned_adapters]
    if tasks: asyncio.create_task(asyncio.wait(tasks))
    trigger_sync_all(cfg.version_id)
    return {"status": "ok"}

async def _proxy_request(rid: str, node_url: str, req: AddRequest):
    try:
        async with client.stream("POST", f"{node_url}/add_request", json=req.dict()) as r:
            if r.status_code != 200:
                _push_stream(rid, json.dumps({"type": "error", "message": f"Proxy Error: {r.status_code}"}))
                return
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content: _push_stream(rid, content)
    except Exception as e: _push_stream(rid, json.dumps({"type": "error", "message": f"Proxy failed: {e}"}))
    finally: _finish_stream(rid)

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    stream_queues[rid] = (Queue(), time.time())
    
    # [實驗修改] Random Mode: 直接亂發
    if DISPATCH_MODE == "random":
        actives = node_mgr.get_healthy_active_nodes()
        if not actives:
             _push_stream(rid, json.dumps({"type": "error", "message": "No active nodes"}))
             _finish_stream(rid)
        else:
             target = random.choice(actives)
             logger.info(f"🎲 Random {req.adapter_id} -> {target}")
             asyncio.create_task(_proxy_request(rid, target, req))
        return {"request_id": rid}

    # Smart Mode
    target_type = node_mgr.lora_types.get(req.adapter_id, "global")
    if target_type != "global" and target_type != AREA_ID:
        _push_stream(rid, json.dumps({"type": "error", "message": "Security Error"}))
        _finish_stream(rid)
        return {"request_id": rid}

    final_id = req.adapter_id
    is_hit = False
    
    with node_mgr.lock:
        if req.adapter_id in node_mgr.allowed_adapters:
            is_hit = True
            final_id = req.adapter_id
        elif ENABLE_SEMANTIC: 
            substitutes = node_mgr.affinity_table.get(req.adapter_id, [])
            valid_subs = [s for s in substitutes if s in node_mgr.allowed_adapters]
            if valid_subs:
                final_id = valid_subs[0]
                is_hit = True
                logger.info(f"🔄 Substitute: {req.adapter_id} -> {final_id}")

    if not is_hit:
        asyncio.create_task(_proxy_request(rid, f"{EFO_URL}/relay_request", req))
        return {"request_id": rid}

    with node_mgr.lock:
        adapter_queues[final_id].append({
            "rid": rid, "prompt": req.prompt, "adapter_id": final_id, "max_new_tokens": req.max_new_tokens
        })
        
    scheduler_wakeup.set()
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str, request: Request):
    if request_id not in stream_queues: raise HTTPException(404, "Not found")
    q, _ = stream_queues[request_id]
    async def gen():
        yield "event: open\ndata: ok\n\n"
        while True:
            if await request.is_disconnected(): break
            try:
                item = q.get_nowait()
                if item is None: yield "event: end\ndata: [DONE]\n\n"; break
                yield f"data: {item}\n\n"
            except Empty: await asyncio.sleep(0.02)
        if request_id in stream_queues: del stream_queues[request_id]
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/status")
def status():
    with node_mgr.lock:
        # [實驗修正] 如果是 Random 模式 (全開)，固定回傳節點總數 (不論是否健康)
        # 這樣 Cost 曲線才不會因為高負載掉下去
        metric_nodes = len(node_mgr.active_urls)
        if INITIAL_NODES == "all":
             metric_nodes = max(metric_nodes, len(ALL_CANDIDATES))

        return {
            "node_type": "CONTROL_NODE",
            "active_nodes": metric_nodes,
            "queues": {k: len(v) for k, v in adapter_queues.items()},
        }

@app.get("/fetch_adapter/{adapter_id}")
async def fetch_adapter_for_compute(adapter_id: str):
    await ensure_local_adapter(adapter_id)
    target_file = os.path.join(LORA_PATH, adapter_id, "adapter_model.safetensors")
    if os.path.exists(target_file):
        return FileResponse(target_file, media_type="application/octet-stream", filename="adapter_model.safetensors")
    raise HTTPException(404, "Not found")

@app.post("/debug/reset")
async def debug_reset():
    """
    [Debug] Force reset all queues and propagate to compute nodes.
    Useful for immediate test teardown without waiting for cooldown.
    """
    logger.warning("🚨 SYSTEM RESET TRIGGERED! Clearing all queues...")
    
    # 1. Clear local queues
    with node_mgr.lock:
        adapter_queues.clear()
        stream_queues.clear()
    
    # 2. Propagate to all known compute nodes
    all_nodes = list(node_mgr.nodes.keys())
    
    async def call_node_reset(url):
        try:
            await client.post(f"{url}/debug/reset", timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to reset node {url}: {e}")

    if all_nodes:
        await asyncio.gather(*[call_node_reset(u) for u in all_nodes])
        
    return {"status": "system_reset_complete"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))
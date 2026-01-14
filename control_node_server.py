import os
import time
import uuid
import threading
import asyncio
import httpx
import json
import logging
from queue import Queue, Empty
from typing import Dict, List, Deque, Optional, Any, Set, Tuple 
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ============================================================
# Config & Logging
# ============================================================
class EndpointFilter(logging.Filter):
    def filter(self, record):
        return "GET /metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ControlNode")

MY_NODE_URL = os.environ.get("MY_NODE_URL", "http://localhost:9000")
EFO_URL = os.environ.get("EFO_URL", "http://localhost:9090")
ALL_CANDIDATES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",")]

SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD", "4"))     
SCALE_COOLDOWN_SEC = float(os.environ.get("SCALE_COOLDOWN_SEC", "5.0"))
QMIN_MULT = int(os.environ.get("QMIN_MULT", "4"))
MIN_NODES = 1

# ============================================================
# Node State Manager
# ============================================================
class NodeManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active_urls: List[str] = []
        self.standby_urls: List[str] = []
        self.nodes: Dict[str, Dict[str, Any]] = {} 
        
        if ALL_CANDIDATES:
            self.active_urls.append(ALL_CANDIDATES[0])
            self.standby_urls.extend(ALL_CANDIDATES[1:])

        self.allowed_adapters: List[str] = []
        self.affinity_table: Dict[str, List[str]] = {}
        self.minimal_set: List[str] = []
        self.merged_assignment: Dict[str, str] = {} 
        
        # [NEW] Config Version
        self.config_version: int = 0

    def update_metrics(self, url: str, metrics: Dict):
        with self.lock:
            if url not in self.nodes:
                self.nodes[url] = {"mode": "NORMAL", "target": None, "last_seen": 0, "merged_at": 0}
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()

    def get_healthy_active_nodes(self) -> List[str]:
        now = time.time()
        res = []
        with self.lock:
            for url in self.active_urls:
                info = self.nodes.get(url)
                if info and info.get("metrics") and (now - info["last_seen"] < 2.0):
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
            if mode == "DRAINING": return False

            merged_on_node = m["lora_state"]["merged_adapter"]
            substitutes = self.affinity_table.get(adapter_id, [])

            if mode == "MERGED":
                if target == adapter_id or target in substitutes: return True
                if merged_on_node and (merged_on_node == adapter_id or merged_on_node in substitutes): return True
                return False

            if merged_on_node == adapter_id: return True
            if merged_on_node and merged_on_node in substitutes: return True
            if not merged_on_node: return True
            
            return False

# ============================================================
# Global Objects
# ============================================================
node_mgr = NodeManager()
adapter_queues: Dict[str, Deque] = defaultdict(deque)
stream_queues: Dict[str, Tuple[Queue, float]] = {} 
scheduler_wakeup = threading.Event()
client = httpx.AsyncClient(timeout=10.0)

# ============================================================
# Background Tasks
# ============================================================
async def sync_adapter_config(target_url: str, adapters: List[str], version_id: int) -> bool:
    """背景推送 Adapter 設定，附帶版本號。回傳 True 表示成功。"""
    payload = {
        "adapters": adapters, 
        "version_id": version_id
    }
    for i in range(5):
        try:
            resp = await client.post(f"{target_url}/sync_adapters", json=payload, timeout=5.0)
            if resp.status_code == 200:
                logger.info(f"✅ Synced adapters (v{version_id}) to {target_url}")
                return True
        except Exception:
            pass
        await asyncio.sleep(2)
    logger.error(f"❌ Failed to sync adapters to {target_url}")
    return False

def trigger_sync_all(version_id: int):
    with node_mgr.lock:
        targets = list(node_mgr.active_urls)
        adapters = list(node_mgr.allowed_adapters)
    for url in targets:
        asyncio.create_task(sync_adapter_config(url, adapters, version_id))

async def activate_node_task(node_url: str, adapters: List[str], version_id: int):
    """
    [NEW] Provisioning Task:
    1. Sync adapters first.
    2. Only add to active_urls if sync succeeds.
    This prevents the scheduler from assigning tasks to a node that hasn't loaded adapters yet.
    """
    logger.info(f"⏳ Provisioning {node_url} (Syncing config v{version_id})...")
    
    # 1. 等待同步完成
    success = await sync_adapter_config(node_url, adapters, version_id)
    
    if success:
        # 2. 只有成功後，才將節點加入 active_urls
        with node_mgr.lock:
            node_mgr.active_urls.append(node_url)
        logger.info(f"🚀 Node {node_url} is now ACTIVE and ready to serve.")
    else:
        # 3. 失敗則退回 standby
        logger.warning(f"⚠️ Provisioning failed for {node_url}. Returning to Standby.")
        with node_mgr.lock:
            node_mgr.standby_urls.append(node_url)

# ============================================================
# Scaling & Scheduler Logic
# ============================================================
last_scale_ts = 0.0

def auto_scaler():
    global last_scale_ts
    now = time.time()
    if now - last_scale_ts < SCALE_COOLDOWN_SEC: return

    with node_mgr.lock:
        q_total = sum(len(q) for q in adapter_queues.values())
        n_active = len(node_mgr.active_urls)
        n_standby = len(node_mgr.standby_urls)
        
        # Scale UP Logic
        if n_standby > 0 and q_total > (SCALE_UP_THRESHOLD * n_active):
            new_node = node_mgr.standby_urls.pop(0)
            # [Modified] Do NOT add to active_urls yet.
            # We defer activation until the node is provisioned.
            
            last_scale_ts = now
            logger.info(f"🚀 Scale UP initiated: {new_node} (Provisioning...)")
            
            # Start provisioning in background
            asyncio.create_task(activate_node_task(
                new_node, 
                list(node_mgr.allowed_adapters), 
                node_mgr.config_version
            ))
            return

        # Scale DOWN Logic
        if n_active > MIN_NODES and q_total == 0:
            candidate = None
            for i in range(len(node_mgr.active_urls) - 1, MIN_NODES - 1, -1):
                url = node_mgr.active_urls[i]
                info = node_mgr.nodes.get(url)
                if info and info["mode"] == "NORMAL" and info["metrics"].get("idle"):
                    candidate = url
                    del node_mgr.active_urls[i]
                    node_mgr.standby_urls.insert(0, url)
                    break
            if candidate:
                last_scale_ts = now
                logger.info(f"💤 Scale DOWN: {candidate}")

def check_merges():
    healthy = node_mgr.get_healthy_active_nodes()
    if not healthy: return

    with node_mgr.lock:
        counts = {a: len(q) for a, q in adapter_queues.items() if len(q) > 0}
    
    total_q = sum(counts.values())
    if total_q >= (QMIN_MULT * len(healthy)):
        best_adapter, max_q = None, -1
        with node_mgr.lock:
            for a, c in counts.items():
                if a not in node_mgr.merged_assignment and c > max_q:
                    best_adapter, max_q = a, c
        
        if best_adapter and max_q > (total_q / len(healthy)):
            target_node = None
            best_score = -9999
            with node_mgr.lock:
                for url in healthy:
                    info = node_mgr.nodes.get(url)
                    if info["mode"] != "NORMAL": continue
                    m = info["metrics"]
                    has_it = 1 if best_adapter in m["lora_state"]["running_adapters"] else 0
                    load = m["load"]["running_batch"]
                    score = (has_it * 100) - load
                    if score > best_score:
                        best_score = score
                        target_node = url
            
            if target_node:
                logger.info(f"🔒 Locking {target_node} to MERGE {best_adapter}")
                with node_mgr.lock:
                    node_mgr.nodes[target_node]["mode"] = "DRAINING"
                    node_mgr.nodes[target_node]["target"] = best_adapter

    with node_mgr.lock:
        for url, info in node_mgr.nodes.items():
            if info["mode"] == "DRAINING" and info["metrics"]["idle"]:
                target = info["target"]
                logger.info(f"🔗 Executing MERGE {target} on {url}")
                asyncio.create_task(do_merge_node(url, target))
                
        to_revert = []
        for adapter, url in node_mgr.merged_assignment.items():
            if len(adapter_queues[adapter]) == 0:
                info = node_mgr.nodes.get(url)
                if info and info["metrics"]["idle"] and (time.time() - info["merged_at"] > 5):
                    to_revert.append(url)
        
        for url in to_revert:
            logger.info(f"🔓 Reverting MERGE on {url}")
            asyncio.create_task(do_unmerge_node(url))

async def do_merge_node(url: str, adapter_id: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": True})
        await client.post(f"{url}/merge", json={"adapter_id": adapter_id, "force": True})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.nodes[url]["mode"] = "MERGED"
                node_mgr.nodes[url]["merged_at"] = time.time()
                node_mgr.merged_assignment[adapter_id] = url
    except Exception as e:
        logger.error(f"Merge failed on {url}: {e}")

async def do_unmerge_node(url: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": False})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.nodes[url]["mode"] = "NORMAL"
                node_mgr.nodes[url]["target"] = None
                for k, v in list(node_mgr.merged_assignment.items()):
                    if v == url: del node_mgr.merged_assignment[k]
    except Exception:
        pass

async def dispatch_request(url: str, req: Dict):
    try:
        async with client.stream("POST", f"{url}/add_request", json={
            "prompt": req["prompt"],
            "adapter_id": req["adapter_id"],
            "max_new_tokens": req["max_new_tokens"]
        }, timeout=None) as r:
            if r.status_code != 200:
                _push_stream(req["rid"], json.dumps({"type": "error", "message": f"Compute Node Error: {r.status_code}"}))
                return

            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content and content != "[DONE]":
                        _push_stream(req["rid"], content)
    except Exception as e:
        logger.error(f"Dispatch to {url} failed: {e}")
        _push_stream(req["rid"], json.dumps({"type": "error", "message": str(e)}))
    finally:
        _finish_stream(req["rid"])

def _push_stream(rid, data):
    if rid in stream_queues:
        stream_queues[rid][0].put(data)

def _finish_stream(rid):
    if rid in stream_queues:
        stream_queues[rid][0].put(None)

async def scheduler_loop():
    logger.info("📅 Scheduler loop started.")
    while True:
        await asyncio.to_thread(scheduler_wakeup.wait) 
        
        auto_scaler()
        check_merges()
        
        healthy_nodes = node_mgr.get_healthy_active_nodes()
        if not healthy_nodes:
            scheduler_wakeup.clear()
            await asyncio.sleep(1)
            continue

        did_work = False
        with node_mgr.lock:
            pending_adapters = [a for a, q in adapter_queues.items() if len(q) > 0]
            merged_map = node_mgr.merged_assignment.copy()

        for aid in pending_adapters:
            target_node = merged_map.get(aid)
            if not target_node:
                for url in healthy_nodes:
                    if node_mgr.can_node_accept(url, aid):
                        target_node = url
                        break
            else:
                if target_node not in healthy_nodes or not node_mgr.can_node_accept(target_node, aid):
                    target_node = None

            if target_node:
                req = None
                with node_mgr.lock:
                    if adapter_queues[aid]: req = adapter_queues[aid].popleft()
                
                if req:
                    final_id = req["adapter_id"]
                    with node_mgr.lock:
                        info = node_mgr.nodes.get(target_node)
                        if info:
                            merged_id = info["metrics"]["lora_state"]["merged_adapter"]
                            if merged_id and merged_id != final_id:
                                substitutes = node_mgr.affinity_table.get(final_id, [])
                                if merged_id in substitutes: final_id = merged_id
                    
                    req_to_send = req.copy()
                    req_to_send["adapter_id"] = final_id
                    asyncio.create_task(dispatch_request(target_node, req_to_send))
                    did_work = True

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
                r = await client.get(f"{url}/metrics", timeout=0.3)
                node_mgr.update_metrics(url, r.json())
            except Exception: pass
        scheduler_wakeup.set()
        await asyncio.sleep(0.5)

async def heartbeat_task():
    while True:
        try:
            await client.post(f"{EFO_URL}/heartbeat", json={"control_node_url": MY_NODE_URL}, timeout=3.0)
        except Exception: pass
        await asyncio.sleep(30.0)

async def reaper_task():
    while True:
        now = time.time()
        to_del = [rid for rid, (_, ts) in stream_queues.items() if now - ts > 60]
        for rid in to_del: del stream_queues[rid]
        await asyncio.sleep(10)

# ============================================================
# API & App
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(scheduler_loop())
    asyncio.create_task(poller_task())
    asyncio.create_task(heartbeat_task())
    asyncio.create_task(reaper_task())
    asyncio.create_task(client.post(f"{EFO_URL}/register_node", json={"control_node_url": MY_NODE_URL}))
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
    version_id: int # [NEW]

@app.post("/update_config")
async def update_config(cfg: ConfigUpdate):
    with node_mgr.lock:
        # [NEW] Version Check
        if cfg.version_id < node_mgr.config_version:
            logger.warning(f"⚠️ Ignoring obsolete config v{cfg.version_id} (Current: v{node_mgr.config_version})")
            return {"status": "ignored", "reason": "obsolete_version"}
        
        node_mgr.config_version = cfg.version_id
        node_mgr.allowed_adapters = cfg.assigned_adapters
        node_mgr.affinity_table = cfg.affinity_table
        node_mgr.minimal_set = cfg.minimal_set
    
    logger.info(f"📥 Config updated to v{cfg.version_id}. Allowed: {len(cfg.assigned_adapters)}")
    
    # Always sync when config updates (to enforce the new list)
    trigger_sync_all(cfg.version_id)
    return {"status": "ok"}

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    stream_queues[rid] = (Queue(), time.time())
    
    is_local = False
    final_id = req.adapter_id
    
    with node_mgr.lock:
        if not node_mgr.allowed_adapters or req.adapter_id in node_mgr.allowed_adapters: is_local = True
        if not is_local:
            subs = node_mgr.affinity_table.get(req.adapter_id, [])
            for s in subs:
                if s in node_mgr.allowed_adapters:
                    is_local = True; final_id = s; break
            if not is_local:
                for url in node_mgr.active_urls:
                    info = node_mgr.nodes.get(url)
                    if info and info.get("metrics"):
                        m = info["metrics"]["lora_state"]["merged_adapter"]
                        if m and m in subs: is_local = True; break
    
    if is_local:
        with node_mgr.lock:
            adapter_queues[final_id].append({
                "rid": rid, "prompt": req.prompt, "adapter_id": final_id, "max_new_tokens": req.max_new_tokens
            })
        scheduler_wakeup.set()
    else:
        asyncio.create_task(_proxy_efo(rid, req))

    return {"request_id": rid}

async def _proxy_efo(rid, req):
    try:
        async with client.stream("POST", f"{EFO_URL}/relay_request", json=req.dict()) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content: _push_stream(rid, content)
    except Exception as e:
        _push_stream(rid, json.dumps({"type": "error", "message": str(e)}))
    finally:
        _finish_stream(rid)

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
        return {
            "node_type": "CONTROL_NODE",
            "active_nodes": len(node_mgr.active_urls),
            "queues": {k: len(v) for k, v in adapter_queues.items()},
            "merged_map": node_mgr.merged_assignment,
            "nodes": node_mgr.nodes,
            "config_version": node_mgr.config_version
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))
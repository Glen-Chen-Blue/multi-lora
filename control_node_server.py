import os
import time
import uuid
import threading
import asyncio
import httpx
import json
import logging
import shutil
from queue import Queue, Empty
from typing import Dict, List, Deque, Optional, Any, Set, Tuple 
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
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
LORA_PATH = os.environ.get("LORA_PATH", "./lora_repo/control")
ALL_CANDIDATES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",")]

SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD", "4"))     
SCALE_COOLDOWN_SEC = float(os.environ.get("SCALE_COOLDOWN_SEC", "5.0"))
MIN_NODES = 1

# Merge Strategy Config
TTFT_THRESHOLD = 6.0  # 秒 (預期排隊時間超過此值觸發 Merge)
EST_PREFILL_TIME = 0.12 # 秒 (預估處理一個請求的時間，視硬體調整)
DOMINANCE_THRESHOLD = 0.6 # 佔比門檻 (超過 60% 流量)

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
        
        self.config_version: int = 0
        self.cluster_loaded_adapters: Set[str] = set()

    def update_metrics(self, url: str, metrics: Dict):
        with self.lock:
            if url not in self.nodes:
                self.nodes[url] = {
                    "mode": "NORMAL", 
                    "target": None, 
                    "last_seen": 0, 
                    "merged_at": 0,
                    "mode_ts": time.time()
                }
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()
            self._rebuild_cluster_cache()

    def _rebuild_cluster_cache(self):
        new_set = set()
        for url in self.active_urls:
            info = self.nodes.get(url)
            if info and "metrics" in info:
                loaded = info["metrics"].get("lora_state", {}).get("loaded_adapters", [])
                new_set.update(loaded)
        self.cluster_loaded_adapters = new_set

    def set_mode(self, url: str, mode: str, target: Optional[str] = None):
        with self.lock:
            if url in self.nodes:
                if self.nodes[url]["mode"] != mode:
                    self.nodes[url]["mode"] = mode
                    self.nodes[url]["target"] = target
                    self.nodes[url]["mode_ts"] = time.time()
                    logger.info(f"🔄 Node {url} state -> {mode} (Target: {target})")

    def get_healthy_active_nodes(self) -> List[str]:
        now = time.time()
        res = []
        with self.lock:
            for url in self.active_urls:
                info = self.nodes.get(url)
                if info and info.get("metrics") and (now - info["last_seen"] < 5.0):
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
            substitutes = self.affinity_table.get(adapter_id, [])

            # [策略] PRE_MERGE 階段：只允許 Target Adapter 進入 (Gating)
            if mode == "PRE_MERGE":
                if target == adapter_id or target in substitutes: return True
                return False # 拒絕其他 Adapter，讓其自然 Drain 掉

            # [策略] MERGED 階段：只允許 Target Adapter 進入
            if mode == "MERGED":
                if target == adapter_id or target in substitutes: return True
                # 即使已經 Merge 了，理論上不該接受其他 Adapter，因為會導致 Unmerge
                return False

            # NORMAL 階段：如果有殘留的 Merge 狀態，優先匹配
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
client = httpx.AsyncClient(timeout=60.0) 
download_lock = asyncio.Lock()

# ============================================================
# Background Tasks
# ============================================================
async def sync_adapter_config(target_url: str, adapters: List[str], version_id: int) -> bool:
    payload = {"adapters": adapters, "version_id": version_id}
    for i in range(10):
        try:
            resp = await client.post(f"{target_url}/sync_adapters", json=payload, timeout=30.0)
            if resp.status_code == 200:
                logger.info(f"✅ Synced adapters (v{version_id}) to {target_url}")
                return True
        except Exception: pass
        await asyncio.sleep(2)
    return False

def trigger_sync_all(version_id: int):
    with node_mgr.lock:
        targets = list(node_mgr.active_urls)
        adapters = list(node_mgr.allowed_adapters)
    for url in targets:
        asyncio.create_task(sync_adapter_config(url, adapters, version_id))

async def activate_node_task(node_url: str, adapters: List[str], version_id: int):
    logger.info(f"⏳ Provisioning {node_url} (Syncing config v{version_id})...")
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
        
        threshold = total_capacity * 0.5
        
        if n_standby > 0 and q_total > threshold:
            new_node = node_mgr.standby_urls.pop(0)
            last_scale_ts = now
            logger.info(f"🚀 Scale UP: {new_node} (Q:{q_total} > {threshold})")
            asyncio.create_task(activate_node_task(new_node, list(node_mgr.allowed_adapters), node_mgr.config_version))
            return

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

# ============================================================
# NEW: Optimized Merge Logic
# ============================================================
def check_merges_optimized():
    healthy = node_mgr.get_healthy_active_nodes()
    if not healthy: return

    # 1. 檢視當前排隊狀況
    with node_mgr.lock:
        queues = {a: list(q) for a, q in adapter_queues.items()}
    
    total_reqs = sum(len(q) for q in queues.values())
    if total_reqs == 0:
        # 若完全無請求，檢查是否需要 Unmerge 以節省成本
        check_unmerges_optimized()
        return

    merged_adapters = set(node_mgr.merged_assignment.keys())
    candidate_adapter = None
    
    # 2. 識別熱點 (Hotspot Detection)
    # 條件：排隊導致的預期延遲 > TTFT_THRESHOLD  OR  佔比 > DOMINANCE_THRESHOLD
    for aid, reqs in queues.items():
        if aid in merged_adapters: continue # 已有專車
        
        q_len = len(reqs)
        dominance = q_len / total_reqs
        
        # 簡單估算：假設 Dynamic Mode 下單節點平均併發為 4 (Slot 限制)
        est_wait_time = (q_len * EST_PREFILL_TIME) / 4 
        
        if est_wait_time > TTFT_THRESHOLD or (dominance > DOMINANCE_THRESHOLD and q_len > 10):
            logger.info(f"🔥 Hotspot detected: {aid} (Q:{q_len}, Dom:{dominance:.2f}, EstWait:{est_wait_time:.2f}s)")
            candidate_adapter = aid
            break # 一次處理一個，避免震盪
    
    if not candidate_adapter:
        check_unmerges_optimized()
        return

    # 3. 選擇最佳節點 (Node Selection)
    target_node = select_node_for_merge(candidate_adapter, healthy)
    
    if target_node:
        logger.info(f"🛡️ Strategy: CONVERT {target_node} to Dedicated for {candidate_adapter}")
        node_mgr.set_mode(target_node, "PRE_MERGE", candidate_adapter)

def select_node_for_merge(adapter_id, healthy_nodes):
    best_node = None
    max_running_count = -1
    min_cost = float('inf')
    
    with node_mgr.lock:
        for url in healthy_nodes:
            info = node_mgr.nodes.get(url)
            if info["mode"] != "NORMAL": continue # 只選目前是 Normal 的
            
            m = info.get("metrics", {})
            running_list = [r["adapter_id"] for r in m.get("lora_state", {}).get("running_adapters_detail", [])]
            # 若 metrics 沒 detail，退回用 simple list
            if not running_list:
                running_list = m.get("lora_state", {}).get("running_adapters", [])

            # 統計該節點上有多少請求是目標 Adapter
            target_count = running_list.count(adapter_id)
            load = m.get("load", {}).get("running_batch", 0)
            is_idle = m.get("idle", False)

            # Cost Function: 
            # 優先：正在跑該 Adapter 且量最大的 (符合 "選擇最多該 LoRA 的 node")
            # 次之：閒置節點
            
            # 給予 target_count 極大權重，讓他優先被選
            score = target_count * 100 
            
            if is_idle:
                score += 50 # 閒置也不錯，可以直接用
            else:
                score -= load # 越忙扣分越多 (除非都在跑 target)
            
            # 我們要選 Score 最高的 (這裡轉換成 Cost 最小化邏輯)
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
            
            # 條件：Queue 空了 且 節點也閒置 (無 Running Request)
            # 這樣可以盡快釋放節點回歸 Normal Pool
            if q_len == 0 and info and info.get("metrics", {}).get("idle"):
                # 這裡可以加一個簡單的時間防抖動 (例如持續空閒 5 秒)
                # 為求反應速度，此處直接釋放
                to_revert.append(url)

    for url in to_revert:
        logger.info(f"❄️ Cooldown: Reverting {url} to NORMAL")
        asyncio.create_task(do_unmerge_node(url))

async def process_transitions():
    # 處理 PRE_MERGE -> MERGED 的轉換
    # 必須等到節點上只剩下 Target Adapter (Drain 完成)
    
    tasks = []
    with node_mgr.lock:
        for url, info in list(node_mgr.nodes.items()):
            if info["mode"] == "PRE_MERGE":
                target = info["target"]
                m = info.get("metrics")
                if m:
                    # 檢查 Running Queue
                    # running_adapters 是 adapter_id 的 list
                    running = m["lora_state"]["running_adapters"]
                    
                    # 檢查是否還有非 Target 的 Adapter 在跑
                    others = [x for x in running if x != target]
                    
                    if not others:
                        # 完美，只剩 Target 或空閒，可以 Merge 了
                        logger.info(f"⚡ Drained! Executing merge {target} on {url}.")
                        node_mgr.set_mode(url, "SWITCHING", target) 
                        tasks.append(do_merge_node(url, target))
                    else:
                        # 還在 Drain，等待下一輪
                        pass
    
    if tasks:
        await asyncio.gather(*tasks)

async def do_merge_node(url: str, adapter_id: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": True})
        await client.post(f"{url}/merge", json={"adapter_id": adapter_id, "force": True})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.set_mode(url, "MERGED", adapter_id)
                node_mgr.nodes[url]["merged_at"] = time.time()
                node_mgr.merged_assignment[adapter_id] = url
    except Exception:
        node_mgr.set_mode(url, "NORMAL", None)

async def do_unmerge_node(url: str):
    try:
        await client.post(f"{url}/unmerge", json={"force": True})
        with node_mgr.lock:
            if url in node_mgr.nodes:
                node_mgr.set_mode(url, "NORMAL", None)
                for k, v in list(node_mgr.merged_assignment.items()):
                    if v == url: del node_mgr.merged_assignment[k]
    except Exception: pass

async def dispatch_request(url: str, req: Dict):
    try:
        async with client.stream("POST", f"{url}/add_request", json={
            "prompt": req["prompt"],
            "adapter_id": req["adapter_id"],
            "max_new_tokens": req["max_new_tokens"]
        }, timeout=None) as r:
            if r.status_code != 200:
                _push_stream(req["rid"], json.dumps({"type": "error", "message": f"Node Error: {r.status_code}"}))
                return

            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content and content != "[DONE]":
                        _push_stream(req["rid"], content)
    except Exception as e:
        logger.error(f"Dispatch error: {e}")
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
        check_merges_optimized() # 使用新的 Merge 檢查邏輯
        await process_transitions() # 處理狀態轉換
        
        healthy_nodes = node_mgr.get_healthy_active_nodes()
        if not healthy_nodes:
            scheduler_wakeup.clear()
            await asyncio.sleep(1)
            continue

        did_work = False
        
        # 1. 優先處理 Merged Assignment (專車)
        # 這裡包含 Spillover 邏輯：如果專車滿了，允許溢出到 Normal 節點以維持 P95
        merged_map = node_mgr.merged_assignment.copy()
        
        for aid, dedicated_node in merged_map.items():
            if len(adapter_queues[aid]) > 0:
                dispatched_to_dedicated = False
                
                # 檢查專車容量
                info = node_mgr.nodes.get(dedicated_node)
                if info and info.get("metrics"):
                    cap = info["metrics"]["capacity"]["max_batch_size"]
                    load = info["metrics"]["load"]["running_batch"]
                    
                    if load < cap:
                        req = adapter_queues[aid].popleft()
                        asyncio.create_task(dispatch_request(dedicated_node, req))
                        did_work = True
                        dispatched_to_dedicated = True
                
                # 如果專車滿了，且還有其他 Normal 節點可用，執行 Spillover
                # 這樣可以避免專車單點瓶頸導致 P95 爆炸
                if not dispatched_to_dedicated:
                    # 嘗試找其他 NORMAL 節點
                    spillover_node = None
                    for url in healthy_nodes:
                        if url == dedicated_node: continue
                        if node_mgr.can_node_accept(url, aid):
                            spillover_node = url
                            break
                    
                    if spillover_node:
                        req = adapter_queues[aid].popleft()
                        asyncio.create_task(dispatch_request(spillover_node, req))
                        logger.info(f"🌊 Spillover {aid} to {spillover_node}")
                        did_work = True

        # 2. 處理剩餘請求 (Normal Dispatch)
        pending_adapters = [a for a, q in adapter_queues.items() if len(q) > 0]
        
        for aid in pending_adapters:
            # 如果這個 Adapter 已經有專車 (且在上面邏輯沒被處理掉，代表專車滿了且沒地方 Spillover)
            # 就跳過，避免重複處理
            if aid in merged_map and len(adapter_queues[aid]) == 0:
                continue

            target_node = None
            
            # 尋找可用節點
            for url in healthy_nodes:
                if node_mgr.can_node_accept(url, aid):
                    target_node = url
                    break
            
            if target_node:
                req = adapter_queues[aid].popleft()
                asyncio.create_task(dispatch_request(target_node, req))
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
        await asyncio.sleep(0.1)

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
        except Exception: 
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
    version_id: int 

@app.post("/update_config")
async def update_config(cfg: ConfigUpdate):
    with node_mgr.lock:
        if cfg.version_id < node_mgr.config_version: return {"status": "ignored"}
        node_mgr.config_version = cfg.version_id
        node_mgr.allowed_adapters = cfg.assigned_adapters
        node_mgr.affinity_table = cfg.affinity_table
        node_mgr.minimal_set = cfg.minimal_set
    
    tasks = [ensure_local_adapter(aid) for aid in cfg.assigned_adapters]
    if tasks: asyncio.create_task(asyncio.wait(tasks))
    trigger_sync_all(cfg.version_id)
    return {"status": "ok"}

async def _proxy_efo(rid, req: AddRequest):
    try:
        logger.info(f"🛰️ Offloading Request {rid} (Adapter {req.adapter_id}) to EFO")
        async with client.stream("POST", f"{EFO_URL}/relay_request", json=req.dict()) as r:
            if r.status_code != 200:
                _push_stream(rid, json.dumps({"type": "error", "message": f"EFO Error: {r.status_code}"}))
                return
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].rstrip("\n")
                    if content: _push_stream(rid, content)
    except Exception as e:
        _push_stream(rid, json.dumps({"type": "error", "message": f"Offload failed: {e}"}))
    finally:
        _finish_stream(rid)

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    stream_queues[rid] = (Queue(), time.time())
    
    final_id = req.adapter_id
    is_allowed = False
    
    with node_mgr.lock:
        if req.adapter_id in node_mgr.allowed_adapters:
            is_allowed = True
            final_id = req.adapter_id
        else:
            substitutes = node_mgr.affinity_table.get(req.adapter_id, [])
            valid_subs = [s for s in substitutes if s in node_mgr.allowed_adapters]
            if valid_subs:
                is_allowed = True
                final_id = valid_subs[0]
    
    if not is_allowed:
        asyncio.create_task(_proxy_efo(rid, req))
        return {"request_id": rid}

    selected_id = final_id
    # 這裡的邏輯移到 scheduler 處理，這裡只負責入隊
    with node_mgr.lock:
        adapter_queues[selected_id].append({
            "rid": rid, "prompt": req.prompt, "adapter_id": selected_id, "max_new_tokens": req.max_new_tokens
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
        return {
            "node_type": "CONTROL_NODE",
            "active_nodes": len(node_mgr.active_urls),
            "queues": {k: len(v) for k, v in adapter_queues.items()},
            "merged_map": node_mgr.merged_assignment,
            "nodes": node_mgr.nodes # Expose node detail for debugging
        }

@app.get("/fetch_adapter/{adapter_id}")
async def fetch_adapter_for_compute(adapter_id: str):
    await ensure_local_adapter(adapter_id)
    target_file = os.path.join(LORA_PATH, adapter_id, "adapter_model.safetensors")
    if os.path.exists(target_file):
        return FileResponse(target_file, media_type="application/octet-stream", filename="adapter_model.safetensors")
    raise HTTPException(404, "Adapter could not be fetched.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))
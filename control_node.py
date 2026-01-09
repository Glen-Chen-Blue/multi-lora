import os
import time
import uuid
import threading
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional, Deque, List, Tuple
from collections import deque, defaultdict
from queue import Queue, Empty

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ============================================================
# Config
# ============================================================
# e.g. COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"
ALL_CANDIDATE_NODES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",") if x.strip()]

POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "0.05"))
SCHED_INTERVAL_SEC = float(os.environ.get("SCHED_INTERVAL_SEC", "0.02"))

# Merge Trigger
QMIN_MULT = int(os.environ.get("QMIN_MULT", "4"))

# Auto-Scaling
SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD", "4"))
SCALE_COOLDOWN_SEC = float(os.environ.get("SCALE_COOLDOWN_SEC", "4"))

# [NEW] Stream Timeout: 若 Client 在 10 秒內沒建立 SSE 連線，則視為無效請求並清除
STREAM_TIMEOUT_SEC = 30.0

STALE_SEC = float(os.environ.get("STALE_SEC", "0.5"))

# ============================================================
# Data types
# ============================================================
@dataclass
class PendingRequest:
    request_id: str
    prompt: str
    adapter_id: str
    max_new_tokens: int
    enqueue_ts: float

class AddRequestBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: Optional[int] = 128

# ============================================================
# Control Node State
# ============================================================
app = FastAPI(title="Control Node (Final Secure)")

lock = threading.Lock()

# Resource Pool
active_node_urls: List[str] = []
standby_node_urls: List[str] = []

if len(ALL_CANDIDATE_NODES) > 0:
    active_node_urls.append(ALL_CANDIDATE_NODES[0])
    standby_node_urls.extend(ALL_CANDIDATE_NODES[1:])

# Nodes State
nodes: Dict[str, Dict[str, Any]] = {}

# Buffers
adapter_queues: Dict[str, Deque[PendingRequest]] = defaultdict(deque)

# [MODIFIED] Stream Queues: 存儲 (Queue, created_at_ts) 以便 Reaper 檢查超時
stream_queues: Dict[str, Tuple[Queue, float]] = {}

merged_assignment: Dict[str, str] = {}
wakeup = threading.Event()
last_scale_action_ts: float = 0.0

# ============================================================
# Helpers
# ============================================================
def _now() -> float:
    return time.time()

def _http_get_json(url: str, path: str, timeout: float = 1.5) -> Optional[Dict[str, Any]]:
    try:
        r = httpx.get(f"{url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _http_post_json(url: str, path: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        r = httpx.post(f"{url}{path}", json=payload, timeout=timeout)
        if r.status_code >= 400: return None
        return r.json()
    except Exception:
        return None

def _mark_node(url: str, metrics: Optional[Dict[str, Any]]):
    now = _now()
    with lock:
        st = nodes.get(url)
        if st is None:
            st = {"metrics": None, "local_load": 0, "last_seen": 0.0, "healthy": False, "mode": "NORMAL", "target": None}
            nodes[url] = st
        if metrics is not None:
            st["metrics"] = metrics
            st["last_seen"] = now
            # Sync local_load with reality
            real_load = int(metrics["load"]["running_batch"]) + int(metrics["load"]["waiting_queue"])
            st["local_load"] = real_load

def _recompute_health():
    now = _now()
    with lock:
        for url, st in nodes.items():
            if url in active_node_urls:
                st["healthy"] = (now - float(st.get("last_seen", 0.0))) <= STALE_SEC
            else:
                st["healthy"] = False

def _healthy_nodes() -> List[str]:
    with lock:
        return [u for u, st in nodes.items() if st.get("healthy", False) and u in active_node_urls]

def _total_queue_len() -> int:
    with lock: return sum(len(q) for q in adapter_queues.values())

def _adapter_counts_snapshot() -> Dict[str, int]:
    with lock: return {a: len(q) for a, q in adapter_queues.items() if len(q) > 0}

def _node_metrics(url: str) -> Optional[Dict[str, Any]]:
    with lock:
        st = nodes.get(url)
        if not st or not st.get("healthy", False): return None
        return st.get("metrics")

def _node_mode(url: str) -> Tuple[str, Optional[str]]:
    with lock:
        st = nodes.get(url, {})
        return str(st.get("mode", "NORMAL")), st.get("target")

def _set_node_mode(url: str, mode: str, target: Optional[str]):
    with lock:
        st = nodes.get(url)
        if st is None: return
        st["mode"] = mode
        st["target"] = target

def _node_can_accept(url: str, adapter_id: str) -> bool:
    with lock:
        st = nodes.get(url)
        if not st or not st.get("metrics"): return False
        
        # Check Optimistic Load
        current_load = st.get("local_load", 0)
        max_bs = int(st["metrics"]["capacity"]["max_batch_size"])
        if current_load >= max_bs: return False

        mode = str(st.get("mode", "NORMAL"))
        target = st.get("target")
        metrics = st["metrics"]

    if mode == "DRAINING": return False
    merged = metrics["lora_state"]["merged_adapter"]
    if merged is not None and str(merged) != str(adapter_id): return False
    if mode == "MERGED" and target is not None and str(target) != str(adapter_id): return False
    return True

def _choose_drain_node_for_target(target_adapter: str) -> Optional[str]:
    candidates: List[Tuple[int, int, int, str]] = []
    for url in _healthy_nodes():
        mode, _t = _node_mode(url)
        if mode != "NORMAL": continue
        m = _node_metrics(url)
        if m is None: continue
        running_adapters = set(map(str, m["lora_state"].get("running_adapters", [])))
        contains_target = 1 if str(target_adapter) in running_adapters else 0
        running_batch = int(m["load"]["running_batch"])
        waiting_queue = int(m["load"]["waiting_queue"])
        candidates.append((contains_target, running_batch, waiting_queue, url))
    
    if not candidates: return None
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return candidates[0][3]

# ============================================================
# Streaming & Cleanup Infrastructure
# ============================================================
def _ensure_stream_queue(request_id: str):
    with lock:
        if request_id not in stream_queues:
            # [MODIFIED] Store (Queue, Timestamp)
            stream_queues[request_id] = (Queue(), time.time())

def _push_stream_line(request_id: str, line: str):
    with lock:
        entry = stream_queues.get(request_id)
    if entry is not None:
        q, _ = entry
        q.put(line)

def _finish_stream(request_id: str):
    with lock:
        entry = stream_queues.get(request_id)
    if entry is not None:
        q, _ = entry
        q.put(None)

def _start_forwarder_to_compute(node_url: str, pending: PendingRequest):
    def run():
        payload = {"prompt": pending.prompt, "adapter_id": pending.adapter_id, "max_new_tokens": pending.max_new_tokens}
        try:
            with httpx.stream("POST", f"{node_url}/add_request", json=payload, timeout=None) as r:
                if r.status_code >= 400:
                    _push_stream_line(pending.request_id, "data: [ERROR]\n\n")
                    return
                for raw in r.iter_lines():
                    if not raw: continue
                    line = raw.strip()
                    if line.startswith("event:"): continue
                    if line.startswith("data:"): content = line[len("data:"):].lstrip()
                    else: content = line
                    if content: _push_stream_line(pending.request_id, f"data: {content}\n\n")
        except Exception:
            _push_stream_line(pending.request_id, "data: [ERROR]\n\n")
        finally:
            _finish_stream(pending.request_id)
    threading.Thread(target=run, daemon=True).start()

# [NEW] Reaper Loop: 清理超時未連線的 ID
def reaper_loop():
    while True:
        time.sleep(5.0) # Check every 5s
        now = time.time()
        ids_to_remove = []
        
        with lock:
            for rid, (q, created_ts) in stream_queues.items():
                if (now - created_ts) > STREAM_TIMEOUT_SEC:
                    ids_to_remove.append(rid)
        
        if ids_to_remove:
            print(f"[REAPER] 💀 Cleaning up {len(ids_to_remove)} stale streams: {ids_to_remove}")
            with lock:
                for rid in ids_to_remove:
                    stream_queues.pop(rid, None)

# ============================================================
# Poller
# ============================================================
def poller_loop():
    while True:
        with lock:
            current_targets = list(active_node_urls)
            for url in current_targets:
                if url not in nodes:
                    nodes[url] = {"metrics": None, "local_load": 0, "last_seen": 0.0, "healthy": False, "mode": "NORMAL", "target": None}
        for url in current_targets:
            metrics = _http_get_json(url, "/metrics", timeout=1.5)
            _mark_node(url, metrics)
        _recompute_health()
        wakeup.set()
        time.sleep(POLL_INTERVAL_SEC)

# ============================================================
# Scaling Logic
# ============================================================
def _check_autoscaling():
    global last_scale_action_ts
    
    with lock:
        q_total = sum(len(q) for q in adapter_queues.values())
        n_active = len(active_node_urls)
        n_standby = len(standby_node_urls)
    
    now = _now()
    if (now - last_scale_action_ts) < SCALE_COOLDOWN_SEC:
        return

    # Scale UP
    if n_active > 0 and n_standby > 0:
        if q_total > (SCALE_UP_THRESHOLD * n_active):
            with lock:
                if standby_node_urls:
                    new_node = standby_node_urls.pop(0)
                    active_node_urls.append(new_node)
                    if new_node not in nodes:
                        nodes[new_node] = {"metrics": None, "local_load": 0, "last_seen": 0.0, "healthy": False, "mode": "NORMAL", "target": None}
                    print(f"[AUTOSCALER] 🚀 Scaling UP! Activated: {new_node} (Queue: {q_total})")
                    last_scale_action_ts = now
            return

    # Scale DOWN
    if n_active > 1 and q_total == 0:
        with lock:
            candidate_index = -1
            candidate_url = None
            # Search for any idle node (skipping primary at index 0)
            for i in range(n_active - 1, 0, -1):
                url = active_node_urls[i]
                st = nodes.get(url)
                if st and st.get("metrics"):
                    metrics = st["metrics"]
                    is_idle = metrics.get("idle", False)
                    local_load = st.get("local_load", 0)
                    if is_idle and local_load == 0:
                        candidate_index = i
                        candidate_url = url
                        break
            
            if candidate_url:
                removed_node = active_node_urls.pop(candidate_index)
                standby_node_urls.insert(0, removed_node)
                if removed_node in nodes: nodes[removed_node]["healthy"] = False
                print(f"[AUTOSCALER] 💤 Scaling DOWN! Deactivated: {removed_node} (Queue: {q_total})")
                last_scale_action_ts = now

# ============================================================
# Scheduler
# ============================================================
def _maybe_trigger_merge():
    healthy = _healthy_nodes()
    N = len(healthy)
    if N <= 0: return
    counts = _adapter_counts_snapshot()
    Q = sum(counts.values())
    Q_min = QMIN_MULT * N
    if Q == 0 or Q < Q_min: return
    threshold = 1.0 / N
    with lock: assigned = set(merged_assignment.keys())
    hot = []
    for a, c in counts.items():
        if a in assigned: continue
        if (c / Q) > threshold: hot.append((c, a))
    if not hot: return
    hot.sort(reverse=True)
    _, target = hot[0]
    for url in healthy:
        mode, tgt = _node_mode(url)
        if mode == "MERGED" and str(tgt) == str(target):
            with lock: merged_assignment[str(target)] = url
            return
    drain_node = _choose_drain_node_for_target(target)
    if drain_node is None: return
    print(f"[SCHED] Triggering DRAIN on {drain_node} for target {target}")
    _set_node_mode(drain_node, "DRAINING", str(target))

def _maybe_finalize_drains():
    healthy = _healthy_nodes()
    for url in healthy:
        mode, target = _node_mode(url)
        if mode != "DRAINING" or not target: continue

        m = _node_metrics(url)
        if not m: continue
        
        # 雖然這裡檢查了 Idle，但 HTTP 請求到達時可能有微小時間差
        if bool(m.get("idle", False)) is not True: continue

        current_merged = m["lora_state"].get("merged_adapter", None)
        if current_merged is not None and str(current_merged) != str(target):
            # [FIX] 使用 force=True 避免 409 Conflict
            _http_post_json(url, "/unmerge", payload={"force": True}, timeout=5.0)

        # [FIX] 使用 force=True 強制合併，因為我們已經確認處於 Draining 狀態
        ok = _http_post_json(url, "/merge", payload={"adapter_id": str(target), "force": True}, timeout=10.0)
        
        if ok is None: continue

        print(f"[SCHED] Node {url} finalized MERGE for {target}")
        _set_node_mode(url, "MERGED", str(target))
        with lock:
            merged_assignment[str(target)] = url

def _maybe_revert_merges():
    with lock: current_assignments = list(merged_assignment.items())
    for adapter_id, node_url in current_assignments:
        q_len = 0
        with lock:
            if adapter_id in adapter_queues: q_len = len(adapter_queues[adapter_id])
        if q_len > 0: continue
        m = _node_metrics(node_url)
        if not m: continue
        with lock: local_load = nodes[node_url].get("local_load", 0)
        if local_load > 0: continue
        is_idle = bool(m.get("idle", False))
        if not is_idle: continue
        print(f"[SCHED] Reverting MERGE for {adapter_id} on {node_url} (Idle)")
        _http_post_json(node_url, "/unmerge", payload={"force": False}, timeout=5.0)
        _set_node_mode(node_url, "NORMAL", target=None)
        with lock:
            if merged_assignment.get(adapter_id) == node_url: del merged_assignment[adapter_id]

def _dispatch_one_ttft_first() -> bool:
    healthy = _healthy_nodes()
    if not healthy: return False

    with lock: merged_items = [(len(adapter_queues[a]), a, merged_assignment[a]) for a in merged_assignment.keys() if len(adapter_queues[a]) > 0]
    merged_items.sort(reverse=True)

    for _, a, node_url in merged_items:
        if not _node_can_accept(node_url, a): continue
        req = None
        with lock:
            if len(adapter_queues[a]) == 0: continue
            req = adapter_queues[a].popleft()
        
        # [SOLUTION] Zombie Check
        with lock: is_online = req.request_id in stream_queues
        if not is_online:
            print(f"[SCHED] 👻 Zombie task dropped: {req.request_id[:8]}")
            continue

        with lock:
            if node_url in nodes: nodes[node_url]["local_load"] += 1
        _start_forwarder_to_compute(node_url, req)
        return True

    with lock: normal_candidates = [(len(q), a) for a, q in adapter_queues.items() if len(q) > 0 and a not in merged_assignment]
    normal_candidates.sort(reverse=True)

    for _, a in normal_candidates:
        for url in healthy:
            if _node_can_accept(url, a):
                req = None
                with lock:
                    if len(adapter_queues[a]) == 0: break
                    req = adapter_queues[a].popleft()

                # [SOLUTION] Zombie Check
                with lock: is_online = req.request_id in stream_queues
                if not is_online:
                    print(f"[SCHED] 👻 Zombie task dropped: {req.request_id[:8]}")
                    continue

                with lock:
                    if url in nodes: nodes[url]["local_load"] += 1
                _start_forwarder_to_compute(url, req)
                return True
    return False

def scheduler_loop():
    while True:
        wakeup.wait()
        _check_autoscaling()
        _maybe_trigger_merge()
        _maybe_finalize_drains()
        _maybe_revert_merges()

        did = False
        for _ in range(32):
            if not _dispatch_one_ttft_first(): break
            did = True
        if _total_queue_len() == 0: wakeup.clear()
        if not did: time.sleep(SCHED_INTERVAL_SEC)

# ============================================================
# Start threads
# ============================================================
threading.Thread(target=poller_loop, daemon=True).start()
threading.Thread(target=scheduler_loop, daemon=True).start()
threading.Thread(target=reaper_loop, daemon=True).start() # [NEW] Start Reaper

# ============================================================
# APIs
# ============================================================
@app.get("/status")
def status() -> Dict[str, Any]:
    with lock:
        healthy = {u: st for u, st in nodes.items() if st.get("healthy", False)}
        q_counts = {a: len(q) for a, q in adapter_queues.items() if len(q) > 0}
        return {
            "healthy_nodes": list(healthy.keys()),
            "active_pool": active_node_urls,
            "standby_pool": standby_node_urls,
            "nodes": nodes,
            "queue_counts": q_counts,
            "queue_total": sum(q_counts.values()),
            "merged_assignment": dict(merged_assignment),
        }

@app.post("/send_request")
def send_request(req: AddRequestBody) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    adapter_id = str(req.adapter_id)
    max_new_tokens = int(req.max_new_tokens or 128)
    pending = PendingRequest(
        request_id=request_id, prompt=req.prompt, adapter_id=adapter_id,
        max_new_tokens=max_new_tokens, enqueue_ts=_now()
    )
    _ensure_stream_queue(request_id)
    with lock: adapter_queues[adapter_id].append(pending)
    wakeup.set()
    return {"request_id": request_id}

@app.get("/stream/{request_id}")
async def stream(request_id: str, request: Request):
    with lock:
        entry = stream_queues.get(request_id)
    
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown request_id or expired")
    
    q, _ = entry

    async def gen():
        try:
            yield "event: open\n"
            yield "data: ok\n\n"
            while True:
                # Disconnect Detection
                if await request.is_disconnected():
                    print(f"[STREAM] Client disconnected: {request_id[:8]}")
                    break
                try:
                    item = q.get_nowait()
                    if item is None:
                        yield "event: end\n"
                        yield "data: [DONE]\n\n"
                        break
                    yield item
                except Empty:
                    await asyncio.sleep(0.02)
        finally:
            with lock:
                stream_queues.pop(request_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")
import os
import uuid
import threading
import time
import logging
import json
import asyncio
import httpx
from queue import Queue, Empty
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from multilora_system import MultiLoRAEngine

# ============================================================
# Logging
# ============================================================
class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "GET /metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("uvicorn.access").addFilter(MetricsFilter())
logger = logging.getLogger("ComputeNode")

# ============================================================
# Global State & Engine
# ============================================================
NODE_ID = os.environ.get("NODE_ID", "cn-1")
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")
# LORA_PATH is technically not needed for storage anymore, but maybe for fallback or just ignore
LORA_PATH = os.environ.get("LORA_PATH", "./lora_repo/compute")
CONTROL_NODE_URL = os.environ.get("CONTROL_NODE_URL", "http://localhost:9000")

# [Auto-Scale] 設定硬上限
MAX_BATCH_SIZE_LIMIT = int(os.environ.get("MAX_BATCH_SIZE", "32"))
# [CPU Cache] 設定 CPU LoRA 上限
MAX_CPU_LORAS = int(os.environ.get("MAX_CPU_LORAS", "10"))

engine: Optional[MultiLoRAEngine] = None
engine_wakeup = threading.Event()
shutdown_event = threading.Event()

# Streaming state
stream_queues: Dict[str, Queue] = {}
decoding_state: Dict[str, int] = {}
stream_lock = threading.Lock()

# Config Versioning State
last_config_version: int = -1
config_lock = threading.Lock()

client = httpx.AsyncClient(timeout=120.0)

# ============================================================
# Callbacks
# ============================================================
def on_token(rid: str, tokens_list: List[int]):
    with stream_lock:
        if rid not in stream_queues: return
        q = stream_queues[rid]
        
        start_len = decoding_state.get(rid, 0)
        full_text = engine.tokenizer.decode(tokens_list, skip_special_tokens=True)
        
        if len(full_text) > start_len:
            delta = full_text[start_len:]
            if delta.endswith("\ufffd"): return
            q.put(delta)
            decoding_state[rid] = len(full_text)

def on_finish(rid: str, reason: str):
    with stream_lock:
        if rid in stream_queues:
            q = stream_queues[rid]
            if reason == "aborted_by_merge":
                q.put({"type": "error", "message": "Request aborted by system merge."})
            else:
                q.put({"type": "final", "reason": reason})
            q.put(None) 
        decoding_state.pop(rid, None)

# ============================================================
# Network Fetcher (Sync for Engine Thread)
# ============================================================
def fetch_adapter_sync(adapter_id: str) -> bytes:
    """
    Synchronously fetch adapter bytes from Control Node.
    This runs inside the engine thread when a cache miss occurs.
    """
    url = f"{CONTROL_NODE_URL}/fetch_adapter/{adapter_id}"
    try:
        # Use a fresh sync client for thread safety within the engine loop
        with httpx.Client(timeout=60.0) as sync_client:
            resp = sync_client.get(url)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code} from {url}")
            return resp.content
    except Exception as e:
        logger.error(f"❌ Failed to fetch adapter {adapter_id}: {e}")
        raise e

# ============================================================
# Engine Loop
# ============================================================
def engine_loop_thread():
    logger.info("🚀 Engine loop started.")
    while not shutdown_event.is_set():
        engine_wakeup.wait(timeout=1.0)
        if shutdown_event.is_set(): break
        try:
            did_work = engine.step()
            if not did_work:
                if engine.is_idle(): engine_wakeup.clear()
                else: time.sleep(0.001) 
        except Exception as e:
            logger.error(f"❌ Engine step error: {e}", exc_info=True)
            time.sleep(1)

# ============================================================
# Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # We don't necessarily need to create LORA_PATH directory anymore since we are diskless
    # os.makedirs(LORA_PATH, exist_ok=True) 
    
    logger.info(f"Initializing Compute Node {NODE_ID} (Max CPU LoRAs: {MAX_CPU_LORAS}, Diskless Mode)...")
    
    engine = MultiLoRAEngine(
        model_id=MODEL_ID,
        adapter_slots=8,
        max_batch_size=MAX_BATCH_SIZE_LIMIT,
        max_cpu_loras=MAX_CPU_LORAS,
        enable_monitor=True,
        adapter_fetcher=fetch_adapter_sync  # Inject the network fetcher
    )
    engine.on_token = on_token
    engine.on_finish = on_finish
    
    # NOTE: We no longer scan local disk.
    # We wait for 'sync_adapters' to tell us what is available,
    # or rely on on-demand fetching.
    
    t = threading.Thread(target=engine_loop_thread, daemon=True)
    t.start()
    yield
    logger.info("Shutting down...")
    shutdown_event.set()
    engine_wakeup.set()
    t.join(timeout=5)
    await client.aclose()

app = FastAPI(title=f"Compute Node {NODE_ID}", lifespan=lifespan)

# ============================================================
# API Models
# ============================================================
class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

class MergeRequest(BaseModel):
    adapter_id: str
    force: bool = False

class UnmergeRequest(BaseModel):
    force: bool = False

class SyncAdaptersRequest(BaseModel):
    adapters: List[str]
    version_id: int 

# ============================================================
# Endpoints
# ============================================================
@app.get("/metrics")
def metrics():
    if not engine: return {}
    return {
        "node_id": NODE_ID,
        "load": {
            "running_batch": len(engine.running_queue),
            "waiting_queue": len(engine.request_queue)
        },
        "lora_state": {
            "merged_adapter": engine.current_merged_adapter,
            "running_adapters": list({str(r["adapter_id"]) for r in engine.running_queue}),
            "loaded_adapters": list(engine.cpu_cache.keys()) 
        },
        "capacity": {
            "max_batch_size": engine.max_batch_size,
            "max_cpu_loras": engine.max_cpu_loras
        },
        "idle": engine.is_idle(),
        "draining": engine.is_draining,
        "config_version": last_config_version
    }

@app.post("/sync_adapters")
async def sync_adapters(req: SyncAdaptersRequest):
    global last_config_version
    
    with config_lock:
        if req.version_id <= last_config_version:
            return {"status": "ignored"}
        last_config_version = req.version_id

    try:
        # Instead of downloading files to disk, we just update the allowed list in Engine.
        # The Engine will fetch them on-demand if they are needed for inference.
        logger.info(f"🔄 Syncing adapter list (v{req.version_id}): {len(req.adapters)} items")
        engine.update_known_adapters(req.adapters)
        
        return {
            "status": "ok", 
            "version_applied": req.version_id,
            "loaded": list(engine.cpu_cache.keys())
        }
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/add_request")
def add_request(req: AddRequest):
    rid = str(uuid.uuid4())
    q = Queue()
    
    with stream_lock:
        stream_queues[rid] = q
        decoding_state[rid] = 0
    
    try:
        engine.add_request(req.prompt, req.adapter_id, rid, req.max_new_tokens)
        engine_wakeup.set()
    except KeyError as e:
        with stream_lock:
            stream_queues.pop(rid, None)
            decoding_state.pop(rid, None)
        raise HTTPException(400, f"Adapter {req.adapter_id} error: {e}")

    def event_generator():
        try:
            while True:
                try:
                    item = q.get(timeout=60) 
                except Empty:
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                
                if isinstance(item, dict):
                    if item.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps(item['message'])}\n\n"
                        break
                    continue

                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            logger.warning(f"Stream broken for {rid}: {e}")
        finally:
            with stream_lock:
                stream_queues.pop(rid, None)
                decoding_state.pop(rid, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/merge")
def merge(req: MergeRequest):
    try:
        engine.merge_adapter(req.adapter_id, force=req.force)
        return {"status": "merged", "adapter": req.adapter_id}
    except Exception as e:
        raise HTTPException(400, f"Merge failed: {e}")

@app.post("/unmerge")
def unmerge(req: UnmergeRequest):
    if not req.force and not engine.is_idle():
        raise HTTPException(409, "Engine not idle")
    engine.unmerge_all()
    return {"status": "unmerged"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
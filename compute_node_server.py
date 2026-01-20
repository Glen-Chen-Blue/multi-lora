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
LORA_PATH = os.environ.get("LORA_PATH", "./lora_repo/compute")
CONTROL_NODE_URL = os.environ.get("CONTROL_NODE_URL", "http://localhost:9000")

# [Auto-Scale] 設定硬上限，預設 32。動態調整會在此範圍內運作。
MAX_BATCH_SIZE_LIMIT = int(os.environ.get("MAX_BATCH_SIZE", "32"))

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

client = httpx.AsyncClient(timeout=120.0) # Long timeout for download

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
    os.makedirs(LORA_PATH, exist_ok=True)
    logger.info(f"Initializing Compute Node {NODE_ID}...")
    
    engine = MultiLoRAEngine(
        model_id=MODEL_ID,
        adapter_slots=8,
        max_batch_size=MAX_BATCH_SIZE_LIMIT, # 傳入硬上限
        enable_monitor=True
    )
    engine.on_token = on_token
    engine.on_finish = on_finish
    
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
# Download Logic
# ============================================================
async def download_missing_adapters(adapters: List[str]):
    """
    Check LORA_PATH for missing adapters.
    If missing, download from Control Node.
    """
    for aid in adapters:
        target_dir = os.path.join(LORA_PATH, aid)
        target_file = os.path.join(target_dir, "adapter_model.safetensors")
        
        if os.path.exists(target_file):
            continue
            
        logger.info(f"📥 Missing {aid}, downloading from {CONTROL_NODE_URL}...")
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            async with client.stream("GET", f"{CONTROL_NODE_URL}/fetch_adapter/{aid}") as resp:
                if resp.status_code != 200:
                    logger.error(f"Failed to fetch {aid} from Control Node: {resp.status_code}")
                    continue
                
                with open(target_file, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
            logger.info(f"✅ Downloaded {aid} successfully.")
        except Exception as e:
            logger.error(f"Download error for {aid}: {e}")
            if os.path.exists(target_file): os.remove(target_file)

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
        "capacity": {"max_batch_size": engine.max_batch_size},
        "idle": engine.is_idle(),
        "draining": engine.is_draining,
        "config_version": last_config_version
    }

@app.post("/sync_adapters")
async def sync_adapters(req: SyncAdaptersRequest):
    global last_config_version
    
    with config_lock:
        if req.version_id <= last_config_version:
            logger.warning(f"⚠️ Ignoring stale config v{req.version_id} (Current: v{last_config_version})")
            return {
                "status": "ignored", 
                "reason": "stale_version",
                "loaded": list(engine.cpu_cache.keys())
            }
        last_config_version = req.version_id

    try:
        # [NEW] Download step
        await download_missing_adapters(req.adapters)
        
        # Then load from local disk
        engine.load_adapters_to_cpu(LORA_PATH, allowed_adapters=req.adapters)
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
        # Note: If adapter is not loaded, this will fail. 
        # Ideally, Control Node ensures sync_adapters is called before dispatching.
        engine.add_request(req.prompt, req.adapter_id, rid, req.max_new_tokens)
        engine_wakeup.set()
    except KeyError as e:
        with stream_lock:
            stream_queues.pop(rid, None)
            decoding_state.pop(rid, None)
        raise HTTPException(400, f"Adapter {req.adapter_id} not available: {e}")

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
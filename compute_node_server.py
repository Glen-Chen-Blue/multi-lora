# compute_node_server.py
import logging
import os
import threading
import time
import uuid
from queue import Queue
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from multilora_system import MultiLoRAEngine

# ============================================================
# Logging Setup
# ============================================================
class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "GET /metrics" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(MetricsFilter())

# ============================================================
# Config
# ============================================================
NODE_ID = os.environ.get("NODE_ID", "cn-1")
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")
# 注意：為了測試 Backpressure，建議 ADAPTER_SLOTS 和 MAX_BATCH_SIZE 設小一點
ADAPTER_SLOTS = int(os.environ.get("ADAPTER_SLOTS", "4"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
ENABLE_MONITOR = bool(int(os.environ.get("ENABLE_MONITOR", "0")))
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")

# ============================================================
# Engine init
# ============================================================
engine = MultiLoRAEngine(
    model_id=MODEL_ID,
    adapter_slots=ADAPTER_SLOTS,
    max_batch_size=MAX_BATCH_SIZE,
    enable_monitor=ENABLE_MONITOR,
)
engine.load_adapters_to_cpu(LORA_PATH)

engine_wakeup = threading.Event()

# ============================================================
# Streaming Infrastructure
# ============================================================
stream_queues: Dict[str, Queue] = {}
full_text_buffer: Dict[str, str] = {}
stream_lock = threading.Lock()

def _safe_decode_token(token_id: int) -> str:
    if token_id == engine.tokenizer.eos_token_id:
        return ""
    try:
        return engine.tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return ""

def on_token(request_id: str, token_id: int):
    text = _safe_decode_token(token_id)
    if text == "":
        return
    
    # 寫入 Buffer
    with stream_lock:
        if request_id in full_text_buffer:
            full_text_buffer[request_id] += text
            
    # 寫入 Queue 供 SSE 使用
    with stream_lock:
        q = stream_queues.get(request_id)
    if q is not None:
        q.put(text)

def on_finish(request_id: str, reason: str):
    with stream_lock:
        full_text = full_text_buffer.get(request_id, "")
        q = stream_queues.get(request_id)

    if q is not None:
        q.put({
            "type": "final",
            "text": full_text,
            "reason": reason,
        })
        q.put(None)  # Sentinel for end of stream

engine.on_token = on_token
engine.on_finish = on_finish

def engine_loop():
    while True:
        engine_wakeup.wait()
        did_work = engine.step()
        if not did_work:
            if engine.is_idle():
                engine_wakeup.clear()
            else:
                # 即使 idle 也可以稍作休息避免 CPU 此 loop 100%
                time.sleep(0.001)

threading.Thread(target=engine_loop, daemon=True).start()

app = FastAPI(title=f"Compute Node ({NODE_ID})")

# ============================================================
# Pydantic Models
# ============================================================
class AddRequestBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: Optional[int] = 128

class MergeBody(BaseModel):
    adapter_id: str
    force: Optional[bool] = False

# ============================================================
# SSE Generator
# ============================================================
def sse_stream(request_id: str):
    with stream_lock:
        q = stream_queues.get(request_id)

    if q is None:
        yield "event: end\ndata: [DONE]\n\n"
        return

    try:
        while True:
            item = q.get()
            if item is None:
                yield "event: end\ndata: [DONE]\n\n"
                break

            if isinstance(item, dict) and item.get("type") == "final":
                yield "event: final\n"
                yield f"data: {item['text']}\n\n"
                continue

            # 一般 token
            yield f"data: {item}\n\n"

    finally:
        with stream_lock:
            stream_queues.pop(request_id, None)
            full_text_buffer.pop(request_id, None)

# ============================================================
# API Endpoints
# ============================================================
@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    # running_queue: 正在 GPU 計算或準備計算的
    # request_queue: 排隊中，尚未進入 batch 的 (Backpressure)
    running_batch = len(engine.running_queue)
    waiting_queue = len(engine.request_queue)
    
    merged = engine.current_merged_adapter
    running_adapters: List[str] = sorted({str(r.get("adapter_id")) for r in engine.running_queue})
    
    return {
        "node_id": NODE_ID,
        "capacity": {
            "max_batch_size": engine.max_batch_size,
            "adapter_slots": engine.adapter_slots,
        },
        "load": {
            "running_batch": running_batch,
            "waiting_queue": waiting_queue,
        },
        "lora_state": {
            "merged_adapter": merged,
            "running_adapters": running_adapters,
            "gpu_slots": dict(engine.gpu_slots),
        },
        "idle": engine.is_idle(),
    }

@app.post("/add_request")
def add_request(req: AddRequestBody):
    request_id = str(uuid.uuid4())

    q = Queue()
    with stream_lock:
        stream_queues[request_id] = q
        full_text_buffer[request_id] = ""

    try:
        engine.add_request(
            prompt=req.prompt,
            adapter_id=req.adapter_id,
            request_id=request_id,
            max_new_tokens=int(req.max_new_tokens or 128),
        )
    except KeyError as e:
        with stream_lock:
            stream_queues.pop(request_id, None)
            full_text_buffer.pop(request_id, None)
        raise HTTPException(status_code=400, detail=str(e))

    engine_wakeup.set()
    return StreamingResponse(sse_stream(request_id), media_type="text/event-stream")

@app.post("/merge")
def merge(req: MergeBody):
    # Merge 是一個重操作，通常要求 idle，除非 force=True
    if (not req.force) and (not engine.is_idle()):
        raise HTTPException(status_code=409, detail="Engine not idle. Merge allowed only when idle (or force=true).")
    try:
        engine.merge_adapter(req.adapter_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "merged": req.adapter_id, "node_id": NODE_ID}

@app.post("/unmerge")
def unmerge(force: bool = False):
    if (not force) and (not engine.is_idle()):
        raise HTTPException(status_code=409, detail="Engine not idle. Unmerge allowed only when idle (or force=true).")
    engine.unmerge_all()
    return {"status": "ok", "merged": None, "node_id": NODE_ID}
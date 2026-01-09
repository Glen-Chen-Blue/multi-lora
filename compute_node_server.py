import os
import uuid
import threading
import time
import logging
from queue import Queue
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from multilora_system import MultiLoRAEngine

# ============================================================
# Logging Setup (防止 /metrics 洗版)
# ============================================================
class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "GET /metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(MetricsFilter())

# ============================================================
# Config & Init
# ============================================================
NODE_ID = os.environ.get("NODE_ID", "cn-1")
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")

engine = MultiLoRAEngine(
    model_id=MODEL_ID,
    adapter_slots=4,
    max_batch_size=8,
    enable_monitor=True
)
engine.load_adapters_to_cpu(LORA_PATH)
engine_wakeup = threading.Event()

# ============================================================
# Streaming
# ============================================================
stream_queues: Dict[str, Queue] = {}
stream_lock = threading.Lock()

def on_token(rid, tid):
    text = engine.tokenizer.decode([tid], skip_special_tokens=True)
    with stream_lock:
        if rid in stream_queues: stream_queues[rid].put(text)

def on_finish(rid, reason):
    with stream_lock:
        if rid in stream_queues:
            stream_queues[rid].put({"type": "final", "reason": reason})
            stream_queues[rid].put(None)

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
                time.sleep(0.001)

threading.Thread(target=engine_loop, daemon=True).start()

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

# [FIX] 新增這個 Model 讓 force 可以從 JSON Body 讀取
class UnmergeRequest(BaseModel):
    force: bool = False

# ============================================================
# API Endpoints
# ============================================================
app = FastAPI(title=f"Compute Node {NODE_ID}")

@app.get("/metrics")
def metrics():
    return {
        "node_id": NODE_ID,
        "load": {
            "running_batch": len(engine.running_queue),
            "waiting_queue": len(engine.request_queue)
        },
        "lora_state": {
            "merged_adapter": engine.current_merged_adapter,
            "running_adapters": list({str(r["adapter_id"]) for r in engine.running_queue})
        },
        "capacity": {"max_batch_size": engine.max_batch_size},
        "idle": engine.is_idle()
    }

@app.post("/add_request")
def add_request(req: AddRequest):
    rid = str(uuid.uuid4())
    q = Queue()
    with stream_lock: stream_queues[rid] = q
    
    try:
        engine.add_request(req.prompt, req.adapter_id, rid, req.max_new_tokens)
        engine_wakeup.set()
    except KeyError as e:
        with stream_lock: del stream_queues[rid]
        raise HTTPException(400, str(e))

    def generator():
        try:
            while True:
                item = q.get()
                if item is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                if isinstance(item, dict): continue
                yield f"data: {item}\n\n"
        finally:
            with stream_lock: stream_queues.pop(rid, None)

    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/merge")
def merge(req: MergeRequest):
    # 如果不是強制，且引擎不空閒，則拒絕
    if not req.force and not engine.is_idle():
        raise HTTPException(409, "Engine not idle")
    try:
        engine.merge_adapter(req.adapter_id)
        return {"status": "merged", "adapter": req.adapter_id}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/unmerge")
def unmerge(req: UnmergeRequest): # [FIX] 使用 UnmergeRequest 接收 Body
    # 這裡現在能正確讀到 req.force 了
    if not req.force and not engine.is_idle():
        raise HTTPException(409, "Engine not idle")
    
    engine.unmerge_all()
    return {"status": "unmerged"}
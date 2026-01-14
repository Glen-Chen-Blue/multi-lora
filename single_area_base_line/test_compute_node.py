import os
import uuid
import threading
import time
import logging
import json
from queue import Queue, Empty
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from test_multilora_system import MultiLoRAEngine

# ============================================================
# Logging & Config
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [Compute] %(message)s")
logger = logging.getLogger("ComputeNode")

NODE_ID = os.environ.get("NODE_ID", "cn-test")
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")
LORA_PATH = os.environ.get("LORA_PATH", "../testLoRA")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))

engine: Optional[MultiLoRAEngine] = None
engine_wakeup = threading.Event()
stream_queues: Dict[str, Queue] = {}
stream_lock = threading.Lock()

# ============================================================
# Callbacks
# ============================================================
def on_token(rid: str, tokens_list: List[int]):
    with stream_lock:
        if rid in stream_queues:
            # 簡單解碼：這裡只傳回最新生成的文字 (簡易版)
            # 實際應用建議用 incremental decode，這裡為求測試一致性
            # 我們假設 Engine 已經幫忙處理好，或者 Client 自己處理
            # 為了與原版相容，我們重新 decode 全部並取差異 (比較慢但安全)
            full_text = engine.tokenizer.decode(tokens_list, skip_special_tokens=True)
            stream_queues[rid].put({"type": "token", "text": full_text})

def on_finish(rid: str, reason: str):
    with stream_lock:
        if rid in stream_queues:
            stream_queues[rid].put({"type": "final", "reason": reason})
            stream_queues[rid].put(None)

# ============================================================
# Engine Thread
# ============================================================
def engine_loop():
    while True:
        engine_wakeup.wait()
        did_work = engine.step()
        if not did_work:
            if engine.is_idle():
                engine_wakeup.clear()
            else:
                time.sleep(0.001)

# ============================================================
# App Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info(f"🚀 Starting Compute Node {NODE_ID} (Fixed Batch: {MAX_BATCH_SIZE})")
    
    # [Fix] 移除 enable_monitor 參數
    engine = MultiLoRAEngine(
        model_id=MODEL_ID,
        adapter_slots=8, 
        max_batch_size=MAX_BATCH_SIZE
    )
    engine.on_token = on_token
    engine.on_finish = on_finish
    
    # [Requirement] Load ALL LoRAs at startup
    logger.info("📦 Loading ALL adapters to CPU...")
    engine.load_adapters_to_cpu(LORA_PATH)
    
    t = threading.Thread(target=engine_loop, daemon=True)
    t.start()
    yield

app = FastAPI(lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

# ============================================================
# APIs
# ============================================================
@app.post("/add_request")
def add_request(req: AddRequest):
    rid = str(uuid.uuid4())
    q = Queue()
    
    # 這裡要做一點簡單的 diff 處理，因為 callback 給的是 full text
    decoding_len = 0 
    
    with stream_lock:
        stream_queues[rid] = q
    
    try:
        engine.add_request(req.prompt, req.adapter_id, rid, req.max_new_tokens)
        engine_wakeup.set()
    except KeyError as e:
        with stream_lock: del stream_queues[rid]
        raise HTTPException(400, f"Adapter not found: {e}")

    def event_generator():
        nonlocal decoding_len
        try:
            while True:
                item = q.get()
                if item is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                
                if item["type"] == "token":
                    full = item["text"]
                    if len(full) > decoding_len:
                        delta = full[decoding_len:]
                        # 避免 unicode replacement character 造成的截斷問題
                        if not delta.endswith("\ufffd"):
                            yield f"data: {json.dumps(delta)}\n\n"
                            decoding_len = len(full)
                elif item["type"] == "final":
                     pass # 結束
        finally:
            with stream_lock:
                stream_queues.pop(rid, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
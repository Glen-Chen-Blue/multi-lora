import os
import uuid
import threading
import time
import logging
import json
from queue import Queue, Empty
from typing import Dict, Any, Optional, List
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

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(MetricsFilter())

# ============================================================
# Config & Init
# ============================================================
NODE_ID = os.environ.get("NODE_ID", "cn-1")
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")

print(f"[{NODE_ID}] Initializing Engine...")
engine = MultiLoRAEngine(
    model_id=MODEL_ID,
    adapter_slots=4,
    max_batch_size=8,
    enable_monitor=True
)
# [Modified] Start empty! Wait for Control Node to sync via /sync_adapters
# engine.load_adapters_to_cpu(LORA_PATH) 

engine_wakeup = threading.Event()

# ============================================================
# Streaming & Decoding State
# ============================================================
stream_queues: Dict[str, Queue] = {}
decoding_state: Dict[str, int] = {} 
stream_lock = threading.Lock()

def on_token(rid, tokens_list):
    """
    接收完整的 tokens_list，進行增量解碼
    """
    with stream_lock:
        if rid not in stream_queues: return
        
        # 取得上一次解碼的長度
        start_len = decoding_state.get(rid, 0)
        
        # 解碼全部
        full_text = engine.tokenizer.decode(tokens_list, skip_special_tokens=True)
        
        # 取出新增的部分 (Delta)
        if len(full_text) > start_len:
            delta = full_text[start_len:]
            
            # [FIXED] 關鍵修正：檢查 Replacement Character (\ufffd)
            if delta.endswith("\ufffd"):
                return 
                
            stream_queues[rid].put(delta)
            decoding_state[rid] = len(full_text)

def on_finish(rid, reason):
    with stream_lock:
        if rid in stream_queues:
            # 針對被強制中止的請求提供錯誤訊息
            if reason == "aborted_by_merge":
                stream_queues[rid].put({
                    "type": "error",
                    "message": "Processing aborted due to forced model merge."
                })
            else:
                stream_queues[rid].put({"type": "final", "reason": reason})
            stream_queues[rid].put(None) # Sentinel to stop generator
            
        # 清理狀態
        decoding_state.pop(rid, None)

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

class UnmergeRequest(BaseModel):
    force: bool = False

class SyncAdaptersRequest(BaseModel):
    adapters: List[str]

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
            "running_adapters": list({str(r["adapter_id"]) for r in engine.running_queue}),
            "loaded_adapters": list(engine.cpu_cache.keys())
        },
        "capacity": {"max_batch_size": engine.max_batch_size},
        "idle": engine.is_idle(),
        "draining": engine.is_draining
    }

@app.post("/sync_adapters")
def sync_adapters(req: SyncAdaptersRequest):
    """
    [NEW] Control Node 呼叫此接口，強制 Compute Node 只加載指定的 Adapters
    """
    try:
        # Reload with whitelist
        engine.load_adapters_to_cpu(LORA_PATH, allowed_adapters=req.adapters)
        return {
            "status": "ok", 
            "loaded_count": len(engine.cpu_cache),
            "loaded_ids": list(engine.cpu_cache.keys())
        }
    except Exception as e:
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
        # 這裡會捕捉到因為 pruning 導致找不到 adapter 的錯誤
        with stream_lock: 
            del stream_queues[rid]
            decoding_state.pop(rid, None)
        raise HTTPException(400, f"Adapter Error: {str(e)}")

    def generator():
        try:
            while True:
                item = q.get()
                if item is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                
                if isinstance(item, dict): 
                    if item.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps(item['message'])}\n\n"
                        break
                    # Final event with reason
                    if item.get("type") == "final":
                         continue

                yield f"data: {json.dumps(item)}\n\n"
        finally:
            with stream_lock: 
                stream_queues.pop(rid, None)
                decoding_state.pop(rid, None)

    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/merge")
def merge(req: MergeRequest):
    try:
        engine.merge_adapter(req.adapter_id, force=req.force)
        return {
            "status": "merged", 
            "adapter": req.adapter_id, 
            "mode": "force" if req.force else "graceful"
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/unmerge")
def unmerge(req: UnmergeRequest):
    if not req.force and not engine.is_idle():
        raise HTTPException(409, "Engine not idle")
    
    engine.unmerge_all()
    return {"status": "unmerged"}
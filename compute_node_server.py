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
# 匯入集中管理的設定
from config import MODEL_ID, FIXED_OUTPUT_LEN

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

CONTROL_NODE_URL = os.environ.get("CONTROL_NODE_URL", "http://localhost:9000")

engine: Optional[MultiLoRAEngine] = None
engine_wakeup = threading.Event()
shutdown_event = threading.Event()

stream_queues: Dict[str, Queue] = {}
decoding_state: Dict[str, int] = {}
stream_lock = threading.Lock()

last_config_version: int = -1
config_lock = threading.Lock()

client = httpx.AsyncClient(timeout=120.0)

# [新增] 狀態機狀態：active | standby | draining
current_status = "standby"

# [新增] 累加型監控指標 (Cumulative Metrics)
cumulative_metrics = {
    "effective_inference_time": 0.0  # 累計的 Active Batching Time (秒)
}
metrics_lock = threading.Lock()

# ============================================================
# Background Registration Task
# ============================================================
async def register_with_control_node():
    """
    背景任務：在啟動時不斷嘗試向 Control Node 註冊自己的存在
    """
    port = os.environ.get("PORT", "8001")
    my_url = os.environ.get("COMPUTE_NODE_URL", f"http://127.0.0.1:{port}")
    
    logger.info(f"🔄 Attempting to register to Control Node at {CONTROL_NODE_URL} with my URL: {my_url}")
    
    while not shutdown_event.is_set():
        try:
            async with httpx.AsyncClient() as c:
                resp = await c.post(
                    f"{CONTROL_NODE_URL}/register_node", 
                    json={"url": my_url}, 
                    timeout=5.0
                )
                if resp.status_code == 200:
                    logger.info(f"✅ Successfully registered to Control Node!")
                    break  # 註冊成功，跳出迴圈
                else:
                    logger.warning(f"⚠️ Registration rejected: {resp.text}, retrying in 3s...")
        except Exception as e:
            logger.warning(f"⚠️ Control Node unreachable ({e}), retrying in 3s...")
        
        await asyncio.sleep(3.0)

# ============================================================
# Callbacks & Network Fetcher
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
        if rid in decoding_state:
            del decoding_state[rid]

def fetch_adapter_sync(adapter_id: str) -> bytes:
    url = f"{CONTROL_NODE_URL}/fetch_adapter/{adapter_id}"
    try:
        with httpx.Client(timeout=60.0) as sync_client:
            resp = sync_client.get(url)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code} from {url}")
            return resp.content
    except Exception as e:
        logger.error(f"❌ Failed to fetch adapter {adapter_id}: {e}")
        raise e

# ============================================================
# Engine Loop & Lifecycle
# ============================================================
def engine_loop_thread():
    global current_status
    logger.info("🚀 Engine loop started.")
    while not shutdown_event.is_set():
        engine_wakeup.wait(timeout=1.0)
        if shutdown_event.is_set(): break
        try:
            # [修改] 加入精確計時器，計算 Effective Inference Time
            start_time = time.time()
            did_work = engine.step()
            
            # 若 engine.step() 有真正在推進推論（Batch 中有任務）
            if did_work:
                elapsed = time.time() - start_time
                with metrics_lock:
                    cumulative_metrics["effective_inference_time"] += elapsed

            # [核心機制] 如果處於排空模式，且手上的任務都清空了，自動轉為 Standby
            if current_status == "draining" and engine.is_idle():
                logger.info("❄️ 所有任務已排空，節點自動轉為 Standby")
                current_status = "standby"
                
            if not did_work:
                if engine.is_idle(): engine_wakeup.clear()
                else: time.sleep(0.001) 
        except Exception as e:
            logger.error(f"❌ Engine step error: {e}", exc_info=True)
            time.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info(f"Initializing Compute Node {NODE_ID} (Research Note Config: Merged=15, Unmerged=12)...")
    
    engine = MultiLoRAEngine(
        model_id=MODEL_ID,
        adapter_fetcher=fetch_adapter_sync,
        enable_monitor=True
    )
    engine.on_token = on_token
    engine.on_finish = on_finish
    
    t = threading.Thread(target=engine_loop_thread, daemon=True)
    t.start()
    
    asyncio.create_task(register_with_control_node())
    
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
    max_new_tokens: int = 256

class MergeRequest(BaseModel):
    adapter_id: str
    force: bool = False

class UnmergeRequest(BaseModel):
    force: bool = False

class SyncAdaptersRequest(BaseModel):
    adapters: List[str]
    version_id: int 

class SetStatusRequest(BaseModel):
    status: str

# ============================================================
# Endpoints
# ============================================================
@app.post("/set_status")
def set_status(req: SetStatusRequest):
    """由 Control Node 主動呼叫，用於直接指派狀態 (例如 Wake up)"""
    global current_status
    current_status = req.status
    logger.info(f"🔄 狀態變更為: {current_status}")
    return {"ok": True}

@app.post("/drain")
def drain():
    """由 Control Node 呼叫，開始執行優雅排空"""
    global current_status
    current_status = "draining"
    logger.info("🚰 收到 Drain 指令，停止接收新請求，等待手上任務排空...")
    return {"ok": True}

@app.get("/metrics")
def metrics():
    """回報節點的狀態與負載，Control Node 將以這裡的 status 為準"""
    if not engine: return {}
    
    # [新增] 安全讀取累積數據
    with metrics_lock:
        inf_time = cumulative_metrics["effective_inference_time"]

    with engine.lock:
        running_cnt = len(engine.running_queue)
        waiting_cnt = len(engine.request_queue)
        current_mode = "merge" if engine.current_merged_adapter else "unmerge"
        
        all_reqs = engine.running_queue + engine.request_queue
        
        request_set = []
        for req in all_reqs:
            gen_count = len(req.get("tokens_gen", []))
            remaining = max(0, FIXED_OUTPUT_LEN - gen_count)
            request_set.append({
                "adapter_id": req["adapter_id"],
                "remaining_tokens": remaining
            })
            
        loaded_adapters = list(engine.cpu_cache.keys())
        running_adapters_list = list({str(r["adapter_id"]) for r in all_reqs})
        current_max_batch = engine.merged_capacity if engine.current_merged_adapter else engine.unmerged_capacity

    return {
        "node_id": NODE_ID,
        "mode": current_mode,
        "status": current_status,  # 回報當前狀態，讓 Control Node 同步
        "request_set": request_set, 
        "load": {
            "running_batch": running_cnt + waiting_cnt,
            "waiting_queue": waiting_cnt
        },
        "lora_state": {
            "merged_adapter": engine.current_merged_adapter,
            "running_adapters": running_adapters_list,
            "loaded_adapters": loaded_adapters
        },
        "capacity": {
            "max_batch_size": current_max_batch,
            "max_cpu_loras": engine.max_cpu_loras
        },
        "idle": engine.is_idle(),
        "config_version": last_config_version,
        # [新增] 將指標回報給 Control Node
        "metrics": {
            "effective_inference_time": inf_time
        }
    }

@app.post("/sync_adapters")
def sync_adapters(req: SyncAdaptersRequest):
    global last_config_version
    with config_lock:
        if req.version_id <= last_config_version:
            return {"status": "ignored"}
        last_config_version = req.version_id

    try:
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
    # [安全防護] 只有 Active 狀態才允許接收新 Request
    if current_status != "active":
        raise HTTPException(status_code=503, detail=f"Node is not active (current: {current_status})")

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
    if current_status != "active": return {"status": "error", "reason": "Not active"}
    try:
        engine.merge_adapter(req.adapter_id, force=req.force)
        return {"status": "merged", "adapter": req.adapter_id}
    except Exception as e:
        raise HTTPException(400, f"Merge failed: {e}")

@app.post("/unmerge")
def unmerge(req: UnmergeRequest):
    if current_status != "active": return {"status": "error", "reason": "Not active"}
    engine.unmerge_all()
    return {"status": "unmerged"}

@app.post("/debug/reset")
def debug_reset():
    logger.warning("🚨 NODE RESET TRIGGERED! Clearing local queues...")
    with engine.lock:
        engine.request_queue.clear()
        engine.running_queue.clear()
    engine.unmerge_all()
    with stream_lock:
        stream_queues.clear()
        decoding_state.clear()
    return {"status": "node_reset_complete"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
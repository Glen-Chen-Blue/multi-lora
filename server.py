import threading
import time
import uuid
from queue import Queue
from typing import Dict, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from multilora_system import MultiLoRAEngine


# ============================================================
# Engine init
# ============================================================
engine = MultiLoRAEngine(
    model_id="unsloth/Meta-Llama-3.1-8B",
    adapter_slots=4,
    max_batch_size=4,
    enable_monitor=False,  # 要看 RAM/VRAM 就改 True
)
engine.load_adapters_to_cpu("./testLoRA")


# ============================================================
# Wakeup event (避免 idle busy loop)
# ============================================================
engine_wakeup = threading.Event()


# ============================================================
# Streaming infra
# ============================================================
# request_id -> Queue[Union[str, dict, None]]
stream_queues: Dict[str, Queue] = {}
# request_id -> accumulated text
full_text_buffer: Dict[str, str] = {}

stream_lock = threading.Lock()


# ============================================================
# Token decoding helper
# ============================================================
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


# ============================================================
# Engine callbacks
# ============================================================
def on_token(request_id: str, token_id: int):
    text = _safe_decode_token(token_id)
    if text == "":
        return

    with stream_lock:
        q = stream_queues.get(request_id)
        if request_id in full_text_buffer:
            full_text_buffer[request_id] += text

    if q is not None:
        # streaming delta
        q.put(text)


def on_finish(request_id: str, reason: str):
    with stream_lock:
        q = stream_queues.get(request_id)
        full_text = full_text_buffer.get(request_id, "")

    if q is not None:
        # final full text event
        q.put({
            "type": "final",
            "text": full_text,
            "reason": reason,
        })
        q.put(None)


engine.on_token = on_token
engine.on_finish = on_finish


# ============================================================
# Engine loop thread
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


threading.Thread(target=engine_loop, daemon=True).start()


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="MultiLoRA Server")


class AddRequestBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: Optional[int] = 128


class MergeBody(BaseModel):
    adapter_id: str
    force: Optional[bool] = False


# ============================================================
# SSE stream
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

            # final full text
            if isinstance(item, dict) and item.get("type") == "final":
                yield "event: final\n"
                yield f"data: {item['text']}\n\n"
                continue

            # normal token delta
            yield f"data: {item}\n\n"

    finally:
        with stream_lock:
            stream_queues.pop(request_id, None)
            full_text_buffer.pop(request_id, None)


# ============================================================
# APIs
# ============================================================
@app.post("/add_request")
def add_request(req: AddRequestBody):
    request_id = str(uuid.uuid4())

    # ⚠️ 先建立 queue，避免 engine 先吐 token
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
    if (not req.force) and (not engine.is_idle()):
        raise HTTPException(
            status_code=409,
            detail="Engine not idle. Merge allowed only when idle (or force=true)."
        )
    try:
        engine.merge_adapter(req.adapter_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "merged": req.adapter_id}


@app.post("/unmerge")
def unmerge(force: bool = False):
    if (not force) and (not engine.is_idle()):
        raise HTTPException(
            status_code=409,
            detail="Engine not idle. Unmerge allowed only when idle (or force=true)."
        )
    engine.unmerge_all()
    return {"status": "ok", "merged": None}


@app.get("/status")
def status():
    return engine.get_system_status()

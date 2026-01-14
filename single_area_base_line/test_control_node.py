import os
import uuid
import random
import logging
import json
import asyncio
import httpx
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Control] %(message)s")
logger = logging.getLogger("ControlNode")

# [Requirement] Random Dispatch, No EFO
NODES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",")]
client = httpx.AsyncClient(timeout=60.0)

# 用於轉發 Stream 的 Queue
# rid -> asyncio.Queue
relay_queues: Dict[str, asyncio.Queue] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Control Node Started. Nodes: {NODES}")
    yield
    await client.aclose()

app = FastAPI(lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

async def forward_task(rid: str, node_url: str, payload: dict):
    q = relay_queues[rid]
    try:
        async with client.stream("POST", f"{node_url}/add_request", json=payload) as resp:
            if resp.status_code != 200:
                await q.put(f"[ERROR] Node {node_url} returned {resp.status_code}")
                await q.put(None)
                return

            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    # 轉發原始 data 行
                    content = line[len("data:"):].strip()
                    if content == "[DONE]":
                        await q.put(None)
                        break
                    # 為了保持格式一致，我們直接把 compute node 的 json 轉發出去
                    if content:
                        await q.put(content)
    except Exception as e:
        logger.error(f"Forward failed: {e}")
        await q.put(f"[ERROR] {str(e)}")
        await q.put(None)

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    node_url = random.choice(NODES)
    
    logger.info(f"🎲 Dispatching {rid} -> {node_url} (Adapter: {req.adapter_id})")
    
    q = asyncio.Queue()
    relay_queues[rid] = q
    
    # 啟動背景任務轉發
    asyncio.create_task(forward_task(rid, node_url, req.dict()))
    
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str):
    if request_id not in relay_queues:
        raise HTTPException(404, "Request ID not found")
    
    q = relay_queues[request_id]
    
    async def gen():
        yield "event: open\ndata: ok\n\n"
        try:
            while True:
                data = await q.get()
                if data is None:
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                # data 已經是 json string 或是 error string
                yield f"data: {data}\n\n"
        finally:
            del relay_queues[request_id]

    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
import os
import time
import uuid
import asyncio
import httpx
import logging
import shutil
from collections import deque, defaultdict
from typing import Dict, List, Optional, Deque, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL: %(message)s")
logger = logging.getLogger("ControlNode")

# 設定 LoRA 存放路徑
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")

# [修正] 對應測試腳本的命名 LoRA_1, LoRA_2...
ALLOWED_ADAPTERS = ["LoRA_1", "LoRA_2", "LoRA_3"]

# 初始節點列表
INITIAL_NODES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",") if x.strip()]

limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# Global State
# ============================================================
class NodeManager:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {} # url -> metrics_data
        for url in INITIAL_NODES:
            self.register_node(url)

    def register_node(self, url: str):
        if url not in self.nodes:
            self.nodes[url] = {"metrics": None, "last_seen": 0}
            logger.info(f"✅ Registered Node: {url}")

    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()

    def get_available_nodes(self) -> List[str]:
        """Phase 1: 簡單過濾可用節點"""
        available = []
        for url, data in self.nodes.items():
            m = data.get("metrics")
            if m:
                current_load = m["load"]["running_batch"]
                max_load = m["capacity"]["max_batch_size"]
                if current_load < max_load:
                    available.append(url)
        return available
    
    def get_status_snapshot(self) -> Dict[str, Any]:
        """[新增] 提供給 /status 使用的狀態快照"""
        active_count = len(self.get_available_nodes())
        return {
            "active_nodes": active_count,
            "total_nodes": len(self.nodes),
            "nodes_detail": self.nodes
        }

node_mgr = NodeManager()
request_queues: Dict[str, Deque] = defaultdict(deque)
stream_queues: Dict[str, asyncio.Queue] = {}

# ============================================================
# Background Tasks
# ============================================================

async def metrics_poller():
    while True:
        for url in list(node_mgr.nodes.keys()):
            try:
                # [修正] 放寬 Timeout 避免繁忙時誤報
                resp = await client.get(f"{url}/metrics", timeout=5.0)
                if resp.status_code == 200:
                    node_mgr.update_metrics(url, resp.json())
            except Exception as e:
                # 只在連線真的掛掉時報錯，避免刷屏
                pass 
                # logger.warning(f"⚠️ Failed to fetch metrics from {url}")
        await asyncio.sleep(1)

async def dispatch_task(node_url: str, req_data: dict):
    rid = req_data["request_id"]
    user_q = stream_queues.get(rid)
    
    try:
        payload = {
            "prompt": req_data["prompt"],
            "adapter_id": req_data["adapter_id"],
            "max_new_tokens": req_data["max_new_tokens"]
        }
        
        async with client.stream("POST", f"{node_url}/add_request", json=payload) as response:
            if response.status_code != 200:
                logger.error(f"❌ Dispatch failed to {node_url}: {response.status_code}")
                if user_q: await user_q.put({"type": "error", "message": f"Node error {response.status_code}"})
                return

            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content and user_q:
                        await user_q.put(content)
                        
    except Exception as e:
        logger.error(f"❌ Error during dispatch to {node_url}: {e}")
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None) 

async def scheduler_loop():
    logger.info("⏳ Scheduler loop started (1s interval).")
    while True:
        available_nodes = node_mgr.get_available_nodes()
        
        for adapter_id, queue in list(request_queues.items()):
            while queue and available_nodes:
                target_node = available_nodes[0] 
                req = queue.popleft()
                logger.info(f"🚀 Dispatching Req {req['request_id']} ({req['adapter_id']}) -> {target_node}")
                asyncio.create_task(dispatch_task(target_node, req))
                
                # 簡單 Round-Robin: 移到隊尾
                available_nodes.pop(0) 

        await asyncio.sleep(1.0)

# ============================================================
# API Endpoints
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    asyncio.create_task(metrics_poller())
    asyncio.create_task(scheduler_loop())
    yield
    await client.aclose()

app = FastAPI(title="Control Node (Phase 1)", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.post("/register_node")
async def register_node(request: Request):
    data = await request.json()
    url = data.get("control_node_url") or str(request.base_url)
    node_mgr.register_node(url)
    return {"status": "registered", "url": url}

@app.get("/status")
async def status():
    """[新增] 回傳 Cluster 狀態，解決測試腳本 404 問題"""
    return node_mgr.get_status_snapshot()

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    stream_queues[rid] = asyncio.Queue()
    
    request_queues[req.adapter_id].append({
        "request_id": rid,
        "prompt": req.prompt,
        "adapter_id": req.adapter_id,
        "max_new_tokens": req.max_new_tokens
    })
    
    logger.info(f"📥 Received Request {rid} for {req.adapter_id}. Queued.")
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str):
    if request_id not in stream_queues:
        raise HTTPException(404, "Request ID not found")
    
    q = stream_queues[request_id]
    
    async def event_generator():
        yield "event: open\ndata: connected\n\n"
        try:
            while True:
                data = await q.get()
                if data is None: 
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                
                if isinstance(data, dict):
                    import json
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"data: {data}\n\n"
        finally:
            stream_queues.pop(request_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/fetch_adapter/{adapter_id}")
async def fetch_adapter(adapter_id: str):
    target_path = os.path.join(LORA_PATH, adapter_id, "adapter_model.safetensors")
    
    if os.path.exists(target_path):
        return FileResponse(target_path, media_type="application/octet-stream", filename="adapter_model.safetensors")
    
    logger.error(f"❌ Adapter not found: {target_path}")
    raise HTTPException(404, f"Adapter {adapter_id} not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
import os
import time
import httpx
import logging
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ============================================================
# Logging Config (徹底靜音模式)
# ============================================================
class EndpointFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "GET /status" not in msg

# 1. 設定基礎 Log Level
logging.basicConfig(level=logging.INFO)

# 2. 靜音 httpx (關鍵修正：防止 HTTP Request 洗版)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 3. 過濾 uvicorn 對 status 的存取紀錄
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

logger = logging.getLogger("EFO")

# ============================================================
# Config
# ============================================================
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")

# Global State
ALL_AVAILABLE_LORAS = []
registered_nodes = {} # {url: last_seen}
lora_assignment = {} # {adapter_id: node_url}
node_assignments = {} # {node_url: [adapter_ids]}

client = httpx.AsyncClient(timeout=None)

# ============================================================
# Helpers
# ============================================================
def scan_available_loras(base_path: str = LORA_PATH) -> List[str]:
    """
    輕量化掃描：只檢查資料夾結構和 Safetensors 檔案存在，不載入模型。
    """
    if not os.path.exists(base_path):
        logger.warning(f"⚠️ LoRA path '{base_path}' not found.")
        return []

    found_ids = []
    # 掃描資料夾
    try:
        adapters = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        for folder in adapters:
            st_path = os.path.join(base_path, folder, "adapter_model.safetensors")
            if os.path.exists(st_path):
                # 依照使用者習慣解析 ID (例如: chat_lora -> chat)
                aid = str(folder.split("_")[-1]) if "_" in folder else str(folder)
                found_ids.append(aid)
    except Exception as e:
        logger.error(f"Error scanning LoRAs: {e}")
    
    logger.info(f"✅ Scanned Adapters: {found_ids}")
    return found_ids

def rebalance():
    global lora_assignment, node_assignments
    nodes = list(registered_nodes.keys())
    if not nodes or not ALL_AVAILABLE_LORAS:
        lora_assignment = {}
        node_assignments = {}
        return

    new_assign = {}
    new_node_map = {n: [] for n in nodes}
    
    # Round-Robin 分配
    for i, aid in enumerate(ALL_AVAILABLE_LORAS):
        node = nodes[i % len(nodes)]
        new_assign[aid] = node
        new_node_map[node].append(aid)
    
    lora_assignment = new_assign
    node_assignments = new_node_map
    logger.info(f"⚖️ Rebalanced: {node_assignments}")

# Init
ALL_AVAILABLE_LORAS = scan_available_loras()

# ============================================================
# API
# ============================================================
app = FastAPI(title="EFO Server")

class RegisterBody(BaseModel):
    control_node_url: str

class RelayBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.get("/status")
def status():
    return {
        "adapters": ALL_AVAILABLE_LORAS,
        "nodes": list(registered_nodes.keys()),
        "assignments": node_assignments
    }

@app.post("/register_node")
def register(body: RegisterBody):
    url = body.control_node_url
    if url not in registered_nodes:
        logger.info(f"New Node: {url}")
    registered_nodes[url] = time.time()
    rebalance()
    return {"status": "ok", "assigned_adapters": node_assignments.get(url, [])}

@app.post("/relay_request")
async def relay(req: RelayBody):
    target_node = lora_assignment.get(req.adapter_id)
    
    if not target_node:
        # Fallback: 如果該 LoRA 沒被分配 (例如剛掃描到但沒 Node)，隨機找一個 Node 試試
        if registered_nodes:
            target_node = list(registered_nodes.keys())[0]
        else:
            raise HTTPException(503, "No available nodes")

    async def proxy():
        try:
            # Step 1: 取得 Request ID
            resp = await client.post(f"{target_node}/send_request", 
                                     json={"prompt": req.prompt, "adapter_id": req.adapter_id, "max_new_tokens": req.max_new_tokens})
            resp.raise_for_status()
            rid = resp.json()["request_id"]
            
            # Step 2: 串流轉發
            async with client.stream("GET", f"{target_node}/stream/{rid}") as r:
                async for line in r.aiter_lines():
                    if line: yield f"{line}\n"
        except Exception as e:
            yield f"event: end\ndata: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(proxy(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
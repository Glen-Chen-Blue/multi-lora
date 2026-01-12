import os
import time
import httpx
import logging
import asyncio
import json
import random
from typing import Dict, List, Set, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ============================================================
# Config
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EFO")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
AFFINITY_FILE = "lora_affinity.json"

# ============================================================
# Global State
# ============================================================
class GlobalState:
    def __init__(self):
        self.all_loras: List[str] = []
        self.affinity_table: Dict[str, List[str]] = {}
        self.minimal_set: List[str] = []
        
        self.registered_nodes: Dict[str, float] = {} # url -> last_heartbeat
        self.node_assignments: Dict[str, List[str]] = {} # node -> allowed_adapters
        self.lora_routing: Dict[str, str] = {} # adapter -> target_node
        
        # [NEW] Config Versioning
        self.config_version: int = 0

state = GlobalState()
client = httpx.AsyncClient(timeout=10.0)

# ============================================================
# Affinity Logic
# ============================================================
def scan_loras():
    if not os.path.exists(LORA_PATH): return []
    try:
        dirs = [d for d in os.listdir(LORA_PATH) if os.path.isdir(os.path.join(LORA_PATH, d))]
        return sorted([d.split("_")[-1] if "_" in d else d for d in dirs])
    except Exception:
        return []

def update_affinity():
    loras = scan_loras()
    if not loras: return
    
    state.all_loras = loras
    
    # 模擬生成 Affinity Table
    table = {}
    for aid in loras:
        subs = [aid]
        others = [x for x in loras if x != aid]
        if others and random.random() > 0.7:
            subs.append(random.choice(others))
        table[aid] = list(set(subs))
    
    state.affinity_table = table
    with open(AFFINITY_FILE, "w") as f:
        json.dump(table, f, indent=4)
        
    # 計算最小覆蓋集
    universe = set(loras)
    covered = set()
    selected = []
    
    while covered != universe:
        best_cand = None
        best_cover_diff = set()
        
        for cand in loras:
            can_cover = {target for target, subs in table.items() if cand in subs}
            diff = can_cover - covered
            if len(diff) > len(best_cover_diff):
                best_cand = cand
                best_cover_diff = diff
        
        if not best_cand: break
        selected.append(best_cand)
        covered.update(best_cover_diff)
            
    state.minimal_set = selected
    logger.info(f"Updated logic. Adapters: {len(loras)}, Minimal Set: {selected}")

# ============================================================
# Rebalance & Broadcast
# ============================================================
async def broadcast_config():
    tasks = []
    # [NEW] Payload includes version_id
    payload_base = {
        "affinity_table": state.affinity_table,
        "minimal_set": state.minimal_set,
        "version_id": state.config_version
    }
    
    for node, allowed in state.node_assignments.items():
        payload = payload_base.copy()
        payload["assigned_adapters"] = allowed
        logger.info(f"Pushing config v{state.config_version} to {node} (Adapters: {len(allowed)})")
        tasks.append(client.post(f"{node}/update_config", json=payload))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

def rebalance_assignments():
    nodes = list(state.registered_nodes.keys())
    if not nodes: return

    # [NEW] Generate new version ID (Timestamp-based to prevent restart collisions)
    state.config_version = int(time.time() * 1000)

    new_node_map = {n: [] for n in nodes}
    new_routing = {}
    
    # 1. Assign Minimal Set
    for i, expert in enumerate(state.minimal_set):
        target = nodes[i % len(nodes)]
        new_node_map[target].append(expert)
        
        for aid, subs in state.affinity_table.items():
            if expert in subs:
                if aid not in new_routing:
                    new_routing[aid] = target
    
    # 2. Fallback
    for aid in state.all_loras:
        if aid not in new_routing:
            new_routing[aid] = nodes[0]
            
    state.node_assignments = new_node_map
    state.lora_routing = new_routing
    
    asyncio.create_task(broadcast_config())

# ============================================================
# Background Tasks
# ============================================================
async def monitor_nodes():
    while True:
        await asyncio.sleep(10)
        now = time.time()
        dead = [url for url, ts in state.registered_nodes.items() if now - ts > 30]
        
        if dead:
            logger.warning(f"Removing dead nodes: {dead}")
            for d in dead:
                del state.registered_nodes[d]
                state.node_assignments.pop(d, None)
            rebalance_assignments()

# ============================================================
# API
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    update_affinity()
    asyncio.create_task(monitor_nodes())
    yield
    await client.aclose()

app = FastAPI(title="EFO Server", lifespan=lifespan)

class RegisterBody(BaseModel):
    control_node_url: str

class RelayBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.post("/register_node")
async def register(body: RegisterBody):
    url = body.control_node_url
    is_new = url not in state.registered_nodes
    state.registered_nodes[url] = time.time()
    
    if is_new:
        logger.info(f"New node registered: {url}")
        rebalance_assignments()
    
    return {"status": "registered"}

@app.post("/heartbeat")
async def heartbeat(body: RegisterBody):
    if body.control_node_url in state.registered_nodes:
        state.registered_nodes[body.control_node_url] = time.time()
    return {"status": "ok"}

@app.post("/relay_request")
async def relay_request(req: RelayBody):
    target_node = state.lora_routing.get(req.adapter_id)
    
    if not target_node and state.registered_nodes:
        target_node = list(state.registered_nodes.keys())[0]
        
    if not target_node:
        raise HTTPException(503, "No available Control Nodes")

    async def proxy():
        try:
            # [修正] 改用 client.post 取代 client.stream，因為 send_request 不是串流接口
            # 原始錯誤的寫法: async with client.stream("POST", ... ) as r: ...
            
            r = await client.post(f"{target_node}/send_request", json=req.dict())
            
            if r.status_code != 200:
                yield f"event: error\ndata: {json.dumps('Relay failed')}\n\n"
                return
            
            resp_json = r.json()
            rid = resp_json.get("request_id")
            
            if not rid:
                yield "event: error\ndata: \"No Request ID\"\n\n"
                return
            
            # 這裡保持用 stream，因為 /stream/{rid} 是真正的串流接口
            async with client.stream("GET", f"{target_node}/stream/{rid}") as s:
                 async for line in s.aiter_lines():
                     if line: yield f"{line}\n"

        except Exception as e:
            # 這裡就是原本捕捉到 "Attempted to call a sync iterator..." 的地方
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(proxy(), media_type="text/event-stream")

@app.get("/status")
def status():
    return {
        "nodes": list(state.registered_nodes.keys()),
        "loras": state.all_loras,
        "assignments": state.node_assignments,
        "routing": state.lora_routing,
        "version": state.config_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
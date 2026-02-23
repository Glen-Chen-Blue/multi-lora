import os
import time
import httpx
import logging
import asyncio
import json
import random
from typing import Dict, List, Set, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EFO")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
MAPPING_FILE = "lora_mapping.json"

# [實驗開關] 
ENABLE_SEMANTIC = os.environ.get("ENABLE_SEMANTIC", "true").lower() == "true"

class GlobalState:
    def __init__(self):
        self.all_loras: List[str] = []
        self.affinity_table: Dict[str, List[str]] = {}
        self.minimal_set: List[str] = []
        self.lora_map_data: Dict[str, Dict] = {} 
        self.registered_nodes: Dict[str, float] = {} 
        self.node_assignments: Dict[str, List[str]] = {} 
        self.lora_routing: Dict[str, str] = {} 
        self.node_area_map: Dict[str, str] = {} 
        self.lora_types: Dict[str, str] = {} 
        self.config_version: int = 0

state = GlobalState()
client = httpx.AsyncClient(timeout=10.0)

def load_mapping_and_affinity():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                data = json.load(f)
                state.lora_map_data = data.get("lora_map", {})
            
            state.all_loras = sorted(list(state.lora_map_data.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
            table = {}
            if ENABLE_SEMANTIC:
                # 正常模式：讀取替代品
                for aid, info in state.lora_map_data.items():
                    state.lora_types[aid] = info.get("type", "global")
                    subs = info.get("substitutes", [])
                    valid_subs = [s for s in subs if s in state.lora_map_data]
                    table[aid] = list(set([aid] + valid_subs))
                logger.info("🧠 Semantic Optimization: ENABLED")
            else:
                # 實驗模式：強制 1-to-1 (無替代品)，確保所有 LoRA 都被分派
                for aid, info in state.lora_map_data.items():
                    state.lora_types[aid] = info.get("type", "global")
                    table[aid] = [aid] 
                logger.info("🧠 Semantic Optimization: DISABLED (Full Set Mode)")
            
            state.affinity_table = table
            
            # 計算 Minimal Set (如果 No-Semantic，這就是 Full Set)
            global_loras = [aid for aid in state.all_loras if state.lora_types.get(aid) == "global"]
            calculate_minimal_set(global_loras, state.affinity_table)
            
            logger.info(f"✅ Loaded Mapping. Virtual Adapters: {len(state.all_loras)}")
            return
        except Exception as e:
            logger.error(f"❌ Failed to load {MAPPING_FILE}: {e}")

def calculate_minimal_set(universe_list, table):
    universe = set(universe_list)
    covered = set()
    selected = []
    
    # Greedy Set Cover
    while covered != universe:
        best_cand = None
        best_cover_diff = set()
        candidates = universe_list
        
        for cand in candidates:
            can_serve = {target for target, subs in table.items() if cand in subs and target in universe}
            diff = can_serve - covered
            if len(diff) > len(best_cover_diff):
                best_cand = cand
                best_cover_diff = diff
        
        if not best_cand: 
            remaining = universe - covered
            selected.extend(list(remaining))
            break
            
        selected.append(best_cand)
        covered.update(best_cover_diff)
            
    state.minimal_set = selected
    logger.info(f"Updated Minimal Set: {len(selected)} items")

async def broadcast_config():
    tasks = []
    payload_base = {
        "affinity_table": state.affinity_table,
        "minimal_set": state.minimal_set,
        "lora_types": state.lora_types, 
        "version_id": state.config_version
    }
    
    for node, allowed in state.node_assignments.items():
        payload = payload_base.copy()
        payload["assigned_adapters"] = allowed
        tasks.append(client.post(f"{node}/update_config", json=payload))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

def rebalance_assignments():
    nodes = list(state.registered_nodes.keys())
    if not nodes: return

    state.config_version = int(time.time() * 1000)
    new_node_map = {n: [] for n in nodes}
    new_routing = {}
    
    # 1. Global LoRA 分配 (Round-Robin)
    for i, expert in enumerate(state.minimal_set):
        target = nodes[i % len(nodes)]
        new_node_map[target].append(expert)
        
        for aid, subs in state.affinity_table.items():
            if expert in subs and state.lora_types.get(aid) == "global":
                if aid not in new_routing:
                    new_routing[aid] = target

    # 2. Local & Default Routing
    for aid in state.all_loras:
        if aid not in new_routing and state.lora_types.get(aid) == "global":
            target = nodes[0]
            new_routing[aid] = target
            
    state.node_assignments = new_node_map
    state.lora_routing = new_routing
    
    asyncio.create_task(broadcast_config())

async def monitor_nodes():
    while True:
        await asyncio.sleep(10)
        now = time.time()
        dead = [url for url, ts in state.registered_nodes.items() if now - ts > 30]
        if dead:
            for d in dead:
                del state.registered_nodes[d]
                state.node_assignments.pop(d, None)
            rebalance_assignments()

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    load_mapping_and_affinity()
    asyncio.create_task(monitor_nodes())
    yield
    await client.aclose()

app = FastAPI(title="EFO Server", lifespan=lifespan)

class RegisterBody(BaseModel):
    control_node_url: str
    area_id: str = "1"

class RelayBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.post("/register_node")
async def register(body: RegisterBody):
    url = body.control_node_url
    is_new = url not in state.registered_nodes
    state.registered_nodes[url] = time.time()
    state.node_area_map[url] = body.area_id
    
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
    async def proxy():
        yield f"event: open\ndata: ok\n\n"
        yield f"data: {json.dumps({'type': 'info', 'message': 'EFO Cloud Exec'})}\n\n"
        yield f"event: end\ndata: [DONE]\n\n"
    return StreamingResponse(proxy(), media_type="text/event-stream")

@app.get("/fetch_adapter/{adapter_id}")
def fetch_adapter(adapter_id: str):
    # 這裡簡化：所有虛擬 ID 都回傳同一個實體檔案
    target_path = None
    if adapter_id in state.lora_map_data:
        info = state.lora_map_data[adapter_id]
        source_path = info.get("source_path", "")
        folder_name = os.path.basename(source_path)
        target_path = os.path.join(LORA_PATH, folder_name, "adapter_model.safetensors")
    
    if target_path and os.path.exists(target_path):
        return FileResponse(target_path, media_type="application/octet-stream", filename="adapter_model.safetensors")
    raise HTTPException(404, "Adapter not found")

@app.get("/status")
def status():
    return {
        "nodes": list(state.registered_nodes.keys()),
        "loras": len(state.all_loras),
        "semantic_enabled": ENABLE_SEMANTIC
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)
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

# ============================================================
# Config
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EFO")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
AFFINITY_FILE = "lora_affinity.json"
MAPPING_FILE = "lora_mapping.json"

# ============================================================
# Global State
# ============================================================
class GlobalState:
    def __init__(self):
        self.all_loras: List[str] = []
        self.affinity_table: Dict[str, List[str]] = {}
        self.minimal_set: List[str] = []
        self.lora_map_data: Dict[str, Dict] = {} 
        
        self.registered_nodes: Dict[str, float] = {} 
        self.node_assignments: Dict[str, List[str]] = {} 
        self.lora_routing: Dict[str, str] = {} 
        
        # [新增] 紀錄每個節點的區域 ID: url -> area_id
        self.node_area_map: Dict[str, str] = {} 
        # [新增] 快取所有 LoRA 的 Type: aid -> type
        self.lora_types: Dict[str, str] = {} 

        self.config_version: int = 0

state = GlobalState()
client = httpx.AsyncClient(timeout=10.0)

# ============================================================
# Affinity & Mapping Logic
# ============================================================
def load_mapping_and_affinity():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                data = json.load(f)
                state.lora_map_data = data.get("lora_map", {})
            
            state.all_loras = sorted(list(state.lora_map_data.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
            table = {}
            for aid, info in state.lora_map_data.items():
                # [新增] 讀取並儲存 Type
                state.lora_types[aid] = info.get("type", "global")
                
                subs = info.get("substitutes", [])
                valid_subs = [s for s in subs if s in state.lora_map_data]
                table[aid] = list(set([aid] + valid_subs))
            
            state.affinity_table = table
            
            # 計算 Global 模型的 Minimal Set (Local 模型必須強制分配，不需要計算覆蓋)
            global_loras = [aid for aid in state.all_loras if state.lora_types.get(aid) == "global"]
            calculate_minimal_set(global_loras, state.affinity_table)
            
            logger.info(f"✅ Loaded Mapping. Virtual Adapters: {len(state.all_loras)}")
            return
        except Exception as e:
            logger.error(f"❌ Failed to load {MAPPING_FILE}: {e}")

    logger.warning("⚠️ No mapping file found. Falling back to physical directory scan.")
    scan_physical_loras()

def scan_physical_loras():
    if not os.path.exists(LORA_PATH): 
        os.makedirs(LORA_PATH, exist_ok=True)
        return

    try:
        dirs = [d for d in os.listdir(LORA_PATH) if os.path.isdir(os.path.join(LORA_PATH, d))]
        loras = sorted([d.split("_")[-1] if "_" in d else d for d in dirs])
        state.all_loras = loras
        
        table = {}
        for aid in loras:
            table[aid] = [aid]
            state.lora_types[aid] = "global" # 預設為 Global
        
        state.affinity_table = table
        calculate_minimal_set(loras, table)
    except Exception as e:
        logger.error(f"Scan failed: {e}")

def calculate_minimal_set(universe_list, table):
    universe = set(universe_list)
    covered = set()
    selected = []
    
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
    logger.info(f"Updated Minimal Set (Global Only): {len(selected)} items")

# ============================================================
# Rebalance & Broadcast
# ============================================================
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
    
    # [修改] 混合分配策略
    # 1. Global LoRA: 使用 Round-Robin 分配 Minimal Set
    for i, expert in enumerate(state.minimal_set):
        target = nodes[i % len(nodes)]
        new_node_map[target].append(expert)
        
        # 建立 Routing 表 (只針對 Global)
        for aid, subs in state.affinity_table.items():
            if expert in subs and state.lora_types.get(aid) == "global":
                if aid not in new_routing:
                    new_routing[aid] = target

    # 2. Local LoRA: 強制分配給該區域的所有節點
    # 這裡我們遍歷所有 LoRA，如果是 Local，就加給對應 Area 的節點
    for aid in state.all_loras:
        l_type = state.lora_types.get(aid, "global")
        
        if l_type != "global":
            # 這是區域專用模型
            target_area = l_type
            assigned_nodes = []
            
            for node_url in nodes:
                node_area = state.node_area_map.get(node_url, "1")
                if node_area == target_area:
                    # 這是我的區域，必須分配給我
                    if aid not in new_node_map[node_url]:
                        new_node_map[node_url].append(aid)
                    assigned_nodes.append(node_url)
            
            # Local Routing 可以在這裡做，但通常由 Local Control Node 內部處理
            # 如果需要跨節點 (同一區內)，可以隨機選一個
            if assigned_nodes:
                 new_routing[aid] = assigned_nodes[0]

    # 3. 確保所有 Global LoRA 都有預設路由
    for aid in state.all_loras:
        if state.lora_types.get(aid) == "global" and aid not in new_routing:
            target = nodes[0]
            new_routing[aid] = target
            
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
            for d in dead:
                del state.registered_nodes[d]
                state.node_assignments.pop(d, None)
                state.node_area_map.pop(d, None)
            rebalance_assignments()

# ============================================================
# API
# ============================================================
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
    area_id: str = "1" # [新增] 區域 ID

class RelayBody(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.post("/register_node")
async def register(body: RegisterBody):
    url = body.control_node_url
    is_new = url not in state.registered_nodes
    state.registered_nodes[url] = time.time()
    
    # [新增] 紀錄區域
    state.node_area_map[url] = body.area_id
    
    if is_new:
        logger.info(f"New node registered: {url} (Area: {body.area_id})")
        rebalance_assignments()
    return {"status": "registered"}

@app.post("/heartbeat")
async def heartbeat(body: RegisterBody):
    if body.control_node_url in state.registered_nodes:
        state.registered_nodes[body.control_node_url] = time.time()
    return {"status": "ok"}

@app.post("/relay_request")
async def relay_request(req: RelayBody):
    """
    接收 Control Node 無法處理的請求並進行轉發 (模擬 Offloading/Cloud Execution)。
    """
    logger.info(f"☁️ Received OFFLOAD Request for {req.adapter_id}")
    
    target_node = state.lora_routing.get(req.adapter_id)
    if not target_node and state.registered_nodes:
        target_node = list(state.registered_nodes.keys())[0]

    if not target_node:
        raise HTTPException(503, "No nodes available for relay")

    async def proxy():
        try:
            # 這裡模擬 EFO/Cloud 處理請求
            yield f"event: open\ndata: ok\n\n"
            yield f"data: {json.dumps({'type': 'info', 'message': 'Executed by EFO/Cloud'})}\n\n"
            yield f"data: {json.dumps(' [EFO_Cloud_Response] ')}\n\n"
            yield f"event: end\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(proxy(), media_type="text/event-stream")

@app.get("/fetch_adapter/{adapter_id}")
def fetch_adapter(adapter_id: str):
    target_path = None
    if adapter_id in state.lora_map_data:
        info = state.lora_map_data[adapter_id]
        source_path = info.get("source_path", "")
        folder_name = os.path.basename(source_path)
        target_path = os.path.join(LORA_PATH, folder_name, "adapter_model.safetensors")
        logger.info(f"🔍 Mapping fetch: {adapter_id} -> {folder_name}")
    
    if not target_path or not os.path.exists(target_path):
        fallback_path = os.path.join(LORA_PATH, adapter_id, "adapter_model.safetensors")
        if os.path.exists(fallback_path): target_path = fallback_path

    if target_path and os.path.exists(target_path):
        return FileResponse(target_path, media_type="application/octet-stream", filename="adapter_model.safetensors")
    
    raise HTTPException(404, f"Adapter {adapter_id} file not found in EFO.")

@app.get("/status")
def status():
    return {
        "nodes": list(state.registered_nodes.keys()),
        "loras_count": len(state.all_loras),
        "assignments": state.node_assignments,
        "node_areas": state.node_area_map, # Debug info
        "version": state.config_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
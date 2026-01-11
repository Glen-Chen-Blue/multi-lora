import os
import time
import httpx
import logging
import asyncio
import json
import random
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ============================================================
# Logging Config
# ============================================================
class EndpointFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "GET /status" not in msg and "POST /heartbeat" not in msg

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logger = logging.getLogger("EFO")

# ============================================================
# Config & Global State
# ============================================================
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
AFFINITY_FILE = "lora_affinity.json"

ALL_AVAILABLE_LORAS = []
lora_affinity_table = {}  # {target_lora: [compatible_lora1, ...]}
minimal_lora_set = []     # 最小涵蓋集合

registered_nodes = {}  # {url: last_heartbeat_time}
lora_assignment = {} 
node_assignments = {} 

client = httpx.AsyncClient(timeout=5.0)

# ============================================================
# Affinity & Coverage Logic
# ============================================================
def generate_affinity_table(lora_ids: List[str]) -> Dict[str, List[str]]:
    table = {}
    for aid in lora_ids:
        substitutes = [aid]
        others = [x for x in lora_ids if x != aid]
        if others and random.random() > 0.5:
            substitutes.extend(random.sample(others, min(len(others), 1)))
        table[aid] = list(set(substitutes))
    
    with open(AFFINITY_FILE, "w") as f:
        json.dump(table, f, indent=4)
    logger.info(f"🧬 Affinity table generated and saved to {AFFINITY_FILE}")
    return table

def find_minimal_coverage(lora_ids: List[str], affinity_table: Dict[str, List[str]]) -> List[str]:
    universe = set(lora_ids)
    covered = set()
    selected_loras = []

    provider_coverage = {aid: set() for aid in lora_ids}
    for target, subs in affinity_table.items():
        for s in subs:
            provider_coverage[s].add(target)

    while covered != universe:
        best_provider = max(
            provider_coverage, 
            key=lambda k: len(provider_coverage[k] - covered),
            default=None
        )
        if not best_provider or not (provider_coverage[best_provider] - covered):
            break
            
        selected_loras.append(best_provider)
        covered.update(provider_coverage[best_provider])
             
    logger.info(f"✨ Minimal experts identified: {selected_loras}")
    return selected_loras

# ============================================================
# Core Helpers
# ============================================================
def scan_available_loras(base_path: str = LORA_PATH) -> List[str]:
    if not os.path.exists(base_path):
        return []
    found_ids = []
    try:
        adapters = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        for folder in adapters:
            if os.path.exists(os.path.join(base_path, folder, "adapter_model.safetensors")):
                aid = str(folder.split("_")[-1]) if "_" in folder else str(folder)
                found_ids.append(aid)
    except Exception as e:
        logger.error(f"Error scanning LoRAs: {e}")
    return found_ids

async def broadcast_updates():
    """
    主動將最新的分配結果推播給所有存活的 Control Nodes
    """
    logger.info("📡 Broadcasting updates to all control nodes...")
    tasks = []
    
    for node_url in list(registered_nodes.keys()):
        assigned = node_assignments.get(node_url, [])
        payload = {
            "assigned_adapters": assigned,
            "affinity_table": lora_affinity_table,
            "minimal_set": minimal_lora_set
        }
        
        async def push(url, data):
            try:
                await client.post(f"{url}/update_config", json=data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to push config to {url}: {e}")

        tasks.append(push(node_url, payload))
    
    if tasks:
        await asyncio.gather(*tasks)

def rebalance(broadcast: bool = True):
    global lora_assignment, node_assignments
    nodes = list(registered_nodes.keys())
    if not nodes or not ALL_AVAILABLE_LORAS:
        lora_assignment, node_assignments = {}, {}
        return

    new_assign = {}
    new_node_map = {n: [] for n in nodes}
    expert_nodes = {} # {expert_id: node_url}

    # [MODIFIED] 1. 只分配「最小專家集合」給節點 (決定要 Load 什麼)
    # 這樣 Control Node 收到的 assigned_adapters 就只會包含 minimal set
    for i, aid in enumerate(minimal_lora_set):
        node = nodes[i % len(nodes)]
        new_node_map[node].append(aid)
        new_assign[aid] = node
        expert_nodes[aid] = node
        
    # [MODIFIED] 2. 處理剩下的非 Expert Adapters (決定路由去哪)
    # 根據 Affinity Table，將非 Expert 的請求導向擁有其替代品(Expert)的節點
    for aid in ALL_AVAILABLE_LORAS:
        if aid in new_assign: continue # 已經分配過了 (是 Expert)
        
        # 找替代品
        substitutes = lora_affinity_table.get(aid, [])
        target_node = None
        
        # 嘗試在已分配的 Experts 中找替代品
        for sub in substitutes:
            if sub in expert_nodes:
                target_node = expert_nodes[sub]
                break
        
        # 如果找不到 (理論上 find_minimal_coverage 保證找得到)，回退到隨機分配
        if not target_node:
            target_node = nodes[0] 
        
        new_assign[aid] = target_node
    
    lora_assignment = new_assign
    node_assignments = new_node_map

    if broadcast:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(broadcast_updates())
        except RuntimeError:
            logger.error("❌ Failed to schedule broadcast: No running event loop!")

# Init
ALL_AVAILABLE_LORAS = scan_available_loras()
lora_affinity_table = generate_affinity_table(ALL_AVAILABLE_LORAS)
minimal_lora_set = find_minimal_coverage(ALL_AVAILABLE_LORAS, lora_affinity_table)

# ============================================================
# API Endpoints
# ============================================================
app = FastAPI(title="EFO Server with Affinity")

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
        "minimal_experts": minimal_lora_set,
        "affinity_table": lora_affinity_table,
        "nodes": list(registered_nodes.keys()),
        "assignments": node_assignments
    }

@app.post("/register_node")
async def register(body: RegisterBody):
    url = body.control_node_url
    current_time = time.time()
    
    is_new = url not in registered_nodes
    registered_nodes[url] = current_time
    
    if is_new:
        logger.info(f"🆕 New node registered: {url}")
        rebalance(broadcast=False) 
        await broadcast_updates()
    
    return {"status": "registered", "wait_for_push": True}

@app.post("/heartbeat")
async def heartbeat(body: RegisterBody):
    url = body.control_node_url
    if url in registered_nodes:
        registered_nodes[url] = time.time()
    else:
        await register(body)
    return {"status": "ok"}

@app.post("/relay_request")
async def relay(req: RelayBody):
    target_node = lora_assignment.get(req.adapter_id)
    if not target_node:
        target_node = list(registered_nodes.keys())[0] if registered_nodes else None
    
    if not target_node: raise HTTPException(503, "No available nodes")

    async def proxy():
        try:
            resp = await client.post(f"{target_node}/send_request", 
                                     json={"prompt": req.prompt, "adapter_id": req.adapter_id, "max_new_tokens": req.max_new_tokens})
            resp.raise_for_status()
            rid = resp.json()["request_id"]
            async with client.stream("GET", f"{target_node}/stream/{rid}") as r:
                async for line in r.aiter_lines():
                    if line: yield f"{line}\n"
        except Exception as e:
            yield f"event: end\ndata: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(proxy(), media_type="text/event-stream")

# ============================================================
# Background Tasks
# ============================================================
async def node_monitor():
    while True:
        await asyncio.sleep(10)
        now = time.time()
        dead = [url for url, ts in registered_nodes.items() if now - ts > 30]
        if dead:
            logger.info(f"💀 Pruning dead nodes: {dead}")
            for d in dead:
                del registered_nodes[d]
                node_assignments.pop(d, None)
            rebalance(broadcast=True)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(node_monitor())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
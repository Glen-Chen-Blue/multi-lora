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
        return "GET /status" not in msg

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

registered_nodes = {} 
lora_assignment = {} 
node_assignments = {} 

client = httpx.AsyncClient(timeout=None)

# ============================================================
# Affinity & Coverage Logic
# ============================================================
def generate_affinity_table(lora_ids: List[str]) -> Dict[str, List[str]]:
    """
    模擬 Embedding 計算：建立 LoRA 之間的替代關係表。
    """
    table = {}
    for aid in lora_ids:
        # 每個 LoRA 至少可以被自己替代
        substitutes = [aid]
        # 隨機模擬 50% 機率存在其他語意相近的替代品
        others = [x for x in lora_ids if x != aid]
        if others and random.random() > 0.5:
            substitutes.extend(random.sample(others, min(len(others), 1)))
        table[aid] = list(set(substitutes))
    
    with open(AFFINITY_FILE, "w") as f:
        json.dump(table, f, indent=4)
    logger.info(f"🧬 Affinity table generated and saved to {AFFINITY_FILE}")
    return table

def find_minimal_coverage(lora_ids: List[str], affinity_table: Dict[str, List[str]]) -> List[str]:
    """
    使用貪婪演算法求解 Set Cover 問題，找出最少專家集合。
    """
    universe = set(lora_ids)
    covered = set()
    selected_loras = []

    # 建立每個 LoRA 作為提供者能涵蓋哪些需求
    provider_coverage = {aid: set() for aid in lora_ids}
    for target, subs in affinity_table.items():
        for s in subs:
            provider_coverage[s].add(target)

    while covered != universe:
        # 選擇能涵蓋最多「尚未涵蓋任務」的 LoRA
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

def rebalance():
    global lora_assignment, node_assignments
    nodes = list(registered_nodes.keys())
    if not nodes or not ALL_AVAILABLE_LORAS:
        lora_assignment, node_assignments = {}, {}
        return

    new_assign = {}
    new_node_map = {n: [] for n in nodes}
    for i, aid in enumerate(ALL_AVAILABLE_LORAS):
        node = nodes[i % len(nodes)]
        new_assign[aid] = node
        new_node_map[node].append(aid)
    
    lora_assignment = new_assign
    node_assignments = new_node_map

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
def register(body: RegisterBody):
    url = body.control_node_url
    registered_nodes[url] = time.time()
    rebalance()
    # 每次註冊時分發親和力表格與最少集合資訊
    return {
        "status": "ok", 
        "assigned_adapters": node_assignments.get(url, []),
        "affinity_table": lora_affinity_table,
        "minimal_set": minimal_lora_set
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
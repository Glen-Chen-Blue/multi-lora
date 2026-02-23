import os
import json
import logging
import asyncio
import httpx
import random
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EFO: %(message)s")
logger = logging.getLogger("EFOServer")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
LORA_METADATA_PATH = os.environ.get("LORA_METADATA", "./LoRA_metadata.json")
CLUSTERS_ENV = os.environ.get("CLUSTERS", "{}")

# ============================================================
# Global State
# ============================================================
global_lora_metadata: Dict[str, Any] = {}
configured_clusters: Dict[str, str] = {}  # 從環境變數讀取的預期名單
active_clusters: Dict[str, str] = {}      # 實際已經來註冊的活躍名單

# 記錄哪些 LoRA 目前存在於哪個 Cluster 的 Disk 上 (步驟一)
global_lora_disk_inventory: Dict[str, List[str]] = {}

# 紀錄各個 Cluster 中各個 LoRA 的歷史需求量 (時間序列)
# 結構: historical_demand[cluster_name][lora_id] = [count_t1, count_t2, ...]
historical_demand: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

# ============================================================
# SP2: Global Routing & Offloading (10s interval)
# ============================================================
def generate_delays(cluster_names: List[str]) -> Dict[str, Dict[str, int]]:
    """
    產生 Cluster 之間的隨機網路延遲 (50ms ~ 500ms)
    保證雙向延遲對稱 (A->B == B->A)，且自身延遲為 0
    """
    delays = {c: {} for c in cluster_names}
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            c1 = cluster_names[i]
            c2 = cluster_names[j]
            if c1 == c2:
                delays[c1][c2] = 0
            else:
                d = random.randint(50, 500)
                delays[c1][c2] = d
                delays[c2][c1] = d
    return delays

async def sync_global_routing():
    """
    每 10 秒向所有 Control Node 獲取可用容量 (budget) 與 LoRA 狀態，
    並附加上叢集間的網路延遲，統整後廣播給所有 Cluster。
    """
    if not active_clusters:
        return

    cluster_names = list(active_clusters.keys())
    delays = generate_delays(cluster_names)
    routing_table = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        # 1. 向各個 Control Node 獲取狀態 (呼叫尚未實作的 API: GET /offload_status)
        for cluster_name, url in active_clusters.items():
            try:
                resp = await client.get(f"{url}/offload_status")
                if resp.status_code == 200:
                    data = resp.json()
                    routing_table[cluster_name] = {
                        "ip": url,
                        "budget": data.get("budget", 0),
                        "lora_status": data.get("lora_status", {
                            "merged": [],
                            "loaded": [],
                            "unloaded": []
                        }),
                        "delay": delays[cluster_name]
                    }
                else:
                    logger.warning(f"⚠️ [Routing] Failed to get status from {cluster_name}: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"❌ [Routing] Error getting status from {cluster_name}: {e}")

        # 若完全沒有抓到任何節點的狀態，則跳過廣播
        if not routing_table:
            return

        # 2. 將整理好的 Routing Table 廣播給所有 Control Node (呼叫尚未實作的 API: POST /update_global_routing)
        for cluster_name, url in active_clusters.items():
            try:
                await client.post(f"{url}/update_global_routing", json={"routing_table": routing_table})
                logger.info(f"🌐 [Routing] Broadcasted global routing table to {cluster_name}")
            except Exception as e:
                logger.error(f"❌ [Routing] Error broadcasting to {cluster_name}: {e}")


# ============================================================
# SP1: Forecasting & Provisioning (60s interval)
# ============================================================
async def fetch_cluster_stats():
    """從各個 active cluster 拉取上一週期的 LoRA 請求統計，並加入歷史紀錄中"""
    if not active_clusters or not global_lora_metadata:
        return
        
    async with httpx.AsyncClient(timeout=5.0) as client:
        for cluster_name, url in active_clusters.items():
            try:
                resp = await client.get(f"{url}/pop_lora_stats")
                if resp.status_code == 200:
                    data = resp.json()
                    stats = data.get("stats", {})
                    
                    for lora_id in global_lora_metadata.keys():
                        count = stats.get(lora_id, 0)
                        historical_demand[cluster_name][lora_id].append(count)
                        
                    active_stats = {k: v for k, v in stats.items() if v > 0}
                    if active_stats:
                        logger.info(f"📊 Fetched stats from {cluster_name}: {active_stats}")
                else:
                    logger.warning(f"⚠️ Failed to fetch stats from {cluster_name}: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"❌ Error fetching stats from {cluster_name}: {e}")

def hybrid_forecasting_engine():
    """預測下一時段的 LoRA 需求 (假的預測引擎，未來會使用 historical_demand 進行計算)"""
    pass

async def run_sp1_provisioning():
    """長週期預配置決策 (SP1: LoRA Provisioning)"""
    if not global_lora_metadata or not active_clusters:
        return

    all_loras = list(global_lora_metadata.keys())
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for cluster_name, url in active_clusters.items():
            global_lora_disk_inventory[cluster_name] = all_loras.copy()
            try:
                await client.post(
                    f"{url}/update_local_loras",
                    json={"loras": all_loras}
                )
                logger.info(f"✅ Provisioned {len(all_loras)} LoRAs to {cluster_name}")
            except Exception as e:
                logger.error(f"❌ Failed to provision LoRAs to {cluster_name}: {e}")

# ============================================================
# Background Tasks
# ============================================================
async def sp2_routing_loop():
    """SP2: 每 10 秒處理一次跨區卸載的路由狀態收集與廣播"""
    logger.info("⏳ SP2 Global Routing Loop started (10s interval).")
    while True:
        await sync_global_routing()
        await asyncio.sleep(10)

async def sp1_provisioning_loop():
    """SP1: 每 60 秒抓取統計、預測並重新配置"""
    logger.info("⏳ SP1 Provisioning Loop started (60s interval).")
    while True:
        await fetch_cluster_stats()
        hybrid_forecasting_engine()
        await run_sp1_provisioning()
        await asyncio.sleep(60)

# ============================================================
# API Models
# ============================================================
class RegisterClusterRequest(BaseModel):
    cluster_name: str
    control_node_url: str

# ============================================================
# Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_lora_metadata, configured_clusters
    try:
        configured_clusters = json.loads(CLUSTERS_ENV)
        logger.info(f"📂 Loaded expected cluster topology from env: {list(configured_clusters.keys())}")
    except json.JSONDecodeError:
        logger.error("❌ Failed to parse CLUSTERS environment variable.")
    
    if os.path.exists(LORA_METADATA_PATH):
        try:
            with open(LORA_METADATA_PATH, "r", encoding="utf-8") as f:
                global_lora_metadata = json.load(f)
            logger.info(f"✅ Loaded LoRA metadata from {LORA_METADATA_PATH}")
        except Exception as e:
            logger.error(f"❌ Error loading LoRA metadata: {e}")
    else:
        logger.warning(f"⚠️ LoRA metadata file not found at {LORA_METADATA_PATH}")
    
    # 將任務分開，確保互相不阻塞
    asyncio.create_task(sp2_routing_loop())
    asyncio.create_task(sp1_provisioning_loop())
    
    yield
        

app = FastAPI(title="Edge Federation Orchestrator (EFO)", lifespan=lifespan)

# ============================================================
# API Routes
# ============================================================
@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    """
    供 Control Node 在啟動時呼叫。
    註冊其 URL，並獲取全域的 LoRA Metadata。
    """
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered and Active: {req.cluster_name} -> {req.control_node_url}")
    
    asyncio.create_task(run_sp1_provisioning())
    
    return {
        "status": "ok",
        "cluster_name": req.cluster_name,
        "metadata": global_lora_metadata
    }

@app.get("/status")
async def get_status():
    """查看 EFO 狀態"""
    history_lengths = {
        cluster: {lora: len(history) for lora, history in loras.items()}
        for cluster, loras in historical_demand.items()
    }
    
    return {
        "configured_clusters": configured_clusters,
        "active_clusters": active_clusters,
        "total_loras": len(global_lora_metadata),
        "metadata_keys": global_lora_metadata,
        "disk_inventory": global_lora_disk_inventory,
        "history_lengths": history_lengths
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9100))
    uvicorn.run(app, host="0.0.0.0", port=port)
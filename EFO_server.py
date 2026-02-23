import os
import json
import logging
import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import Dict, Any, List

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

# ============================================================
# SP1: Forecasting & Provisioning
# ============================================================
def hybrid_forecasting_engine():
    """預測下一時段的 LoRA 需求 (假的預測引擎，先直接 pass)"""
    pass

async def run_sp1_provisioning():
    """長週期預配置決策 (SP1: LoRA Provisioning)"""
    # 只針對「已經完成註冊」的 active_clusters 進行分配
    if not global_lora_metadata or not active_clusters:
        return

    # 先直接將所有 LoRA 分配給所有活躍的 Control Node
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

async def efo_background_loop():
    """EFO 背景排程迴圈"""
    while True:
        hybrid_forecasting_engine()
        await run_sp1_provisioning()
        await asyncio.sleep(10)  # 每 10 秒執行一次 SP1 配置與廣播

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
    
    # 啟動 EFO 的預測與分配背景任務
    asyncio.create_task(efo_background_loop())
    
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
    # 將註冊的節點加入「活躍名單」
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered and Active: {req.cluster_name} -> {req.control_node_url}")
    
    # 註冊成功後，立刻觸發一次分配給這個剛上線的節點
    asyncio.create_task(run_sp1_provisioning())
    
    return {
        "status": "ok",
        "cluster_name": req.cluster_name,
        "metadata": global_lora_metadata
    }

@app.get("/status")
async def get_status():
    """查看 EFO 狀態"""
    return {
        "configured_clusters": configured_clusters,
        "active_clusters": active_clusters,
        "total_loras": len(global_lora_metadata),
        "metadata_keys": global_lora_metadata,
        "disk_inventory": global_lora_disk_inventory
    }

if __name__ == "__main__":
    import uvicorn
    # 預設使用 test_start.sh 給予的 9100 port
    port = int(os.environ.get("PORT", 9100))
    uvicorn.run(app, host="0.0.0.0", port=port)
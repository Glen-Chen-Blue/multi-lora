import os
import json
import logging
import asyncio
import httpx
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Set
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel

# 匯入集中管理的設定
from config import (
    LORA_METADATA_PATH, LOG_PATH,
    LORA_SIZE_GB, DISK_CAPACITY_GB,
    SP1_INTERVAL_SECONDS
)

# ============================================================
# Config & Logging
# ============================================================
class MetricsAccessFilter(logging.Filter):
    def filter(self, record):
        return "/cluster_metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EFO(dLoRA-LFU): %(message)s")
logger = logging.getLogger("EFOServer_dLoRA")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(MetricsAccessFilter())
CLUSTERS_ENV = os.environ.get("CLUSTERS", "{}")

# ============================================================
# 📊 EFO Global Metrics State
# ============================================================
class EFOMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        # 累計下載次數 (反映 Cache Miss 的網路成本)
        self.artifact_downloads = defaultdict(int)

efo_metrics = EFOMetrics()

# ============================================================
# Global State & System Variables
# ============================================================
global_lora_metadata: Dict[str, Any] = {}
configured_clusters: Dict[str, str] = {}
active_clusters: Dict[str, str] = {}

# dLoRA Baseline 狀態追蹤器 (LFU: Least Frequently Used)
cluster_disk_state: Dict[str, Set[str]] = {}             # 記錄每個 Cluster 磁碟上有哪些 LoRA
cluster_lora_freq: Dict[str, Dict[str, float]] = {}      # 記錄每個 Cluster 中各 LoRA 的歷史請求頻率

current_time_step = 0
system_start_event: asyncio.Event = None
efo_logging_task: Optional[asyncio.Task] = None

# ============================================================
# dLoRA Historical Frequency (LFU) Management
# ============================================================
def init_cluster_dlora(cluster_name: str):
    """初始化 Cluster 的磁碟狀態，強制放入 Local LoRAs (不可驅逐)"""
    if cluster_name not in cluster_disk_state:
        cluster_disk_state[cluster_name] = set()
        cluster_lora_freq[cluster_name] = defaultdict(float)
        
        for lora_id, info in global_lora_metadata.items():
            if info.get("type") == "local" and info.get("cluster") == cluster_name:
                cluster_disk_state[cluster_name].add(lora_id)
                cluster_lora_freq[cluster_name][lora_id] = 999999.0  # Local 模型給予極高權重保護

def get_max_capacity() -> int:
    return max(1, int(DISK_CAPACITY_GB / LORA_SIZE_GB))

# ============================================================
# 📊 Global Metrics Logging Cycle
# ============================================================
async def run_efo_metrics_cycle(step_id: int):
    """定期拉取 Control Node 數據並寫入 log"""
    logger.info(f"📊 [EFO Metrics] Starting cycle for Time Step {step_id}")
    os.makedirs(LOG_PATH, exist_ok=True)
    log_file = f"{LOG_PATH}/efo_global_metrics.log"
    
    logging_interval = SP1_INTERVAL_SECONDS / 10.0
    
    try:
        for sub_step in range(10):
            await asyncio.sleep(logging_interval)
            
            global_snapshot = {
                "timestamp": time.time(),
                "step_id": step_id,
                "sub_step": sub_step + 1,
                "clusters": {},
                "efo_totals": {
                    "total_inference_time": 0.0,
                    "total_drops": 0,
                    "total_drop_local_congestion": 0,
                    "total_drop_no_target": 0,
                    "total_offloads": 0,
                    "total_local_completed": 0,
                    "total_offload_completed": 0,
                    "artifact_downloads": 0,
                    "total_stored_loras": 0
                }
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for cluster_name, url in active_clusters.items():
                    try:
                        resp = await client.get(f"{url}/cluster_metrics")
                        if resp.status_code == 200:
                            data = resp.json()
                            global_snapshot["clusters"][cluster_name] = data
                            
                            d_local = data.get("drop_local_congestion", 0)
                            d_no_tgt = data.get("drop_no_target", 0)

                            global_snapshot["efo_totals"]["total_inference_time"] += data.get("total_effective_inference_time", 0.0)
                            global_snapshot["efo_totals"]["total_drop_local_congestion"] += d_local
                            global_snapshot["efo_totals"]["total_drop_no_target"] += d_no_tgt
                            global_snapshot["efo_totals"]["total_drops"] += (d_local + d_no_tgt)
                            global_snapshot["efo_totals"]["total_offloads"] += data.get("offload_out", 0)
                            global_snapshot["efo_totals"]["total_local_completed"] += data.get("local_completed", 0)
                            global_snapshot["efo_totals"]["total_offload_completed"] += data.get("offload_in_completed", 0)
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch metrics from {cluster_name}: {e}")
            
            async with efo_metrics.lock:
                global_snapshot["efo_totals"]["artifact_downloads"] = sum(efo_metrics.artifact_downloads.values())
                global_snapshot["efo_totals"]["total_stored_loras"] = sum(len(c) for c in cluster_disk_state.values())
                
                for c_name in active_clusters.keys():
                    if c_name not in global_snapshot["clusters"]:
                        global_snapshot["clusters"][c_name] = {}
                    
                    global_snapshot["clusters"][c_name]["artifact_downloads"] = efo_metrics.artifact_downloads.get(c_name, 0)
                    global_snapshot["clusters"][c_name]["total_stored_loras"] = len(cluster_disk_state.get(c_name, []))
            
            with open(log_file, "a") as f:
                f.write(json.dumps(global_snapshot) + "\n")
                
        logger.info(f"📊 [EFO Metrics] Step {step_id} finished (10/10). Waiting for next Time Edge...")
        
    except asyncio.CancelledError:
        logger.info(f"📊 [EFO Metrics] Step {step_id} cancelled (New Time Edge arrived).")
        raise

# ============================================================
# API Routes (dLoRA Historical Frequency Specific)
# ============================================================
class DLoRARequest(BaseModel):
    cluster_name: str
    lora_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_lora_metadata, configured_clusters, system_start_event
    system_start_event = asyncio.Event()

    try: configured_clusters = json.loads(CLUSTERS_ENV)
    except: pass
    
    if os.path.exists(LORA_METADATA_PATH):
        with open(LORA_METADATA_PATH, "r", encoding="utf-8") as f:
            global_lora_metadata = json.load(f)
            logger.info(f"📂 Loaded LoRA metadata for {len(global_lora_metadata)} LoRAs.")
            
    yield
        
app = FastAPI(title="Edge Federation Orchestrator (dLoRA Baseline)", lifespan=lifespan)

@app.post("/access_lora")
async def access_lora(req: DLoRARequest):
    """記錄請求歷史頻率"""
    freq_map = cluster_lora_freq.get(req.cluster_name)
    if freq_map is not None:
        freq_map[req.lora_id] += 1.0
    return {"status": "ok"}

@app.post("/fetch_and_evict_lora")
async def fetch_and_evict_lora(req: DLoRARequest):
    """模擬下載並執行基於歷史頻率的驅逐 (LFU)"""
    cluster = req.cluster_name
    lora_id = req.lora_id
    
    disk = cluster_disk_state.get(cluster)
    freq_map = cluster_lora_freq.get(cluster)
    
    if disk is None or freq_map is None:
        return {"status": "error", "message": "Cluster not found"}
        
    freq_map[lora_id] += 1.0  # 增加歷史存取權重
        
    if lora_id in disk:
        return {"status": "ok", "evicted": None, "downloaded": False}
        
    # 加入磁碟
    disk.add(lora_id)
    
    # 增加模擬下載成本
    async with efo_metrics.lock:
        efo_metrics.artifact_downloads[cluster] += 1
        
    # 檢查容量限制，執行 LFU Eviction
    CAPACITY = get_max_capacity()
    evicted = None
    
    if len(disk) > CAPACITY:
        # 找出磁碟中「非 Local」且「歷史頻率最低」的 LoRA 踢出
        min_freq = float('inf')
        victim = None
        
        for k in disk:
            info = global_lora_metadata.get(k, {})
            is_local = (info.get("type") == "local" and info.get("cluster") == cluster)
            
            if not is_local:
                if freq_map[k] < min_freq:
                    min_freq = freq_map[k]
                    victim = k
                    
        if victim:
            evicted = victim
            disk.remove(victim)
                
    logger.info(f"💾 [LFU] {cluster} fetched {lora_id} | Evicted: {evicted} | Disk Size: {len(disk)}/{CAPACITY}")
    return {"status": "ok", "evicted": evicted, "downloaded": True, "current_cache": list(disk)}

class RegisterClusterRequest(BaseModel):
    cluster_name: str
    control_node_url: str

@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    active_clusters[req.cluster_name] = req.control_node_url
    
    init_cluster_dlora(req.cluster_name)
    initial_cache = list(cluster_disk_state[req.cluster_name])
    
    logger.info(f"🔗 Cluster Registered: {req.cluster_name}. Initial disk size: {len(initial_cache)}")
    return {"status": "ok", "metadata": global_lora_metadata, "initial_cache": initial_cache}

@app.post("/time_edge")
async def trigger_time_edge():
    """觸發時間推進，並將歷史頻率衰減 (Time Decay)，確保近期流量更具影響力"""
    global current_time_step, efo_logging_task
    
    if not system_start_event.is_set():
        system_start_event.set()
        logger.info("🚀 System initialized via first /time_edge trigger!")

    logger.info(f"\n{'='*50}\n⏱️ [TIME EDGE] Advancing to Time Step {current_time_step}\n{'='*50}")
    
    # 執行頻率衰減 (Time Decay = 0.5)
    for c_name, freq_map in cluster_lora_freq.items():
        for lora_id in freq_map.keys():
            info = global_lora_metadata.get(lora_id, {})
            if not (info.get("type") == "local" and info.get("cluster") == c_name):
                freq_map[lora_id] *= 0.5
    
    if efo_logging_task and not efo_logging_task.done():
        efo_logging_task.cancel()
        try: await efo_logging_task
        except asyncio.CancelledError: pass
            
    efo_logging_task = asyncio.create_task(run_efo_metrics_cycle(current_time_step))
    
    completed_step = current_time_step
    current_time_step += 1
    
    return {"status": "success", "completed_step": completed_step, "next_step": current_time_step}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9100)))
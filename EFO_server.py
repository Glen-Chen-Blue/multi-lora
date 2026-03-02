import os
import csv
import json
import logging
import asyncio
import httpx
import time
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel

# 匯入集中管理的設定
from config import (
    LORA_PATH, LORA_METADATA_PATH,
    LORA_MAPPING_PATH, SIMULATION_DATA_CSV_PATH,
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB, COST_INST_LOCAL,
    COST_NET_TRAFFIC, COST_DROP_PENALTY, LORA_SIZE_GB,
    DISK_CAPACITY_GB, T_MAX_SLO, SWAP_EPSILON,
    NETWORK_SIM_PARAMS, EDGE_SYNC_TIMEOUT,
    SP1_INTERVAL_SECONDS, SP2_INTERVAL_SECONDS
)

# ============================================================
# Config & Logging
# ============================================================
class MetricsAccessFilter(logging.Filter):
    def filter(self, record):
        return "/cluster_metrics" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EFO: %(message)s")
logger = logging.getLogger("EFOServer")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(MetricsAccessFilter())
CLUSTERS_ENV = os.environ.get("CLUSTERS", "{}")

# ============================================================
# 📊 EFO Global Metrics State
# ============================================================
class EFOMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.artifact_downloads = 0  # 累計下載次數 (對應 J_net,artifact)

efo_metrics = EFOMetrics()

# ============================================================
# Global State & System Variables
# ============================================================
global_lora_metadata: Dict[str, Any] = {}
configured_clusters: Dict[str, str] = {}  # 預期名單
active_clusters: Dict[str, str] = {}      # 活躍名單

global_lora_disk_inventory: Dict[str, List[str]] = {}
predicted_demand: Dict[str, Dict[str, float]] = defaultdict(dict)
azure_mapping: Dict[str, Dict[str, str]] = {}

current_time_step = 0  # CSV 的起始行數 (Time step)
system_start_event: asyncio.Event = None

# ============================================================
# Network Simulator (Shifted Lognormal & P95)
# ============================================================
class NetworkSimulator:
    def __init__(self):
        self.params = NETWORK_SIM_PARAMS
        self.matrix = {}
        for (c1, c2), (d_prop, mu, sigma) in self.params.items():
            self.matrix[(c1, c2)] = (d_prop, mu, sigma)
            self.matrix[(c2, c1)] = (d_prop, mu, sigma)
        for c in ["cluster_1", "cluster_2", "cluster_3"]:
            self.matrix[(c, c)] = (0, 0, 0)

    def get_delay(self, src: str, dest: str) -> float:
        if src == dest: return 0.0
        d_prop, mu, sigma = self.matrix.get((src, dest), (50, 3.0, 1.0))
        return d_prop + np.random.lognormal(mu, sigma)

    def get_p95_info(self) -> Dict[str, Dict[str, float]]:
        p95_delays = {}
        clusters = ["cluster_1", "cluster_2", "cluster_3"]
        for c1 in clusters:
            p95_delays[c1] = {}
            for c2 in clusters:
                if c1 == c2:
                    p95_delays[c1][c2] = 0.0
                else:
                    d_prop, mu, sigma = self.matrix.get((c1, c2), (50, 3.0, 1.0))
                    p95_jitter = np.exp(mu + 1.645 * sigma)
                    p95_delays[c1][c2] = round(d_prop + p95_jitter, 2)
        return p95_delays

network_simulator = NetworkSimulator()

# ============================================================
# SP2: Global Routing Broadcast (Short interval)
# ============================================================
async def sync_global_routing():
    if not active_clusters: return
    p95_delays = network_simulator.get_p95_info()
    routing_table = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for cluster_name, url in active_clusters.items():
            try:
                resp = await client.get(f"{url}/offload_status")
                if resp.status_code == 200:
                    data = resp.json()
                    routing_table[cluster_name] = {
                        "ip": url,
                        "budget": data.get("budget", 0),
                        "lora_status": data.get("lora_status", {"merged": [], "loaded": [], "unloaded": []}),
                        "delay": p95_delays.get(cluster_name, {})
                    }
            except Exception as e:
                logger.error(f"❌ [Routing] Error getting status from {cluster_name}: {e}")

        if not routing_table: return

        for cluster_name, url in active_clusters.items():
            try:
                await client.post(f"{url}/update_global_routing", json={"routing_table": routing_table})
            except Exception as e:
                pass

# ============================================================
# 📊 Global Metrics Polling (60s interval)
# ============================================================
async def poll_global_metrics():
    await system_start_event.wait()
    logger.info("📊 Global Metrics Polling started (60s interval).")
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/efo_global_metrics.log"
    
    while True:
        await asyncio.sleep(60)
        global_snapshot = {
            "timestamp": time.time(),
            "clusters": {},
            "efo_totals": {
                "total_inference_time": 0.0,
                "total_drops": 0,
                "total_offloads": 0,
                "total_local_completed": 0,
                "total_offload_completed": 0,
                "artifact_downloads": 0,
                "total_stored_loras": sum(len(loras) for loras in global_lora_disk_inventory.values())
            }
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for cluster_name, url in active_clusters.items():
                try:
                    resp = await client.get(f"{url}/cluster_metrics")
                    if resp.status_code == 200:
                        data = resp.json()
                        global_snapshot["clusters"][cluster_name] = data
                        global_snapshot["efo_totals"]["total_inference_time"] += data.get("total_effective_inference_time", 0.0)
                        global_snapshot["efo_totals"]["total_drops"] += (data.get("drop_queue", 0) + data.get("drop_slo", 0))
                        global_snapshot["efo_totals"]["total_offloads"] += data.get("offload_out", 0)
                        global_snapshot["efo_totals"]["total_local_completed"] += data.get("local_completed", 0)
                        global_snapshot["efo_totals"]["total_offload_completed"] += data.get("offload_in_completed", 0)
                except Exception as e:
                    logger.error(f"❌ Failed to fetch metrics from {cluster_name}: {e}")
        
        async with efo_metrics.lock:
            global_snapshot["efo_totals"]["artifact_downloads"] = efo_metrics.artifact_downloads
        
        with open(log_file, "a") as f:
            f.write(json.dumps(global_snapshot) + "\n")

# ============================================================
# SP1: CSV Forecasting & Provisioning (Step-based)
# ============================================================
def exact_csv_forecasting(time_step: int):
    """
    從 CSV (Trace) 中讀取真實數據，精準計算出 time_step 對應的區間內，
    每個 Cluster 對各個 LoRA 的真實 Request 數量。
    """
    global predicted_demand
    predicted_demand.clear()
    
    # 這裡的常數需與 test_simulation.py 保持對齊
    START_OFFSET = 86400 * 2
    
    # 定義此 time_step 的秒數區間 (這與 test_simulation.py 的 req_interval 完美對齊)
    start_sec = time_step * SP1_INTERVAL_SECONDS
    end_sec = (time_step + 1) * SP1_INTERVAL_SECONDS
    
    # 初始化字典
    for cluster_name in active_clusters.keys():
        predicted_demand[cluster_name] = {lora_id: 0.0 for lora_id in global_lora_metadata.keys()}

    if not os.path.exists(SIMULATION_DATA_CSV_PATH):
        logger.error(f"❌ CSV not found at {SIMULATION_DATA_CSV_PATH}. Using 0 demand for all.")
        return

    with open(SIMULATION_DATA_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                arr_ts = float(row["arrive_timestamp"])
                arr_sec = arr_ts - START_OFFSET
                
                # 判斷是否落在當前要預測的區間內
                if start_sec <= arr_sec < end_sec:
                    cluster = row["cluster"]
                    lora_id_raw = row["lora_id"]
                    lora_id = f"LoRA_{lora_id_raw}" # 轉成系統用的名稱，例如 LoRA_1
                    
                    if cluster in active_clusters and lora_id in predicted_demand[cluster]:
                        predicted_demand[cluster][lora_id] += 1.0
            except (ValueError, KeyError):
                continue
                
    # 印出 Log 確認讀取結果
    for cluster, demands in predicted_demand.items():
        total_reqs = sum(demands.values())
        active_loras = sum(1 for v in demands.values() if v > 0)
        logger.info(f"📈 [CSV Forecast] {cluster} step {time_step} ({start_sec}s-{end_sec}s): {total_reqs} reqs, {active_loras} active LoRAs")

async def run_sp1_provisioning_and_wait():
    """SP1: 計算 Adaptive CSG-Swap，下發配置並等待所有節點排空重置"""
    if not global_lora_metadata or not active_clusters: return
    logger.info("⚙️ Running SP1 Adaptive CSG-Swap Provisioning (CSV based)...")

    p95_delays = network_simulator.get_p95_info()
    
    C_STORE = COST_STORE_PER_GB
    C_DL = COST_DOWNLOAD_PER_GB
    C_INST = COST_INST_LOCAL
    C_NET = COST_NET_TRAFFIC
    C_DROP = COST_DROP_PENALTY
    S_LORA = LORA_SIZE_GB
    CAPACITY = int(DISK_CAPACITY_GB / S_LORA)
    T_MAX = T_MAX_SLO
    EPSILON = SWAP_EPSILON

    cluster_targets = {}

    # 1. 計算所有活躍 Cluster 的最優 LoRA 配置
    for cluster_name in active_clusters.keys():
        target_disk = set()
        mandatory_set = set()
        utilities = {}

        valid_loras = []
        for lora_id, info in global_lora_metadata.items():
            if info.get("type") == "global" or (info.get("type") == "local" and info.get("cluster") == cluster_name):
                valid_loras.append(lora_id)

        gains = {}
        for lora_id in valid_loras:
            is_local = (global_lora_metadata[lora_id].get("type") == "local")
            if is_local:
                best_offload_cost = C_DROP
            else:
                offload_costs = []
                for k in active_clusters.keys():
                    if k == cluster_name: continue
                    delay_sec = p95_delays.get(cluster_name, {}).get(k, 1000.0) / 1000.0
                    if delay_sec >= T_MAX:
                        gamma = float('inf')
                    else:
                        gamma = T_MAX / (T_MAX - delay_sec)
                    offload_costs.append(gamma * C_INST + C_NET)
                    
                best_offload_cost = min(offload_costs) if offload_costs else C_DROP
                best_offload_cost = min(best_offload_cost, C_DROP)

            gains[lora_id] = max(0.0, best_offload_cost - C_INST)

        # Step 0: Mandatory Sets
        for lora_id in valid_loras:
            if global_lora_metadata[lora_id].get("type") == "local":
                mandatory_set.add(lora_id)
                target_disk.add(lora_id)

        # Step 1: Evaluation and Eviction
        current_disk = set(global_lora_disk_inventory.get(cluster_name, []))
        for lora_id in current_disk:
            if lora_id in mandatory_set or lora_id not in valid_loras:
                continue
            lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
            u_retention = (lambd * gains[lora_id]) - (S_LORA * C_STORE)
            if u_retention >= 0:
                target_disk.add(lora_id)
                utilities[lora_id] = u_retention

        # Step 2: Iterative Expansion with Swap
        candidates = []
        for lora_id in valid_loras:
            if lora_id not in target_disk and lora_id not in mandatory_set:
                lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
                u_download = (lambd * gains[lora_id]) - (S_LORA * (C_STORE + C_DL))
                if u_download > 0:
                    candidates.append((lora_id, u_download))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for lora_id, u_v in candidates:
            if len(target_disk) < CAPACITY:
                target_disk.add(lora_id)
                utilities[lora_id] = u_v
            else:
                swappable_items = [u for u in target_disk if u not in mandatory_set]
                if not swappable_items: break
                
                u_min_id = min(swappable_items, key=lambda x: utilities[x])
                u_min_val = utilities[u_min_id]

                if (u_v - u_min_val) > EPSILON:
                    target_disk.remove(u_min_id)
                    del utilities[u_min_id]
                    target_disk.add(lora_id)
                    utilities[lora_id] = u_v

        # 計算下載量並印出 Log
        new_downloads = len(target_disk - current_disk)
        if new_downloads > 0:
            async with efo_metrics.lock:
                efo_metrics.artifact_downloads += new_downloads

        cluster_targets[cluster_name] = list(target_disk)
        global_lora_disk_inventory[cluster_name] = cluster_targets[cluster_name]
        logger.info(f"📊 [SP1] {cluster_name}: Target {len(target_disk)} LoRAs (New DLs: {new_downloads})")

    # 2. 發送並「阻塞等待」所有 Control Node 排空與重置
    logger.info("⏳ Dispatching SP1 to Control Nodes and WAITING for system drain & reset...")
    async with httpx.AsyncClient(timeout=EDGE_SYNC_TIMEOUT) as client:
        tasks = []
        for cluster_name, target_loras in cluster_targets.items():
            url = active_clusters[cluster_name]
            payload = {"loras": target_loras}
            # 注意這裡呼叫的是新 API /apply_sp1_and_reset
            tasks.append(client.post(f"{url}/apply_sp1_and_reset", json=payload))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for cluster_name, res in zip(cluster_targets.keys(), results):
                if isinstance(res, Exception):
                    logger.error(f"❌ {cluster_name} apply_sp1_and_reset failed: {res}")
                elif res.status_code != 200:
                    logger.error(f"❌ {cluster_name} returned {res.status_code}: {res.text}")
                else:
                    logger.info(f"✅ {cluster_name} synced successfully.")
    
    logger.info("✨ All clusters have applied SP1 and are ready for the next time step.")

# ============================================================
# Background Tasks (Event Loops)
# ============================================================
async def sp2_routing_loop():
    await system_start_event.wait()
    await sync_global_routing()
    while True:
        await asyncio.sleep(SP2_INTERVAL_SECONDS)
        await sync_global_routing()

# ============================================================
# Lifecycle & API Endpoints
# ============================================================
class RegisterClusterRequest(BaseModel):
    cluster_name: str
    control_node_url: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_lora_metadata, configured_clusters, azure_mapping, system_start_event
    system_start_event = asyncio.Event()

    try:
        configured_clusters = json.loads(CLUSTERS_ENV)
    except: pass
    
    if os.path.exists(LORA_METADATA_PATH):
        with open(LORA_METADATA_PATH, "r", encoding="utf-8") as f:
            global_lora_metadata = json.load(f)
            
    if os.path.exists(LORA_MAPPING_PATH):
        with open(LORA_MAPPING_PATH, "r", encoding="utf-8") as f: 
            azure_mapping = json.load(f)

    asyncio.create_task(sp2_routing_loop())
    asyncio.create_task(poll_global_metrics())
    yield
        
app = FastAPI(title="Edge Federation Orchestrator (EFO)", lifespan=lifespan)

@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered: {req.cluster_name}")
    return {"status": "ok", "metadata": global_lora_metadata}

@app.post("/time_edge")
async def trigger_time_edge():
    """
    接收 test_simulation.py 的步進訊號。
    讀取 CSV -> 計算 SP1 -> 等待所有節點套用 -> 回傳 OK。
    """
    global current_time_step
    
    # 第一次呼叫時，解鎖背景迴圈 (SP2 Routing & Metrics)
    if not system_start_event.is_set():
        system_start_event.set()
        logger.info("🚀 System initialized via first /time_edge trigger!")

    logger.info(f"\n{'='*50}\n⏱️ [TIME EDGE] Advancing to Time Step {current_time_step}\n{'='*50}")
    
    # 1. 精準讀取 CSV (只取 Active Clusters)
    exact_csv_forecasting(current_time_step)
    
    # 2. 執行 SP1 計算，並「等待」所有 Control Node 排空與重置
    await run_sp1_provisioning_and_wait()
    
    # 3. 推進時間步，回傳成功給 test_simulation.py 讓其開始發送請求
    completed_step = current_time_step
    current_time_step += 1
    
    return {
        "status": "success", 
        "completed_step": completed_step,
        "next_step": current_time_step
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9900)))
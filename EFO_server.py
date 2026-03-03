import os
import csv
import json
import logging
import asyncio
import httpx
import time
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
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
    NETWORK_SIM_PARAMS, SP2_INTERVAL_SECONDS, EDGE_SYNC_TIMEOUT,
    SP1_INTERVAL_SECONDS
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
# 📊 EFO Global Metrics State (Modified)
# ============================================================
class EFOMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.artifact_downloads = 0  # 累計下載次數
        self.cumulative_stored_loras = 0  # [新增] 累計 Stored LoRA 數量 (SP1解一次加一次)

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

current_time_step = 0  # 模擬器推進的區間步數
system_start_event: asyncio.Event = None

# [新增] 全域 Log 任務管理
efo_logging_task: Optional[asyncio.Task] = None

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
                pass

        if not routing_table: return

        for cluster_name, url in active_clusters.items():
            try:
                await client.post(f"{url}/update_global_routing", json={"routing_table": routing_table})
            except Exception as e:
                pass

# ============================================================
# 📊 Global Metrics Logging Cycle (Modified)
# ============================================================
async def run_efo_metrics_cycle(step_id: int):
    """
    對應 Time Edge，紀錄 10 次 Global Metrics (間隔 SP1_INTERVAL / 10)
    """
    logger.info(f"📊 [EFO Metrics] Starting cycle for Time Step {step_id}")
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/efo_global_metrics.log"
    
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
                    # [修改] 只保留這兩種 Drop
                    "total_drop_local_congestion": 0,
                    "total_drop_no_target": 0,
                    
                    "total_offloads": 0,
                    "total_local_completed": 0,
                    "total_offload_completed": 0,
                    "artifact_downloads": 0,
                    "total_stored_loras": 0
                }
            }
            
            # 拉取各 Cluster Metrics
            async with httpx.AsyncClient(timeout=10.0) as client:
                for cluster_name, url in active_clusters.items():
                    try:
                        resp = await client.get(f"{url}/cluster_metrics")
                        if resp.status_code == 200:
                            data = resp.json()
                            global_snapshot["clusters"][cluster_name] = data
                            
                            # 讀取並加總 Metrics
                            d_local = data.get("drop_local_congestion", 0)
                            d_no_tgt = data.get("drop_no_target", 0)
                            
                            # 相容舊版欄位 (Optional)
                            if "drop_slo" in data: d_local += data["drop_slo"]
                            if "drop_queue" in data and d_no_tgt == 0: d_no_tgt += data["drop_queue"]

                            global_snapshot["efo_totals"]["total_inference_time"] += data.get("total_effective_inference_time", 0.0)
                            
                            global_snapshot["efo_totals"]["total_drop_local_congestion"] += d_local
                            global_snapshot["efo_totals"]["total_drop_no_target"] += d_no_tgt
                            global_snapshot["efo_totals"]["total_drops"] += (d_local + d_no_tgt)

                            global_snapshot["efo_totals"]["total_offloads"] += data.get("offload_out", 0)
                            global_snapshot["efo_totals"]["total_local_completed"] += data.get("local_completed", 0)
                            global_snapshot["efo_totals"]["total_offload_completed"] += data.get("offload_in_completed", 0)
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch metrics from {cluster_name}: {e}")
            
            # 加入 EFO 自身的累積指標
            async with efo_metrics.lock:
                global_snapshot["efo_totals"]["artifact_downloads"] = efo_metrics.artifact_downloads
                global_snapshot["efo_totals"]["total_stored_loras"] = efo_metrics.cumulative_stored_loras
            
            with open(log_file, "a") as f:
                f.write(json.dumps(global_snapshot) + "\n")
                
        logger.info(f"📊 [EFO Metrics] Step {step_id} finished (10/10). Waiting for next Time Edge...")
        
    except asyncio.CancelledError:
        logger.info(f"📊 [EFO Metrics] Step {step_id} cancelled (New Time Edge arrived).")
        raise

# ============================================================
# SP1: CSV Forecasting (Interval Scanning)
# ============================================================
def exact_csv_forecasting(time_step: int):
    global predicted_demand
    predicted_demand.clear()
    
    START_OFFSET = 86400 * 2
    start_sec = time_step * SP1_INTERVAL_SECONDS
    end_sec = (time_step + 1) * SP1_INTERVAL_SECONDS
    
    for cluster_name in active_clusters.keys():
        predicted_demand[cluster_name] = {lora_id: 0.0 for lora_id in global_lora_metadata.keys()}

    if not os.path.exists(SIMULATION_DATA_CSV_PATH):
        logger.error(f"❌ 找不到 CSV 檔案: {SIMULATION_DATA_CSV_PATH}")
        return

    try:
        df = pd.read_csv(SIMULATION_DATA_CSV_PATH)
        df["arrival_sec"] = df["arrive_timestamp"].astype(float)
        
        if df["arrival_sec"].min() >= START_OFFSET:
            df["arrival_sec"] -= START_OFFSET
        
        df = df[(df["arrival_sec"] >= start_sec) & (df["arrival_sec"] < end_sec)]
        
        target_clusters = list(active_clusters.keys())
        df = df[df["cluster"].isin(target_clusters)]

        for _, row in df.iterrows():
            cluster = str(row["cluster"]).strip()
            lora_id_val = int(float(row["lora_id"]))
            lora_id = f"LoRA_{lora_id_val}"
            
            if lora_id in predicted_demand[cluster]:
                predicted_demand[cluster][lora_id] += 1.0
            elif str(lora_id_val) in predicted_demand[cluster]:
                predicted_demand[cluster][str(lora_id_val)] += 1.0

        for cluster in target_clusters:
            count = sum(predicted_demand[cluster].values())
            logger.info(f"📈 [Pandas Forecast] {cluster} 步進 {time_step}: 預計有 {int(count)} 個請求")

    except Exception as e:
        logger.error(f"❌ Pandas 處理 CSV 發生錯誤: {e}")

# ============================================================
# SP1: Provisioning Algorithm
# ============================================================
async def run_sp1_provisioning_and_wait():
    if not global_lora_metadata or not active_clusters: return
    logger.info("⚙️ Running SP1 Adaptive CSG-Swap Provisioning (With Semantic Similarity)...")

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

    # [新增] 輔助函式：檢查 stored_set 中是否已包含 lora_id 的替代品
    def has_substitute(l_id, stored_set):
        meta = global_lora_metadata.get(l_id, {})
        subs = meta.get("substitutes", [])
        for s in subs:
            if s in stored_set:
                return True
        return False

    # 1. 計算所有活躍 Cluster 的最優 LoRA 配置
    for cluster_name in active_clusters.keys():
        target_disk = set()
        mandatory_set = set()
        utilities = {}

        valid_loras = []
        for lora_id, info in global_lora_metadata.items():
            if info.get("type") == "global" or (info.get("type") == "local" and info.get("cluster") == cluster_name):
                valid_loras.append(lora_id)

        # 計算原始 Gains (假設無本地替代品的情況)
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

        # Step 0: Mandatory Sets (Local)
        # 這些模型是不可協商的，必須存在
        for lora_id in valid_loras:
            if global_lora_metadata[lora_id].get("type") == "local":
                mandatory_set.add(lora_id)
                target_disk.add(lora_id)

        # Step 1: Evaluation and Eviction (Retention)
        # 評估是否保留現有的模型
        current_disk = set(global_lora_disk_inventory.get(cluster_name, []))
        for lora_id in current_disk:
            if lora_id in mandatory_set or lora_id not in valid_loras:
                continue
            
            # [修改] 語意相似度檢查
            # 如果 target_disk 中已經有替代品 (例如 Local Mandatory 或前面已加入的高分模型)，
            # 則視為該需求已被滿足 (Gain 歸零)，將其驅逐以節省空間。
            if has_substitute(lora_id, target_disk):
                continue 

            lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
            u_retention = (lambd * gains[lora_id]) - (S_LORA * C_STORE)
            if u_retention >= 0:
                target_disk.add(lora_id)
                utilities[lora_id] = u_retention

        # Step 2: Iterative Expansion with Swap
        # 評估是否下載新模型
        candidates = []
        for lora_id in valid_loras:
            if lora_id not in target_disk and lora_id not in mandatory_set:
                lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
                u_download = (lambd * gains[lora_id]) - (S_LORA * (C_STORE + C_DL))
                if u_download > 0:
                    candidates.append((lora_id, u_download))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for lora_id, u_v in candidates:
            # [修改] 下載前的語意相似度檢查
            # 若已有替代品，則不下載
            if has_substitute(lora_id, target_disk):
                continue

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

        # 計算實際產生的新下載 (排除 Local 類型)
        diff_set = target_disk - current_disk
        real_new_downloads = 0
        for lora_id in diff_set:
            if global_lora_metadata.get(lora_id, {}).get("type") != "local":
                real_new_downloads += 1
                
        if real_new_downloads > 0:
            async with efo_metrics.lock:
                efo_metrics.artifact_downloads += real_new_downloads

        cluster_targets[cluster_name] = list(target_disk)
        global_lora_disk_inventory[cluster_name] = cluster_targets[cluster_name]
        
        logger.info(f"📊 [SP1 Result] {cluster_name}: Target {len(target_disk)}/{CAPACITY} LoRAs (WAN DLs: {real_new_downloads})")

    # 更新累計存儲量
    current_total_stored = sum(len(loras) for loras in cluster_targets.values())
    async with efo_metrics.lock:
        efo_metrics.cumulative_stored_loras += current_total_stored
    logger.info(f"📦 [SP1 Storage] Added {current_total_stored} to cumulative storage count.")

    # 2. 發送並「阻塞等待」所有 Control Node 排空與重置
    logger.info("⏳ Dispatching SP1 to Control Nodes and WAITING for system drain & reset...")
    async with httpx.AsyncClient(timeout=EDGE_SYNC_TIMEOUT) as client:
        tasks = []
        for cluster_name, target_loras in cluster_targets.items():
            url = active_clusters[cluster_name]
            payload = {"loras": target_loras}
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
# Background Tasks & Lifecycle
# ============================================================
async def sp2_routing_loop():
    await system_start_event.wait()
    await sync_global_routing()
    while True:
        await asyncio.sleep(SP2_INTERVAL_SECONDS)
        await sync_global_routing()

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
    yield
        
app = FastAPI(title="Edge Federation Orchestrator (EFO)", lifespan=lifespan)

@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered: {req.cluster_name}")
    return {"status": "ok", "metadata": global_lora_metadata}

@app.post("/time_edge")
async def trigger_time_edge():
    global current_time_step, efo_logging_task
    
    if not system_start_event.is_set():
        system_start_event.set()
        logger.info("🚀 System initialized via first /time_edge trigger!")

    logger.info(f"\n{'='*50}\n⏱️ [TIME EDGE] Advancing to Time Step {current_time_step}\n{'='*50}")
    
    # === 1. 重置並啟動 EFO Metrics Logging ===
    if efo_logging_task and not efo_logging_task.done():
        efo_logging_task.cancel()
        try:
            await efo_logging_task
        except asyncio.CancelledError: pass
            
    efo_logging_task = asyncio.create_task(run_efo_metrics_cycle(current_time_step))
    
    exact_csv_forecasting(current_time_step)
    await run_sp1_provisioning_and_wait()
    
    completed_step = current_time_step
    current_time_step += 1
    
    return {
        "status": "success", 
        "completed_step": completed_step,
        "next_step": current_time_step
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9100)))
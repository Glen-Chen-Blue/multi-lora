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
    LORA_PATH, LORA_METADATA_PATH, LOG_PATH,
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
    
    # 確保這裡的 OFFSET 與 test_simulation.py 一致
    START_OFFSET = 86400 * 2 
    
    # 計算當前 Time Step 對應的 "歸零後" 時間範圍
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
        
        # === [修正] 強制對齊邏輯 ===
        # 1. 先過濾掉 START_OFFSET 之前的舊資料 (模擬器不跑這些，EFO 也不該看這些)
        df = df[df["arrival_sec"] >= START_OFFSET].copy()
        
        # 2. 強制平移時間軸，將 START_OFFSET 視為 0
        df["arrival_sec"] -= START_OFFSET
        # ==========================
        
        # 3. 根據 Step 選取對應區間的資料
        df = df[(df["arrival_sec"] >= start_sec) & (df["arrival_sec"] < end_sec)]
        
        target_clusters = list(active_clusters.keys())
        df = df[df["cluster"].isin(target_clusters)]

        for _, row in df.iterrows():
            cluster = str(row["cluster"]).strip()
            # 相容整數與字串格式的 LoRA ID
            try:
                lora_id_val = int(float(row["lora_id"]))
                lora_id = f"LoRA_{lora_id_val}"
            except:
                lora_id = str(row["lora_id"])
            
            # 統計需求 (相容 LoRA_X 格式與純數字格式)
            if lora_id in predicted_demand[cluster]:
                predicted_demand[cluster][lora_id] += 1.0
            elif str(lora_id_val) in predicted_demand[cluster]: # fallback
                predicted_demand[cluster][str(lora_id_val)] += 1.0

        for cluster in target_clusters:
            count = sum(predicted_demand[cluster].values())
            logger.info(f"📈 [Pandas Forecast] {cluster} (Step {time_step}, Time {start_sec}-{end_sec}s): 預計有 {int(count)} 個請求")

    except Exception as e:
        logger.error(f"❌ Pandas 處理 CSV 發生錯誤: {e}")


# ============================================================
# SP1: Provisioning Algorithm
# ============================================================
async def run_sp1_provisioning_and_wait():
    if not global_lora_metadata or not active_clusters: return
    logger.info("⚙️ Running SP1 Adaptive CSG-Swap (Enhanced Semantic Awareness)...")

    p95_delays = network_simulator.get_p95_info()
    
    # === 參數設定 ===
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

    # =========================================================================
    # 0. 建立反向替代關係 (Reverse Mapping)
    # serves_map[A] = {A, B, C} 表示若下載 A，可以服務 A, B, C 的請求
    # =========================================================================
    serves_map = defaultdict(set)
    valid_loras = [] # 僅考慮與當前 Cluster 相關或 Global 的

    # 初始化：每個 LoRA 至少服務自己
    for lid in global_lora_metadata.keys():
        serves_map[lid].add(lid)
    
    # 填充替代關係：若 B 的 substitute 是 A (meta[B]['substitutes']包含A)，則 A 服務 B
    for lid, info in global_lora_metadata.items():
        subs = info.get("substitutes", [])
        for parent in subs:
            # parent (A) 可以服務 lid (B)
            serves_map[parent].add(lid)

    # 輔助函式：檢查 l_id 是否已被 stored_set 中的某個模型覆蓋
    def is_covered_by_set(target_id, stored_set):
        # 1. 直接命中
        if target_id in stored_set: return True
        # 2. 語意命中 (stored_set 中有 target_id 的替代品)
        subs = global_lora_metadata.get(target_id, {}).get("substitutes", [])
        for s in subs:
            if s in stored_set: return True
        return False

    # 輔助函式：計算加入 candidate_id 能帶來的「新增」需求滿足量
    # 邏輯：遍歷 candidate_id 能服務的所有 id，若該 id 目前沒被 stored_set 覆蓋，則計入需求
    def calculate_marginal_demand(cluster, candidate_id, current_set):
        total_new_demand = 0.0
        # 找出 candidate 能服務的所有對象
        targets = serves_map.get(candidate_id, set())
        
        for tid in targets:
            # 只有當 tid 目前「還沒被滿足」時，才算作 candidate 的貢獻
            if not is_covered_by_set(tid, current_set):
                d = predicted_demand[cluster].get(tid, 0.0)
                total_new_demand += d
        return total_new_demand

    # =========================================================================
    # Phase 1: 樂觀的個別 Cluster 配置 (Optimistic Local Provisioning)
    # (保持原邏輯，加入 serves_map 的概念優化效益計算)
    # =========================================================================
    for cluster_name in active_clusters.keys():
        target_disk = set()
        mandatory_set = set()
        utilities = {} # 用於記錄每個已存模型的效益

        # 篩選有效 LoRA
        cluster_valid_loras = []
        for lora_id, info in global_lora_metadata.items():
            if info.get("type") == "global" or (info.get("type") == "local" and info.get("cluster") == cluster_name):
                cluster_valid_loras.append(lora_id)

        # Step 0: Mandatory Sets
        for lora_id in cluster_valid_loras:
            if global_lora_metadata[lora_id].get("type") == "local":
                mandatory_set.add(lora_id)
                target_disk.add(lora_id)

        # Step 1: Evaluation and Eviction (Retention)
        # 對於已存在的，我們計算其「保留價值」
        # 保留價值 = (它能獨自覆蓋的需求 * Gain) - 存儲成本
        current_disk = set(global_lora_disk_inventory.get(cluster_name, []))
        
        # 為了計算方便，Phase 1 簡化處理：
        # 如果已被替代品覆蓋，則不保留；否則計算其自身的直接需求效益
        for lora_id in current_disk:
            if lora_id in mandatory_set or lora_id not in cluster_valid_loras:
                continue
            
            # 檢查是否被「其他已存在硬碟的」模型覆蓋 (不含自己)
            temp_set = target_disk.union(current_disk) - {lora_id}
            if is_covered_by_set(lora_id, temp_set):
                continue # 已經有人能取代我了，且我不是 Mandatory

            # 計算效益 (這裡簡化，只看直接需求，完整邊際在 Expansion 算)
            # Gain 估算
            best_offload_cost = C_DROP # 預設 Drop
            offload_costs = []
            for k in active_clusters.keys():
                if k == cluster_name: continue
                delay_sec = p95_delays.get(cluster_name, {}).get(k, 1000.0) / 1000.0
                gamma = T_MAX / (T_MAX - delay_sec) if delay_sec < T_MAX else float('inf')
                offload_costs.append(gamma * C_INST + C_NET)
            if offload_costs: best_offload_cost = min(min(offload_costs), C_DROP)
            gain_per_req = max(0.0, best_offload_cost - C_INST)

            lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
            u_retention = (lambd * gain_per_req) - (S_LORA * C_STORE)
            
            if u_retention >= 0:
                target_disk.add(lora_id)
                utilities[lora_id] = u_retention

        # Step 2: Iterative Expansion with Swap
        # 找出所有候選者 (不在 target_disk 中)
        candidates = [l for l in cluster_valid_loras if l not in target_disk]
        
        # 由於計算邊際效益會隨 target_disk 改變，這裡採用 Iterative Greedy
        # 每次選一個邊際效益最高的加入
        while True:
            best_cand = None
            max_u = -float('inf')

            # 掃描所有候選者，計算當下的邊際效益
            for cand in candidates:
                # 計算：若加入 cand，能多處理多少需求 (包含它能取代的)
                new_demand = calculate_marginal_demand(cluster_name, cand, target_disk)
                
                # Gain 概算 (統一用 C_DROP 當作挽回的效益，簡化計算)
                # 嚴謹來說應該看被 Rescue 的請求原本是 Drop 還是 Offload，這裡假設能放在本地通常是為了救 Drop 或省 Offload
                benefit = new_demand * C_DROP 
                cost = S_LORA * (C_STORE + C_DL) # 新下載成本
                net_u = benefit - cost

                if net_u > max_u:
                    max_u = net_u
                    best_cand = cand
            
            if best_cand is None or max_u <= 0:
                break # 沒有正效益的候選者了

            # 嘗試加入
            if len(target_disk) < CAPACITY:
                target_disk.add(best_cand)
                utilities[best_cand] = max_u # 記錄當下效益
                candidates.remove(best_cand)
            else:
                # 空間滿，嘗試 Swap
                # 找出目前 target_disk 中「邊際效益最低」的非強制項目 (Victim)
                # 重新評估 Victim 的「當前移除損失」
                victim = None
                min_loss = float('inf')
                
                swappable = [t for t in target_disk if t not in mandatory_set]
                if not swappable: break

                for t in swappable:
                    # 如果移除 t，會損失多少需求 (即 t 目前貢獻的邊際需求)
                    # 模擬移除 t 後的集合
                    temp_set = target_disk - {t}
                    loss_demand = calculate_marginal_demand(cluster_name, t, temp_set)
                    loss_val = (loss_demand * C_DROP) - (S_LORA * C_STORE) # 移除它省下存儲費，但損失 Benefit
                    if loss_val < min_loss:
                        min_loss = loss_val
                        victim = t
                
                # 比較：新候選者效益 vs 犧牲者損失 + 門檻
                if max_u > min_loss + EPSILON:
                    target_disk.remove(victim)
                    target_disk.add(best_cand)
                    utilities[best_cand] = max_u
                    candidates.remove(best_cand)
                    candidates.append(victim) # 犧牲者變回候選人
                    logger.info(f"    🔄 [SP1 Local] Swapped {victim} (Loss {min_loss:.2f}) with {best_cand} (Gain {max_u:.2f}) in {cluster_name}")
                else:
                    break # 連效益最高的都換不進去，結束

        # 統計新下載
        current_disk_static = set(global_lora_disk_inventory.get(cluster_name, []))
        real_new_downloads = len(target_disk - current_disk_static)
        if real_new_downloads > 0:
            async with efo_metrics.lock: efo_metrics.artifact_downloads += real_new_downloads

        cluster_targets[cluster_name] = list(target_disk)
        logger.info(f"📊 [SP1 Local] {cluster_name}: Target {len(target_disk)}/{CAPACITY} LoRAs")


    # =========================================================================
    # Phase 2: 全域語意感知救援 (Global Semantic-Aware Rescue)
    # 邏輯：檢查所有 Global LoRA，看是否能透過「跨區配置」來解決尚未滿足的需求
    # =========================================================================
    
    # 建立所有 Global LoRA 的候選名單
    global_candidates = [l for l, info in global_lora_metadata.items() if info.get("type") == "global"]
    
    # 為了避免過度計算，我們採用一輪掃描：
    # 對每個 Global LoRA，找出它在「哪個 Cluster」能產生最大的「邊際效益」
    # 如果該效益 > 成本，且能塞入(或置換)，就執行。
    
    # 隨機打亂順序以避免偏見，或可依照全域總需求排序
    global_candidates.sort(key=lambda l: sum([predicted_demand[c].get(l,0) for c in active_clusters]), reverse=True)

    for cand_id in global_candidates:
        best_cluster = None
        best_net_utility = -float('inf')
        
        # 1. 評估每個 Cluster 放入此模型的潛力
        for c_name in active_clusters.keys():
            current_set = set(cluster_targets[c_name])
            
            # 若已經有了(包含被替代品覆蓋)，則邊際效益為 0
            if is_covered_by_set(cand_id, current_set):
                continue
            
            # 計算加入後的邊際需求滿足量 (New Requests)
            # 這會考慮 cand_id 本身的需求 + 它能取代的其他未滿足需求
            marginal_demand = calculate_marginal_demand(c_name, cand_id, current_set)
            
            if marginal_demand <= 0: continue
            
            # 效益計算
            benefit = marginal_demand * C_DROP # 救回這些請求
            cost = S_LORA * (C_STORE + C_DL)   # 下載成本
            net_u = benefit - cost
            
            if net_u > 0 and net_u > best_net_utility:
                best_net_utility = net_u
                best_cluster = c_name
        
        # 2. 嘗試部署到最佳 Cluster
        if best_cluster:
            target_set = set(cluster_targets[best_cluster])
            deployed = False
            
            # Case A: 有空間
            if len(target_set) < CAPACITY:
                target_set.add(cand_id)
                cluster_targets[best_cluster] = list(target_set)
                async with efo_metrics.lock: efo_metrics.artifact_downloads += 1
                logger.info(f"🚨 [Global Rescue] Added {cand_id} to {best_cluster} (New Reqs: {best_net_utility/C_DROP:.1f}, NetU: {best_net_utility:.2f})")
                deployed = True
            
            # Case B: 沒空間，嘗試 Swap
            else:
                # 尋找犧牲者：移除後損失最小的
                victim = None
                min_victim_loss = float('inf')
                
                swappable = [t for t in target_set if global_lora_metadata.get(t,{}).get("type") != "local"]
                for t in swappable:
                    # 計算移除 t 的損失 (即 t 貢獻的邊際需求)
                    temp_set = target_set - {t}
                    loss_demand = calculate_marginal_demand(best_cluster, t, temp_set)
                    loss_val = (loss_demand * C_DROP) - (S_LORA * C_STORE)
                    
                    if loss_val < min_victim_loss:
                        min_victim_loss = loss_val
                        victim = t
                
                # 只有當 救援效益 明顯大於 犧牲損失 時才做
                if best_net_utility > min_victim_loss + EPSILON:
                    target_set.remove(victim)
                    target_set.add(cand_id)
                    cluster_targets[best_cluster] = list(target_set)
                    async with efo_metrics.lock: efo_metrics.artifact_downloads += 1
                    logger.info(f"🚨 [Global Rescue] Swapped {victim} (Loss {min_victim_loss:.2f}) with {cand_id} (Gain {best_net_utility:.2f}) in {best_cluster}")
                    deployed = True

            if not deployed:
                # 記錄一下，雖有正效益但擠不進去
                pass

    # 更新累計存儲量
    current_total_stored = sum(len(loras) for loras in cluster_targets.values())
    async with efo_metrics.lock:
        efo_metrics.cumulative_stored_loras += current_total_stored
    logger.info(f"📦 [SP1 Storage] Final Count: {current_total_stored} LoRAs stored across clusters.")

    # 3. 發送並「阻塞等待」所有 Control Node 排空與重置
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
            logger.info(f"📂 Loaded LoRA metadata for {(global_lora_metadata)} LoRAs.")
            
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
    
    # 執行 SP1 配置並等待所有節點 Reset 完成
    await run_sp1_provisioning_and_wait()
    
    # === [新增] 強制同步全域路由表，避免時間差 ===
    # 這確保了在我們告訴模擬器 "Time Edge 完成" 之前，
    # 所有 Control Node 都已經拿到最新的路由表 (知道鄰居有什麼 LoRA)，
    # 這樣一開始的 Request 才不會因為路由表空白而噴 "No Target"。
    logger.info("🔄 [Time Edge] Forcing global routing sync before releasing...")
    await sync_global_routing()
    # ==========================================
    
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
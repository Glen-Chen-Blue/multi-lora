import os
import json
import logging
import asyncio
import httpx
import random
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from collections import defaultdict

import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel

# 匯入集中管理的設定
from config import (
    LORA_PATH, LORA_METADATA_PATH,
    LORA_MAPPING_PATH, LORA_HOURLY_COUNTS_PATH,
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB, COST_INST_LOCAL,
    COST_NET_TRAFFIC, COST_DROP_PENALTY, LORA_SIZE_GB,
    DISK_CAPACITY_GB, T_MAX_SLO, SWAP_EPSILON,
    T_TOTAL_HOURS, SEQ_LENGTH, NETWORK_SIM_PARAMS
)

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EFO: %(message)s")
logger = logging.getLogger("EFOServer")
logging.getLogger("httpx").setLevel(logging.WARNING)
CLUSTERS_ENV = os.environ.get("CLUSTERS", "{}")

# ============================================================
# Global State & System Variables
# ============================================================
global_lora_metadata: Dict[str, Any] = {}
configured_clusters: Dict[str, str] = {}  # 預期名單
active_clusters: Dict[str, str] = {}      # 活躍名單

global_lora_disk_inventory: Dict[str, List[str]] = {}
historical_demand: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
predicted_demand: Dict[str, Dict[str, float]] = defaultdict(dict)
azure_mapping: Dict[str, Dict[str, str]] = {}

current_time_step = 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = None
system_start_event: asyncio.Event = None
TRAINING_FUNC_NAMES = sorted([str(i) for i in range(1, 241)])

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
# LSTM Model Definition
# ============================================================
class OverfitLSTM(nn.Module):
    def __init__(self, num_funcs, num_hours, func_emb_dim, hour_emb_dim, hidden, num_layers):
        super().__init__()
        self.func_emb = nn.Embedding(num_funcs, func_emb_dim)
        self.hour_emb = nn.Embedding(num_hours, hour_emb_dim)
        self.lstm = nn.LSTM(
            input_size=1 + hour_emb_dim + func_emb_dim, 
            hidden_size=hidden, 
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.out_act = nn.Softplus()

    def forward(self, x_seq, hour_seq, func_id):
        femb = self.func_emb(func_id)                
        femb_seq = femb.unsqueeze(1).repeat(1, x_seq.shape[1], 1) 
        hemb = self.hour_emb(hour_seq)               
        inp = torch.cat([x_seq, hemb, femb_seq], dim=-1)  
        out, _ = self.lstm(inp)
        y = self.fc(out[:, -1, :])                               
        return self.out_act(y)                          

# ============================================================
# SP2: Global Routing Broadcast (10s interval)
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
# SP1: Forecasting & Provisioning (1-hour interval)
# ============================================================
async def fetch_cluster_stats():
    if not active_clusters or not global_lora_metadata: return
    async with httpx.AsyncClient(timeout=5.0) as client:
        for cluster_name, url in active_clusters.items():
            try:
                resp = await client.get(f"{url}/pop_lora_stats")
                if resp.status_code == 200:
                    stats = resp.json().get("stats", {})
                    for lora_id in global_lora_metadata.keys():
                        historical_demand[cluster_name][lora_id].append(stats.get(lora_id, 0))
            except Exception as e:
                pass

def hybrid_forecasting_engine():
    global current_time_step, predicted_demand
    if lstm_model is None or not azure_mapping: return
    
    logger.info(f"🧠 LSTM Forecasting Start... (Simulated Hour: {current_time_step})")
    lstm_model.eval()
    
    with torch.no_grad():
        for cluster_name, loras in azure_mapping.items():
            for lora_id, azure_id_str in loras.items():
                history = historical_demand[cluster_name].get(lora_id, [0]*SEQ_LENGTH)
                recent_48 = history[-SEQ_LENGTH:]
                if len(recent_48) < SEQ_LENGTH:
                    recent_48 = [0]*(SEQ_LENGTH - len(recent_48)) + recent_48
                
                azure_idx = TRAINING_FUNC_NAMES.index(azure_id_str)
                x_seq = torch.tensor(np.log1p(recent_48), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                
                start_hour = (current_time_step - SEQ_LENGTH) % T_TOTAL_HOURS
                h_seq = [(start_hour + i) % T_TOTAL_HOURS for i in range(SEQ_LENGTH)]
                hour_seq = torch.tensor([h_seq], dtype=torch.long).to(device)
                func_id_tensor = torch.tensor([azure_idx], dtype=torch.long).to(device)
                
                pred_log = lstm_model(x_seq, hour_seq, func_id_tensor)
                predicted_demand[cluster_name][lora_id] = np.expm1(pred_log.item()) 

    current_time_step += 1

async def run_sp1_provisioning():
    """SP1: Adaptive CSG-Swap Algorithm (Cost-Aware Predictive LoRA Placement)"""
    if not global_lora_metadata or not active_clusters: return
    logger.info("⚙️ Running SP1 Adaptive CSG-Swap Provisioning...")

    p95_delays = network_simulator.get_p95_info()
    
    # 提取常數
    C_STORE = COST_STORE_PER_GB
    C_DL = COST_DOWNLOAD_PER_GB
    C_INST = COST_INST_LOCAL
    C_NET = COST_NET_TRAFFIC
    C_DROP = COST_DROP_PENALTY
    S_LORA = LORA_SIZE_GB
    CAPACITY = int(DISK_CAPACITY_GB / S_LORA)
    T_MAX = T_MAX_SLO
    EPSILON = SWAP_EPSILON

    async with httpx.AsyncClient(timeout=10.0) as client:
        for cluster_name, url in active_clusters.items():
            target_disk = set()
            mandatory_set = set()
            utilities = {}

            # ==========================================
            # 前置準備：計算每個 LoRA 的 Marginal Gain
            # ==========================================
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

            # ==========================================
            # Step 0: Mandatory Sets (強制存放 Local LoRA)
            # ==========================================
            for lora_id in valid_loras:
                if global_lora_metadata[lora_id].get("type") == "local":
                    mandatory_set.add(lora_id)
                    target_disk.add(lora_id)

            if len(target_disk) > CAPACITY:
                logger.error(f"❌ {cluster_name}: Disk capacity too small even for mandatory LoRAs!")
                continue

            # ==========================================
            # Step 1: Evaluation and Eviction (評估既有項目)
            # ==========================================
            current_disk = set(global_lora_disk_inventory.get(cluster_name, []))
            
            for lora_id in current_disk:
                if lora_id in mandatory_set or lora_id not in valid_loras:
                    continue
                
                lambd = predicted_demand[cluster_name].get(lora_id, 0.0)
                u_retention = (lambd * gains[lora_id]) - (S_LORA * C_STORE)
                
                if u_retention >= 0:
                    target_disk.add(lora_id)
                    utilities[lora_id] = u_retention

            # ==========================================
            # Step 2: Iterative Expansion with Swap (貪婪替換)
            # ==========================================
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

            # ==========================================
            # 印出 SP1 決策結果與詳細資訊
            # ==========================================
            target_loras = list(target_disk)
            
            local_count = sum(1 for l in target_loras if l in mandatory_set)
            global_count = len(target_loras) - local_count
            
            # 抓出 Global 模型中效用最高的幾個來印
            top_globals = sorted([(l, utilities.get(l, 0)) for l in target_loras if l not in mandatory_set], key=lambda x: x[1], reverse=True)
            top_globals_str = ", ".join([f"{l}(U:{u:.2f})" for l, u in top_globals[:5]])
            if not top_globals_str:
                top_globals_str = "None (No Global LoRAs worth storing)"
            
            logger.info(f"📊 [SP1 Result] {cluster_name}: Total {len(target_loras)}/{CAPACITY} LoRAs")
            logger.info(f"   ┣ Mandatory (Local) : {local_count} models")
            logger.info(f"   ┗ Dynamic (Global)  : {global_count} models (Top: {top_globals_str})")

            # ==========================================
            # 發送最新的配置策略至 Control Node
            # ==========================================
            global_lora_disk_inventory[cluster_name] = target_loras
            try:
                await client.post(f"{url}/update_local_loras", json={"loras": target_loras})
            except Exception as e:
                logger.error(f"❌ Failed to provision LoRAs to {cluster_name}: {e}")

# ============================================================
# Background Tasks (Event Loops)
# ============================================================
async def sp2_routing_loop():
    await system_start_event.wait()
    await sync_global_routing()
    while True:
        await asyncio.sleep(3)
        await sync_global_routing()

async def sp1_provisioning_loop():
    await system_start_event.wait()
    logger.info("🚀 System Started! Running initial SP1 Pipeline...")
    hybrid_forecasting_engine()
    await run_sp1_provisioning()
    while True:
        await asyncio.sleep(3600) 
        await fetch_cluster_stats()
        hybrid_forecasting_engine()
        await run_sp1_provisioning()

# ============================================================
# Lifecycle & API Endpoints
# ============================================================
class RegisterClusterRequest(BaseModel):
    cluster_name: str
    control_node_url: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_lora_metadata, configured_clusters, azure_mapping, lstm_model, system_start_event
    system_start_event = asyncio.Event()

    try:
        configured_clusters = json.loads(CLUSTERS_ENV)
    except: pass
    
    if os.path.exists(LORA_METADATA_PATH):
        with open(LORA_METADATA_PATH, "r", encoding="utf-8") as f:
            global_lora_metadata = json.load(f)
            
    if os.path.exists(LORA_MAPPING_PATH) and os.path.exists(LORA_HOURLY_COUNTS_PATH):
        with open(LORA_MAPPING_PATH, "r") as f: azure_mapping = json.load(f)
        with open(LORA_HOURLY_COUNTS_PATH, "r") as f: hourly_counts = json.load(f)
        for cluster_name, cluster_mapping in azure_mapping.items():
            for lora_id, azure_id in cluster_mapping.items():
                historical_demand[cluster_name][lora_id] = hourly_counts[azure_id][:SEQ_LENGTH]
                
    try:
        lstm_model = OverfitLSTM(240, T_TOTAL_HOURS, 32, 32, 128, 2).to(device)
        lstm_model.load_state_dict(torch.load("./data/azure_lstm_32x_overfit.pth", map_location=device))
        lstm_model.eval()
    except Exception as e: logger.error(f"❌ Failed to load LSTM model: {e}")

    asyncio.create_task(sp2_routing_loop())
    asyncio.create_task(sp1_provisioning_loop())
    yield
        
app = FastAPI(title="Edge Federation Orchestrator (EFO)", lifespan=lifespan)

@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered: {req.cluster_name}")
    return {"status": "ok", "metadata": global_lora_metadata}

@app.post("/start")
async def start_system():
    if system_start_event.is_set(): return {"status": "already running"}
    system_start_event.set()
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9100)))
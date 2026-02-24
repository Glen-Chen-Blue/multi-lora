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

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EFO: %(message)s")
logger = logging.getLogger("EFOServer")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
LORA_METADATA_PATH = os.environ.get("LORA_METADATA", "./lora_metadata.json")
CLUSTERS_ENV = os.environ.get("CLUSTERS", "{}")

# ============================================================
# Global State & LSTM Model Variables
# ============================================================
global_lora_metadata: Dict[str, Any] = {}
configured_clusters: Dict[str, str] = {}  # 從環境變數讀取的預期名單
active_clusters: Dict[str, str] = {}      # 實際已經來註冊的活躍名單

# 記錄哪些 LoRA 目前存在於哪個 Cluster 的 Disk 上
global_lora_disk_inventory: Dict[str, List[str]] = {}

# 紀錄各個 Cluster 中各個 LoRA 的歷史需求量
historical_demand: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
predicted_demand: Dict[str, Dict[str, float]] = defaultdict(dict)
azure_mapping: Dict[str, Dict[str, str]] = {}

current_time_step = 48
T_TOTAL_HOURS = 336 
SEQ_LENGTH = 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = None

# 系統啟動開關
system_start_event: asyncio.Event = None

# 🌟 [修正] 重現 Pandas pivot 的字典排序列表 (1~240)，確保與訓練時的 Index 100% 吻合
TRAINING_FUNC_NAMES = sorted([str(i) for i in range(1, 241)])

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
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.out_act = nn.Softplus()

    def forward(self, x_seq, hour_seq, func_id):
        B, L, _ = x_seq.shape
        femb = self.func_emb(func_id)                
        femb_seq = femb.unsqueeze(1).repeat(1, L, 1) 
        hemb = self.hour_emb(hour_seq)               

        inp = torch.cat([x_seq, hemb, femb_seq], dim=-1)  
        out, _ = self.lstm(inp)
        h = out[:, -1, :]                            
        y = self.fc(h)                               
        y = self.out_act(y)                          
        return y

# ============================================================
# SP2: Global Routing & Offloading (10s interval)
# ============================================================
def generate_delays(cluster_names: List[str]) -> Dict[str, Dict[str, int]]:
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
    if not active_clusters:
        return

    cluster_names = list(active_clusters.keys())
    delays = generate_delays(cluster_names)
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
                        "lora_status": data.get("lora_status", {
                            "merged": [], "loaded": [], "unloaded": []
                        }),
                        "delay": delays[cluster_name]
                    }
            except Exception as e:
                logger.error(f"❌ [Routing] Error getting status from {cluster_name}: {e}")

        if not routing_table:
            return

        for cluster_name, url in active_clusters.items():
            try:
                await client.post(f"{url}/update_global_routing", json={"routing_table": routing_table})
                logger.info(f"🌐 [Routing] Broadcasted routing table to {cluster_name}")
            except Exception as e:
                logger.error(f"❌ [Routing] Error broadcasting to {cluster_name}: {e}")

# ============================================================
# SP1: Forecasting & Provisioning (60s interval)
# ============================================================
async def fetch_cluster_stats():
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
            except Exception as e:
                logger.error(f"❌ Error fetching stats from {cluster_name}: {e}")

def hybrid_forecasting_engine():
    global current_time_step, predicted_demand
    
    if lstm_model is None or not azure_mapping:
        logger.warning("⚠️ LSTM 模型或 Mapping 未載入，略過預測。")
        return
        
    logger.info(f"🧠 LSTM Forecasting Start... (Simulated Hour: {current_time_step})")
    lstm_model.eval()
    
    with torch.no_grad():
        for cluster_name, loras in azure_mapping.items():
            for lora_id, azure_id_str in loras.items():
                history = historical_demand[cluster_name].get(lora_id, [0]*SEQ_LENGTH)
                recent_48 = history[-SEQ_LENGTH:]
                if len(recent_48) < SEQ_LENGTH:
                    pad_len = SEQ_LENGTH - len(recent_48)
                    recent_48 = [0]*pad_len + recent_48
                
                # 🌟 [修正] 嚴格依照字典排序陣列獲取 Index
                azure_idx = TRAINING_FUNC_NAMES.index(azure_id_str)
                
                x_seq = torch.tensor(np.log1p(recent_48), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                
                start_hour = (current_time_step - SEQ_LENGTH) % T_TOTAL_HOURS
                h_seq = [(start_hour + i) % T_TOTAL_HOURS for i in range(SEQ_LENGTH)]
                hour_seq = torch.tensor([h_seq], dtype=torch.long).to(device)
                
                func_id_tensor = torch.tensor([azure_idx], dtype=torch.long).to(device)
                
                pred_log = lstm_model(x_seq, hour_seq, func_id_tensor)
                pred_val = np.expm1(pred_log.item()) 
                
                predicted_demand[cluster_name][lora_id] = pred_val

    current_time_step += 1
    
    for cluster_name in active_clusters.keys():
        sorted_preds = sorted(predicted_demand[cluster_name].items(), key=lambda x: x[1], reverse=True)
        top_3 = [(l_id, round(val, 2)) for l_id, val in sorted_preds[:3]]
        logger.info(f"📈 [Prediction] {cluster_name} Top 3: {top_3}")

async def run_sp1_provisioning():
    if not global_lora_metadata or not active_clusters:
        return

    all_loras = list(global_lora_metadata.keys())
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for cluster_name, url in active_clusters.items():
            global_lora_disk_inventory[cluster_name] = all_loras.copy()
            try:
                await client.post(f"{url}/update_local_loras", json={"loras": all_loras})
            except Exception as e:
                logger.error(f"❌ Failed to provision LoRAs to {cluster_name}: {e}")

# ============================================================
# Background Tasks
# ============================================================
async def sp2_routing_loop():
    logger.info("⏳ SP2 Global Routing Loop is waiting for /start signal...")
    await system_start_event.wait()  # 等待啟動信號
    logger.info("🚀 SP2 Global Routing Loop started (10s interval).")
    
    # 啟動時立刻跑一次，不要呆等 10 秒
    await sync_global_routing()
    
    while True:
        await asyncio.sleep(10)
        await sync_global_routing()

async def sp1_provisioning_loop():
    logger.info("⏳ SP1 Provisioning Loop is waiting for /start signal...")
    await system_start_event.wait()  # 等待啟動信號
    logger.info("🚀 SP1 Provisioning Loop started (1 hour interval).")
    
    # 啟動時立刻跑一次，不要呆等 1 小時
    await fetch_cluster_stats()
    hybrid_forecasting_engine()
    await run_sp1_provisioning()
    
    while True:
        await asyncio.sleep(3600)  # <--- 確保是 3600 秒 (1 小時)
        await fetch_cluster_stats()
        hybrid_forecasting_engine()
        await run_sp1_provisioning()

# ============================================================
# Lifecycle & API Models
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
    
    mapping_path = "lora_mapping.json"
    counts_path = "lora_hourly_counts.json"
    
    if os.path.exists(mapping_path) and os.path.exists(counts_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                azure_mapping = json.load(f)
            with open(counts_path, "r", encoding="utf-8") as f:
                hourly_counts = json.load(f)
            
            for cluster_name, cluster_mapping in azure_mapping.items():
                for lora_id, azure_id in cluster_mapping.items():
                    historical_demand[cluster_name][lora_id] = hourly_counts[azure_id][:SEQ_LENGTH]
                    
            logger.info("✅ Successfully initialized historical_demand with 48 hours of Seed Data.")
        except Exception as e:
            logger.error(f"❌ Error loading mapping or counts JSON: {e}")
    else:
        logger.warning("⚠️ lora_mapping.json 或 lora_hourly_counts.json 不存在，歷史資料將從 0 開始。")

    model_path = "./data/azure_lstm_32x_overfit.pth"
    try:
        logger.info(f"🔄 Loading LSTM model from {model_path} onto {device}...")
        lstm_model = OverfitLSTM(
            num_funcs=240, 
            num_hours=T_TOTAL_HOURS, 
            func_emb_dim=32, 
            hour_emb_dim=32, 
            hidden=128, 
            num_layers=2
        ).to(device)
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        lstm_model.eval()
        logger.info("✅ LSTM Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load LSTM model: {e}")

    asyncio.create_task(sp2_routing_loop())
    asyncio.create_task(sp1_provisioning_loop())
    
    yield
        
app = FastAPI(title="Edge Federation Orchestrator (EFO)", lifespan=lifespan)

# ============================================================
# API Routes
# ============================================================
@app.post("/register_cluster")
async def register_cluster(req: RegisterClusterRequest):
    active_clusters[req.cluster_name] = req.control_node_url
    logger.info(f"🔗 Cluster Registered and Active: {req.cluster_name} -> {req.control_node_url}")
    
    # 移除原本在這裡的 asyncio.create_task(run_sp1_provisioning())
    # 確保不會在尚未完全註冊時就偷跑分配
    
    return {
        "status": "ok",
        "cluster_name": req.cluster_name,
        "metadata": global_lora_metadata
    }

@app.post("/start")
async def start_system():
    """接收外部呼叫，觸發整個 EFO 的背景排程啟動"""
    if system_start_event.is_set():
        return {"status": "already running"}
        
    system_start_event.set()
    logger.info("🏁 System START signal received! Background tasks are now running.")
    return {"status": "started"}

@app.get("/status")
async def get_status():
    history_lengths = {
        cluster: {lora: len(history) for lora, history in loras.items()}
        for cluster, loras in historical_demand.items()
    }
    return {
        "is_running": system_start_event.is_set() if system_start_event else False,
        "configured_clusters": configured_clusters,
        "active_clusters": active_clusters,
        "total_loras": len(global_lora_metadata),
        "metadata_keys": global_lora_metadata,
        "disk_inventory": global_lora_disk_inventory,
        "history_lengths": history_lengths,
        "current_simulated_hour": current_time_step
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9100))
    uvicorn.run(app, host="0.0.0.0", port=port)
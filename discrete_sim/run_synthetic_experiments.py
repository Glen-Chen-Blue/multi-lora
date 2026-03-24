#!/usr/bin/env python3
"""
CLI entry point for Synthetic Multi-LoRA Simulation Experiments.
Uses Poisson distribution for arrival times and Zipf distribution for LoRA selection.
"""

import os
import sys
import json
import pandas as pd

# 1. 自動定位專案根目錄 (往上推一層)
# __file__ 是這個腳本的位置 (discrete_sim/run_synthetic_experiments.py)
# os.path.dirname 兩次會指回專案根目錄 (multi-lora/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 將專案根目錄加入 sys.path，確保能讀取到 config.py 以及 discrete_sim 套件
sys.path.insert(0, PROJECT_ROOT)

# 2. 統一使用完整套件路徑進行 import
from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator
from cost2 import parse_logs
# ==========================================
# 實驗參數設定區 (可自由修改)
# ==========================================
SIMULATION_DAYS = 2                  # 模擬天數 (跑3天)
NUM_CLUSTERS = 3                     # Control Node 數量
COMPUTE_NODES_PER_CLUSTER = 5        # 每個 Cluster 的 Compute Node 數量
RPS_LIST = [i for i in range(1, 21)]   # X軸：全域目標 RPS 陣列 (Requests Per Second)
ZIPF_S_PARAMETER = 1.2               # Zipf 分佈傾斜度 (預設 1.2)

LORA_MAPPING_PATH = os.path.join(PROJECT_ROOT, "information", "lora_mapping.json")
OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
TRACE_CSV_DUMMY = os.path.join(PROJECT_ROOT, "information", "simulation_data.csv")
METADATA_DIR = os.path.join(PROJECT_ROOT, "information")

# 六種 Baseline 的定義 (對應 Experiment 1~6)
BASELINE_STRATEGIES = [
    (1, "Experiment 1 (SP1+SP2)"),
    (2, "Experiment 2 (SP1+SP2 w/o semantic)"),
    (3, "Experiment 3 (SP1+Random)"),
    (4, "Experiment 4 (LRU+Random)"),
    (5, "Experiment 5 (Dlora)"),
    (6, "Experiment 6 (Slora)")
]
# ==========================================

def extract_final_average_cost(log_path: str) -> float:
    """使用 cost2.py 中的 parse_logs 來精準計算並提取最後一刻的 average cost"""
    if not os.path.exists(log_path):
        print(f"[Warning] Log file not found: {log_path}")
        return 0.0
    
    try:
        # 呼叫 cost2.py 裡面的 parse_logs，它會幫我們把所有金額乘上 config 的費率
        df = parse_logs(log_path)
        
        if df.empty:
            print(f"[Warning] Parsed dataframe is empty for {log_path}")
            return 0.0
            
        # 直接拿 DataFrame 最後一個 row 的 cost_per_request
        final_cost = float(df['cost_per_request'].iloc[-1])
        return final_cost
        
    except Exception as e:
        print(f"[Error] Failed to calculate cost from {log_path}: {e}")
        return 0.0

def main():
    duration_hours = SIMULATION_DAYS * 24
    
    # 建立 3 個 cluster 各 5 個 compute node 的 Topology
    topology = {f"cluster_{i}": COMPUTE_NODES_PER_CLUSTER for i in range(1, NUM_CLUSTERS + 1)}
    target_clusters = list(topology.keys())
    
    results_data = []

    print("=" * 65)
    print("🚀 Starting Synthetic Experiments (Poisson + Zipf)")
    print(f"Topology: {NUM_CLUSTERS} Clusters, {COMPUTE_NODES_PER_CLUSTER} Nodes/Cluster")
    print(f"Duration: {SIMULATION_DAYS} Days")
    print(f"Testing Global RPS: {RPS_LIST}")
    print("=" * 65)

    for rps in RPS_LIST:
        # 將全域 RPS 平均分配給各個 Cluster 獨立生成 (例如總共 15，三個 cluster 各為 5)
        cluster_rps = rps / NUM_CLUSTERS

        for exp_id, exp_name in BASELINE_STRATEGIES:
            print(f"\n[{exp_name}] Processing System Load: {rps} RPS (Global) ...")
            
            # 將 log 輸出分開，避免覆寫
            out_dir = f"./results/synthetic/RPS_{rps}/Exp_{exp_id}_logs"
            os.makedirs(out_dir, exist_ok=True)

            # 1. 初始化原始的 Config 與 Simulation 物件
            config = SimulationConfig(
                experiment_id=exp_id,
                cluster_topology=topology,
                start_offset=0,           # Synthetic generator 不依賴 csv，設為 0
                duration_hours=duration_hours,
                target_clusters=target_clusters,
                seed=42,
                output_dir=out_dir,
                trace_csv=TRACE_CSV_DUMMY, # Dummy (會被覆寫)
                metadata_dir=METADATA_DIR
            )
            sim = Simulation(config)

            # 2. 【核心技巧】替換 Trace Generator
            # 這裡我們不修改 simulation.py 的程式碼，直接把我們寫好的 Synthetic Generator 塞進去
            synthetic_gen = SimSyntheticGenerator(
                lora_mapping_path=LORA_MAPPING_PATH,
                duration_s=duration_hours * 3600,
                target_clusters=target_clusters,
                rps_per_cluster=cluster_rps,
                zipf_s=ZIPF_S_PARAMETER,
                seed=42 + exp_id  # 微調亂數種子，保證實驗重現性
            )
            
            sim.trace = synthetic_gen
            sim.TOTAL_REQUESTS = synthetic_gen.total_requests
            sim.PAD_LEN = len(str(sim.TOTAL_REQUESTS))

            # 3. 執行模擬
            print(f"   -> Injecting {sim.TOTAL_REQUESTS} synthetic requests over {SIMULATION_DAYS} days...")
            sim.run()

            # 4. 擷取第三天結束時的 Average Cost
            log_file_path = os.path.join(out_dir, "efo_global_metrics.log")
            avg_cost = extract_final_average_cost(log_file_path)
            
            print(f"✅ {exp_name} @ {rps} RPS -> Final Avg Cost: {avg_cost:.4f}")
            
            # 記錄供最後畫圖用
            results_data.append({
                "Global_RPS": rps,
                "Strategy": exp_name,
                "Average_Cost": avg_cost
            })

    # 5. 將所有結果存成 CSV，交給畫圖腳本
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(OUTPUT_CSV_FILE, index=False)
    print("\n" + "=" * 65)
    print(f"🎉 All experiments finished! Data saved to {OUTPUT_CSV_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
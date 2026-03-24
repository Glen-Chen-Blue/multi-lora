#!/usr/bin/env python3
"""
CLI entry point for Synthetic Multi-LoRA Simulation Experiments.
Uses Poisson distribution for arrival times and Zipf distribution for LoRA selection.
(Multiprocessing Accelerated Version)
"""

import os
import sys
import json
import pandas as pd
import concurrent.futures  # 引入平行運算模組

# 1. 自動定位專案根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 2. 匯入必要的模組
from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator
from cost2 import parse_logs

# ==========================================
# 實驗參數設定區 (可自由修改)
# ==========================================
SIMULATION_DAYS = 2                  # 模擬天數 (跑2天)
NUM_CLUSTERS = 3                     # Control Node 數量
COMPUTE_NODES_PER_CLUSTER = 5        # 每個 Cluster 的 Compute Node 數量
RPS_LIST = [i for i in range(1, 21)] # X軸：全域目標 RPS 陣列 (1 到 20)
ZIPF_S_PARAMETER = 1.2               # Zipf 分佈傾斜度

LORA_MAPPING_PATH = os.path.join(PROJECT_ROOT, "information", "lora_mapping.json")
OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
TRACE_CSV_DUMMY = os.path.join(PROJECT_ROOT, "information", "simulation_data.csv")
METADATA_DIR = os.path.join(PROJECT_ROOT, "information")

# 六種 Baseline
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
    """使用 cost2.py 精準計算並提取最後一刻的 average cost"""
    if not os.path.exists(log_path):
        return 0.0
    try:
        df = parse_logs(log_path)
        if df.empty:
            return 0.0
        return float(df['cost_per_request'].iloc[-1])
    except Exception:
        return 0.0

def run_single_task(args):
    """這是一個獨立的任務函式，用來跑單一 (RPS, Baseline) 組合的模擬"""
    rps, cluster_rps, exp_id, exp_name, topology, target_clusters, duration_hours = args
    
    # 為了避免控制台畫面太亂，我們在進程內部只簡單印出開始與結束
    print(f"[START] RPS: {rps:2d} | {exp_name}")
    
    # 將 log 輸出分開
    out_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 初始化 Config 與 Simulation 物件
    config = SimulationConfig(
        experiment_id=exp_id,
        cluster_topology=topology,
        start_offset=0,
        duration_hours=duration_hours,
        target_clusters=target_clusters,
        seed=42,
        output_dir=out_dir,
        trace_csv=TRACE_CSV_DUMMY,
        metadata_dir=METADATA_DIR
    )
    
    # 暫時把 stdout 導向 null，或者你也可以保留。
    # 這裡我們保留原本的輸出，但因為是多進程，進度條畫面可能會稍微重疊，這是正常的。
    sim = Simulation(config)

    # 2. 替換 Trace Generator
    synthetic_gen = SimSyntheticGenerator(
        lora_mapping_path=LORA_MAPPING_PATH,
        duration_s=duration_hours * 3600,
        target_clusters=target_clusters,
        rps_per_cluster=cluster_rps,
        zipf_s=ZIPF_S_PARAMETER,
        seed=42 + exp_id + rps # 加入 rps 作為 seed 確保多樣性
    )
    
    sim.trace = synthetic_gen
    sim.TOTAL_REQUESTS = synthetic_gen.total_requests
    sim.PAD_LEN = len(str(sim.TOTAL_REQUESTS))

    # 3. 執行模擬
    sim.run()

    # 4. 擷取 Cost
    log_file_path = os.path.join(out_dir, "efo_global_metrics.log")
    avg_cost = extract_final_average_cost(log_file_path)
    
    print(f"[DONE ] RPS: {rps:2d} | {exp_name} -> Avg Cost: {avg_cost:.4f}")
    
    return {
        "Global_RPS": rps,
        "Strategy": exp_name,
        "Average_Cost": avg_cost
    }

def main():
    duration_hours = SIMULATION_DAYS * 24
    topology = {f"cluster_{i}": COMPUTE_NODES_PER_CLUSTER for i in range(1, NUM_CLUSTERS + 1)}
    target_clusters = list(topology.keys())
    
    # 準備所有需要被執行的任務清單
    tasks = []
    for rps in RPS_LIST:
        cluster_rps = rps / NUM_CLUSTERS
        for exp_id, exp_name in BASELINE_STRATEGIES:
            tasks.append((rps, cluster_rps, exp_id, exp_name, topology, target_clusters, duration_hours))

    print("=" * 65)
    print("🚀 Starting Parallel Synthetic Experiments (Poisson + Zipf)")
    print(f"Topology: {NUM_CLUSTERS} Clusters, {COMPUTE_NODES_PER_CLUSTER} Nodes/Cluster")
    print(f"Duration: {SIMULATION_DAYS} Days")
    print(f"Total Tasks: {len(tasks)} (RPS variations x Baselines)")
    print("=" * 65)

    results_data = []

    # 使用 ProcessPoolExecutor 進行多進程加速
    # max_workers 預設會使用你的 CPU 核心數 (例如 8 核心就會同時跑 8 個模擬)
    # 你也可以強制指定 max_workers=6 剛好讓 6 個 baseline 一起跑
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # submit 所有任務並等待完成
        futures = [executor.submit(run_single_task, task_args) for task_args in tasks]
        
        # as_completed 會在任何一個任務完成時馬上回傳結果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results_data.append(result)
            except Exception as exc:
                print(f"[Error] 某個實驗執行時發生錯誤: {exc}")

    # 將所有結果存成 CSV
    df_results = pd.DataFrame(results_data)
    # 依照 RPS 和策略排序，讓 CSV 看起來整齊
    if not df_results.empty:
        df_results = df_results.sort_values(by=['Global_RPS', 'Strategy'])
        df_results.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n" + "=" * 65)
    print(f"🎉 All parallel experiments finished! Data saved to {OUTPUT_CSV_FILE}")
    print("👉 執行 python draw_synthetic_cost.py 即可繪圖")
    print("=" * 65)


if __name__ == "__main__":
    # 在 Windows / macOS 底下跑 multiprocessing 需要保護進入點
    main()
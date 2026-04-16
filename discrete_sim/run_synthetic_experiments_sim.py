#!/usr/bin/env python3
"""
CLI entry point for Synthetic Multi-LoRA Simulation Experiments.
Uses Poisson distribution for arrival times and Zipf distribution for LoRA selection.
(Multiprocessing Accelerated Version - 25 Workers)
"""

import os
import gc
import sys
import json
import time
import pandas as pd
import concurrent.futures  
import contextlib          

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
SIMULATION_DAYS = 2                  # 模擬天數 (跑2天以達到論文穩定狀態)
NUM_CLUSTERS = 3                     # Control Node 數量
COMPUTE_NODES_PER_CLUSTER = 5        # 每個 Cluster 的 Compute Node 數量

# 設定你這次要跑的 RPS 區間
RPS_LIST = [i for i in range(11, 31)]
ZIPF_S_PARAMETER = 1.5               # Zipf 分佈傾斜度

LORA_MAPPING_PATH = os.path.join(PROJECT_ROOT, "information", "lora_mapping.json")
OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
TRACE_CSV_DUMMY = os.path.join(PROJECT_ROOT, "information", "simulation_data.csv")
METADATA_DIR = os.path.join(PROJECT_ROOT, "information")

# 五種 Baseline (已對齊論文與先前實驗的命名)
BASELINE_STRATEGIES = [
    (1, "Ours (SP1+SP2)"),
    (2, "Ours w/o Sem"),
    (3, "Ours w/o SP2"),
    (4, "dLoRA"),
    (5, "S-LoRA")
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
    
    # 【畫面清爽第一步】只印出任務開始
    print(f"[START] RPS: {rps:2d} | {exp_name}")
    
    # 將 log 輸出分開，避免多進程同時寫入同一個資料夾造成 I/O 衝突
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
    
    sim = Simulation(config)

    # 2. 替換 Trace Generator (加入 rps 與 exp_id 作為 seed 確保多樣性)
    synthetic_gen = SimSyntheticGenerator(
        lora_mapping_path=LORA_MAPPING_PATH,
        duration_s=duration_hours * 3600,
        target_clusters=target_clusters,
        rps_per_cluster=cluster_rps,
        zipf_s=ZIPF_S_PARAMETER,
        seed=42 + exp_id + rps 
    )
    
    sim.trace = synthetic_gen
    sim.TOTAL_REQUESTS = synthetic_gen.total_requests
    sim.PAD_LEN = len(str(sim.TOTAL_REQUESTS))

    # =========================================================================
    # 【核心修復：將生成的合成事件轉為 DataFrame，讓 EFO(SP1) 能正確預測】
    # 避免 EFO 去讀取錯誤的 dummy trace csv，導致預測需求一直為 0 的問題
    # =========================================================================
    records = []
    for t_ms, reqs in synthetic_gen._events.items():
        arr_sec = t_ms / 1000.0
        for cluster, lora_id in reqs:
            records.append({
                "arrival_sec": arr_sec,
                "cluster": cluster,
                "lora_id": lora_id
            })
    
    if records:
        df_synthetic = pd.DataFrame(records)
        sim.efo.simulation_df = df_synthetic
    else:
        sim.efo.simulation_df = pd.DataFrame(columns=["arrival_sec", "cluster", "lora_id"])
    # =========================================================================

    # 3. 執行模擬 【開啟黑洞屏蔽大量 print，大幅提升運算速度】
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    # 4. 擷取 Cost
    log_file_path = os.path.join(out_dir, "efo_global_metrics.log")
    avg_cost = extract_final_average_cost(log_file_path)
    
    # 【畫面清爽第二步】算完才印出結果
    print(f"[DONE ] RPS: {rps:2d} | {exp_name:<15} -> Avg Cost: NT${avg_cost:.4f}")
    
    # 主動釋放記憶體，避免 25 個 Process 狂吃 RAM
    del sim, config, synthetic_gen, df_synthetic 
    gc.collect()

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
    print("🚀 Starting Parallel Synthetic Experiments (Average Cost vs RPS)")
    print(f"Topology: {NUM_CLUSTERS} Clusters, {COMPUTE_NODES_PER_CLUSTER} Nodes/Cluster")
    print(f"Duration: {SIMULATION_DAYS} Days ({duration_hours} Hours)")
    print(f"Total Tasks: {len(tasks)} (RPS variations x Baselines)")
    print("=" * 65)

    results_data = []

    # =========================================================================
    # 【極速引擎啟動】使用 25 核心平行處理
    # =========================================================================
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(run_single_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results_data.append(result)
            except Exception as exc:
                print(f"[Error] 某個實驗執行時發生錯誤: {exc}")
    # =========================================================================

    # 5. 自動合併舊數據，保護之前辛苦跑出來的成果
    df_new = pd.DataFrame(results_data)
    if not df_new.empty:
        if os.path.exists(OUTPUT_CSV_FILE):
            print(f"\n📄 發現舊有的 {OUTPUT_CSV_FILE}，正在將新數據無縫合併...")
            df_old = pd.read_csv(OUTPUT_CSV_FILE)
            # 合併新舊數據，並剃除可能重複跑到的 RPS
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Global_RPS', 'Strategy'], keep='last')
        else:
            df_combined = df_new
            
        # 依照 RPS 和策略排序，讓 CSV 乾淨整齊
        # (將 Strategy 轉換為 Categorical 以便按照我們定義的順序排列)
        strategy_order = [s[1] for s in BASELINE_STRATEGIES]
        df_combined['Strategy'] = pd.Categorical(df_combined['Strategy'], categories=strategy_order, ordered=True)
        df_combined = df_combined.sort_values(by=['Global_RPS', 'Strategy'])
        df_combined.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n" + "=" * 65)
    print(f"🎉 All parallel experiments finished! Data safely saved to {OUTPUT_CSV_FILE}")
    print("👉 執行 python draw_synthetic_cost.py 即可繪製包含所有 RPS 的總圖")
    print("=" * 65)

if __name__ == "__main__":
    main()
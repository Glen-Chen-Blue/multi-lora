#!/usr/bin/env python3
import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

# 對應系統中的策略編號 (參考 run_max_throughput.py)
EXP_MAPPING = {
    "ours": {"id": 1, "label": "Ours (Cross-timescale)"},
    "dlora": {"id": 4, "label": "dLoRA"},
    "lru": {"id": 5, "label": "LRU / S-LoRA"}
}

def run_single_experiment(args):
    """
    這是在 ProcessPool 中執行的 Worker。
    因為多進程有獨立記憶體，必須在這裡重新載入並動態覆寫 config。
    """
    rps, strat_key = args
    exp_info = EXP_MAPPING[strat_key]
    exp_id = exp_info["id"]
    strat_label = exp_info["label"]
    
    # ==========================================
    # 1. 動態注入與覆蓋配置 (Monkey Patch)
    # ==========================================
    import config
    import discrete_sim.sim_control_node as scn
    import discrete_sim.sim_efo as sefo

    # 確保系統不因為逾時或壅塞而丟棄任何 Request
    config.ENABLE_DROP = False
    config.T_MAX = float('inf')
    config.MAX_QUEUE_SIZE = float('inf')
    config.MAX_WAITING_TIME = float('inf')
    config.PENALTY_DROP_BASE = float('inf')
    
    # 同步覆寫已經 import 到 module 層級的常數
    if hasattr(scn, 'T_MAX'): scn.T_MAX = float('inf')
    if hasattr(sefo, 'T_MAX'): sefo.T_MAX = float('inf')
    if hasattr(scn, 'ENABLE_DROP'): scn.ENABLE_DROP = False

    # ==========================================
    # 2. 設置模擬參數 (3叢集 x 5節點, 360秒)
    # ==========================================
    topology = {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}
    target_clusters = list(topology.keys())
    num_clusters = len(target_clusters)
    rps_per_cluster = rps / num_clusters
    duration_sec = 360
    duration_hours = duration_sec / 3600.0

    output_dir = f"./results/ttft_rps/exp_{exp_id}_rps_{rps}"
    os.makedirs(output_dir, exist_ok=True)

    sim_config = SimulationConfig(
        experiment_id=exp_id,
        cluster_topology=topology,
        start_offset=0,
        duration_hours=duration_hours,
        target_clusters=target_clusters,
        seed=42,
        output_dir=output_dir,
        metadata_dir="./information/"
    )

    try:
        sim = Simulation(sim_config)
        
        # 使用極端傾斜的 Zipf (10.0) 產生 Synthetic Traces，消除 LoRA Miss 影響
        sim.trace = SimSyntheticGenerator(
            lora_mapping_path="./information/lora_mapping.json",
            duration_s=duration_sec,
            target_clusters=target_clusters,
            rps_per_cluster=rps_per_cluster,
            zipf_s=10.0 
        )
        sim.TOTAL_REQUESTS = sim.trace.total_requests
        
        # 關鍵：將合成的未來負載資料餵給 EFO 以利 SP1 規劃
        if hasattr(sim, 'efo'):
            sim.efo.simulation_df = sim.trace.to_dataframe()

        # ==========================================
        # 3. 靜默執行模擬
        # ==========================================
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                sim.run()

        # ==========================================
        # 4. 收集 P95 TTFT
        # ==========================================
        ttft_records = sim.ttft_records
        if len(ttft_records) > 0:
            p95_ttft = np.percentile(ttft_records, 95)
        else:
            p95_ttft = 0.0
            
    except Exception as e:
        print(f"[!] Error in {strat_label} at RPS {rps}: {str(e)}")
        p95_ttft = float('nan')

    print(f"[+] Finished -> Strategy: {strat_label:<20} | RPS: {rps:>2d} | P95 TTFT: {p95_ttft:7.3f}s")
    
    return {
        'RPS': rps, 
        'Strategy': strat_key, 
        'Label': strat_label,
        'P95_TTFT': p95_ttft
    }

def main():
    print("=" * 70)
    print("=== Parallel P95 TTFT vs. RPS Analysis (No-Drop Mode) ===")
    print("=" * 70)

    # 產生任務 (RPS 1 到 25)
    rps_list = list(range(1, 26))
    strategies = ["ours", "dlora", "lru"]
    
    tasks = [(rps, strat) for strat in strategies for rps in rps_list]
    results = []

    # 使用 25 個 Worker 進行極速平行運算
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    # 將數據排序與存檔
    df = pd.DataFrame(results)
    df = df.sort_values(by=['Strategy', 'RPS'])
    
    os.makedirs('results/ttft_rps', exist_ok=True)
    csv_path = 'results/ttft_rps/p95_ttft_results.csv'
    df.to_csv(csv_path, index=False)
    
    # ==========================================
    # 5. 繪製學術等級的 P95 TTFT 比較圖
    # ==========================================
    plt.figure(figsize=(9, 6), dpi=150)
    colors = {'ours': '#1f77b4', 'dlora': '#ff7f0e', 'lru': '#2ca02c'}
    markers = {'ours': 'o', 'dlora': 's', 'lru': '^'}
    
    for strat_key in strategies:
        strat_df = df[df['Strategy'] == strat_key]
        label = EXP_MAPPING[strat_key]['label']
        plt.plot(strat_df['RPS'], strat_df['P95_TTFT'], 
                 label=label, 
                 color=colors[strat_key], 
                 marker=markers[strat_key],
                 linewidth=2.5, markersize=8)

    # 圖表美化 (貼合 TNSM / IEEE 規格)
    plt.xlabel('System Workload (Requests per Second)', fontsize=14, fontweight='bold')
    plt.ylabel('95th Percentile TTFT (s)', fontsize=14, fontweight='bold')
    plt.title('P95 Tail Latency under Extreme Saturation', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 針對排隊延遲（Queueing Delay）可能造成的指數上升，增加 Y 軸高度限制 (避免某條線飆到無限大壓縮其他線)
    # plt.ylim(0, max_reasonable_delay) 若有需要可以自行開啟
    
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    plot_path = 'results/ttft_rps/p95_ttft_vs_rps.png'
    plt.savefig(plot_path, dpi=300, format='png')
    
    print("\n" + "=" * 70)
    print(f"🎉 Simulation Completed! Data saved to {csv_path}")
    print(f"📊 Plot saved to {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
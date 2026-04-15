#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

EXP_MAPPING = {
    "ours": {"id": 1, "label": "Ours (SP1+SP2)"},
    "ours_no_sem": {"id": 2, "label": "Ours w/o Sem"},
    "ours_no_sp2": {"id": 3, "label": "Ours w/o SP2"},
    "dlora": {"id": 4, "label": "dLoRA"},
    "lru": {"id": 5, "label": "S-LoRA"}
}

def run_single_experiment(args):
    rps, strat_key = args
    exp_info = EXP_MAPPING[strat_key]
    exp_id = exp_info["id"]
    strat_label = exp_info["label"]
    
    # 動態注入配置 (Monkey Patch)
    import config
    import discrete_sim.sim_control_node as scn
    import discrete_sim.sim_efo as sefo

    # 確保跟 max_throughput 的高容量設定一致
    config.BATCH_SIZE_MERGED = 25
    config.BATCH_SIZE_UNMERGED_BASE = 10
    scn.MERGED_CAPACITY = 25
    scn.UNMERGED_CAPACITY = 10

    config.ENABLE_DROP = False
    config.MAX_QUEUE_SIZE = float('inf')
    config.MAX_WAITING_TIME = float('inf')
    
    # 保護 T_MAX 不被無限大化，維持 SP2 Z_debt 的正常運算
    if hasattr(config, 'T_MAX') and config.T_MAX == float('inf'):
        config.T_MAX = 6.0 
    scn.T_MAX = getattr(config, 'T_MAX', 6.0)
    sefo.T_MAX = getattr(config, 'T_MAX', 6.0)
    scn.ENABLE_DROP = False

    topology = {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}
    target_clusters = list(topology.keys())
    num_clusters = len(target_clusters)
    rps_per_cluster = rps / num_clusters
    duration_sec = 360 # 0.1 小時
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
        
        sim.trace = SimSyntheticGenerator(
            lora_mapping_path="./information/lora_mapping.json",
            duration_s=duration_sec,
            target_clusters=target_clusters,
            rps_per_cluster=rps_per_cluster,
            zipf_s=10.0 
        )
        sim.TOTAL_REQUESTS = sim.trace.total_requests
        
        if hasattr(sim, 'efo'):
            sim.efo.simulation_df = sim.trace.to_dataframe()

        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                sim.run()

        # 精準計算 P95
        ttft_records = [r for r in sim.ttft_records if not np.isnan(r)]
        if len(ttft_records) > 0:
            p95_ttft = np.percentile(ttft_records, 95)
        else:
            p95_ttft = float('nan')
            
    except Exception as e:
        print(f"[!] Error in {strat_label} at RPS {rps}: {str(e)}")
        p95_ttft = float('nan')

    print(f"[+] Finished -> Strategy: {strat_label:<15} | RPS: {rps:>2d} | P95 TTFT: {p95_ttft:7.3f}s")
    
    return {
        'RPS': rps, 
        'Strategy': strat_key, 
        'Label': strat_label,
        'P95_TTFT': p95_ttft
    }

def main():
    print("=" * 70)
    print("=== Parallel P95 TTFT vs. RPS Analysis (No-Drop Saturation) ===")
    print("=" * 70)

    rps_list = list(range(1, 26))
    strategies = ["ours", "ours_no_sem", "ours_no_sp2", "dlora", "lru"]
    
    tasks = [(rps, strat) for strat in strategies for rps in rps_list]
    results = []

    # 極速平行處理
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame(results)
    df = df.sort_values(by=['Strategy', 'RPS'])
    
    os.makedirs('results/ttft_rps', exist_ok=True)
    csv_path = 'results/ttft_rps/p95_ttft_results.csv'
    df.to_csv(csv_path, index=False)
    
    # 繪製學術級圖表
    plt.figure(figsize=(9, 6), dpi=150)
    
    style_map = {
        'ours': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},        # Red
        'ours_no_sem': {'color': '#9467bd', 'marker': 'v', 'linestyle': '--'},# Purple
        'dlora': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},       # Orange
        'ours_no_sp2': {'color': '#8c564b', 'marker': 'x', 'linestyle': '-.'},# Brown
        'lru': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-'}          # Blue
    }
    
    for strat_key in strategies:
        strat_df = df[df['Strategy'] == strat_key]
        label = EXP_MAPPING[strat_key]['label']
        st = style_map[strat_key]
        
        plt.plot(strat_df['RPS'], strat_df['P95_TTFT'], 
                 label=label, 
                 color=st['color'], 
                 marker=st['marker'],
                 linestyle=st['linestyle'],
                 linewidth=2.5, markersize=8)

    plt.xlabel('System Workload (Requests per Second)', fontsize=14, fontweight='bold')
    plt.ylabel('95th Percentile TTFT (s)', fontsize=14, fontweight='bold')
    plt.title('P95 Tail Latency under System Saturation', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 自動適應 Y 軸，避免被極端值壓平
    max_reasonable_val = df['P95_TTFT'][df['P95_TTFT'] < 100].max() * 1.5
    plt.ylim(0, max(max_reasonable_val, 10)) 
    
    # 排序 Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index(EXP_MAPPING[s]['label']) for s in strategies]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=12, loc='upper left')
    
    plt.tight_layout()
    
    plot_path = 'p95_ttft_vs_rps.png'
    plt.savefig(plot_path, dpi=300, format='png')
    
    print("\n" + "=" * 70)
    print(f"🎉 Simulation Completed! Data saved to {csv_path}")
    print(f"📊 Plot saved to {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
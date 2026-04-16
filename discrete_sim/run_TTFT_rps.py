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
    "dlora": {"id": 5, "label": "dLoRA"},  # 修正：dLoRA 對應 ID 5
    "lru": {"id": 4, "label": "S-LoRA"}    # 修正：S-LoRA 對應 ID 4
}

def run_single_experiment(args):
    rps, strat_key = args
    exp_info = EXP_MAPPING[strat_key]
    exp_id = exp_info["id"]
    strat_label = exp_info["label"]
    
    # 動態注入配置 (Monkey Patch)
    import config
    import discrete_sim.sim_control_node_ as scn
    import discrete_sim.sim_efo as sefo
    import discrete_sim.sim_compute_node as s_compute

    config.ENABLE_DROP = False
    config.MAX_QUEUE_SIZE = float('inf')
    config.MAX_WAITING_TIME = float('inf')
    
    # 保護 T_MAX 不被無限大化，維持 SP2 Z_debt 的正常運算
    if hasattr(config, 'T_MAX') and config.T_MAX == float('inf'):
        config.T_MAX = 6.0 
    scn.T_MAX = getattr(config, 'T_MAX', 6.0)
    sefo.T_MAX = getattr(config, 'T_MAX', 6.0)
    scn.ENABLE_DROP = False

    # ==========================================
    # [⭐ 核心修復 1] 強制滿載測試時所有節點必須火力全開
    # ==========================================
    if not hasattr(s_compute.SimComputeNode, '_original_full_reset'):
        s_compute.SimComputeNode._original_full_reset = s_compute.SimComputeNode.full_reset
        def patched_full_reset(self):
            self._original_full_reset()
            self.status = s_compute.NodeStatus.ACTIVE # 強制保持 ACTIVE，避免 SP1 重置把它們催眠
        s_compute.SimComputeNode.full_reset = patched_full_reset
    
    # 讓 autoscale 即使沒有 drop 也能因為 Z_debt 喚醒節點 (保險機制)
    scn.SCALE_UP_DROP_THRESHOLD = 0
    # ==========================================

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
        
        # 強制喚醒所有節點 (對抗一開始只有 n1 是 active 的限制)
        for cluster_nodes in sim.all_compute_nodes.values():
            for node in cluster_nodes:
                node.activate()
                
        sim.trace = SimSyntheticGenerator(
            lora_mapping_path="./information/lora_mapping.json",
            duration_s=duration_sec,
            target_clusters=target_clusters,
            rps_per_cluster=rps_per_cluster,
            zipf_s=8 
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

    rps_list = list(range(1, 26, 1))
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
    csv_path = 'p95_ttft_results.csv'
    df.to_csv(csv_path, index=False)

    # 改成和下面那份 code 接近的風格
    fig, ax = plt.subplots(figsize=(12, 8))

    # 使用 matplotlib 預設色盤，不再手動指定 color / marker / linestyle
    for strat_key in strategies:
        strat_df = df[df['Strategy'] == strat_key]
        label = EXP_MAPPING[strat_key]['label']

        if strat_df.empty:
            continue

        ax.plot(
            strat_df['RPS'],
            strat_df['P95_TTFT'],
            linewidth=2.5,
            marker='o',
            markersize=7,
            label=label
        )

    ax.set_xlabel('System Workload (Requests per Second)', fontsize=12)
    ax.set_ylabel('95th Percentile TTFT (s)', fontsize=12)

    # 保留你原本的 y 軸範圍
    ax.set_ylim(0, 15)

    # SLO 線，風格改得更接近你下面那份 code
    ax.axhline(
        y=6,
        color='gray',
        linestyle='--',
        linewidth=2,
        alpha=0.8,
        label='TTFT SLO'
    )

    # 標註文字
    ax.text(
        x=max(rps_list) * 0.95,
        y=6 + 0.3,
        s='TTFT SLO',
        color='gray',
        fontsize=12,
        ha='right'
    )

    # grid 風格跟下面那份 code 對齊
    ax.grid(True, linestyle='--', alpha=0.4)

    # legend 保持簡潔
    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()

    plot_path = 'p95_ttft_vs_rps.png'
    plt.savefig(plot_path, dpi=300, format='png')
    plt.close()

    print("\n" + "=" * 70)
    print(f"🎉 Simulation Completed! Data saved to {csv_path}")
    print(f"📊 Plot saved to {plot_path}")
    print("=" * 70)
if __name__ == '__main__':
    main()
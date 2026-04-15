#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import subprocess
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

EXP_LABELS = {
    1: "Ours",
    2: "w/o-Sem",
    3: "w/o-SP2",
    4: "dLoRA",
    5: "S-LoRA"
}

def get_mapping_path(project_root):
    return os.path.join(project_root, "information/lora_mapping.json")

def run_worker_process(exp_id, rps):
    """獨立啟動 Worker，確保不會有跨實驗的 Memory/Cache 污染"""
    cmd = [sys.executable, __file__, "--worker", "--exp_id", str(exp_id), "--rps", str(rps)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT_JSON:'):
            data = json.loads(line.split('RESULT_JSON:')[1])
            data['exp_id'] = exp_id
            data['rps'] = rps
            print(f"[Master] ✅ 完成 {EXP_LABELS[exp_id]:<10} | RPS = {rps:2.0f} -> P95 TTFT: {data['p95_ttft']:5.2f}s")
            return data
            
    print(f"[Master] ❌ 錯誤: {EXP_LABELS[exp_id]} | RPS = {rps} 失敗。\n{result.stderr}")
    return None

def worker_simulation_logic(exp_id, total_rps):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    import config
    
    # 關閉 Baseline 的 Semantic 能力 (除了 Ours 以外)
    if exp_id != 1 and hasattr(config, 'USE_SEMANTIC'):
        config.USE_SEMANTIC = False
    if exp_id == 3: # w/o-SP2 專屬降級設定
        config.EXECUTION_MODE = "unmerged"

    from discrete_sim.sim_types import SimulationConfig
    from discrete_sim.simulation import Simulation
    from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

    topology = {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}
    target_clusters = list(topology.keys())
    rps_per_cluster = total_rps / len(target_clusters)
    
    # ---------------------------------------------------------
    # 🎯 關鍵修復：先產生 Trace 並實體寫入 CSV，讓 EFO (SP1) 能完美解析
    # ---------------------------------------------------------
    gen = SimSyntheticGenerator(
        lora_mapping_path=get_mapping_path(project_root),
        duration_s=1800,
        target_clusters=target_clusters,
        rps_per_cluster=rps_per_cluster,
        zipf_s=1.2
    )
    df = gen.to_dataframe()
    
    # 補齊 EFO 需要的時間戳欄位
    if 'arrive_timestamp' not in df.columns and 'arrival_time_ms' in df.columns:
        df['arrive_timestamp'] = df['arrival_time_ms'] / 1000.0
        
    temp_csv_path = os.path.join(project_root, f"temp_trace_exp{exp_id}_rps{total_rps}.csv")
    df.to_csv(temp_csv_path, index=False)
    
    # 強制將 config 導向我們剛做好的 Synthetic CSV
    config.trace_csv = temp_csv_path

    sim_config = SimulationConfig(
        experiment_id=exp_id,
        cluster_topology=topology,
        start_offset=0,
        duration_hours=0.5,
        target_clusters=target_clusters,
        seed=42,
        output_dir=os.path.join(project_root, f"results/ttft/exp_{exp_id}_rps_{total_rps}"),
        metadata_dir=os.path.join(project_root, "information")
    )
    
    # 初始化模擬器 (此時 EFO 會讀取 temp_csv_path 並做出正確預測部署)
    sim = Simulation(sim_config)
    sim.trace = gen  # 原生覆寫為 Generator
    sim.TOTAL_REQUESTS = sim.trace.total_requests

    # 靜默執行
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    # ---------------------------------------------------------
    # 🎯 收集真正的 QoE (P95 TTFT)
    # ---------------------------------------------------------
    ttft_list = sim.ttft_records.copy()
    
    # 1. 處理原生 Drop (賦予 15s QoE 懲罰)
    # 只有容量爆滿、或 SLO Timeout 才會觸發原生 Drop
    num_dropped = len(getattr(sim, 'dropped_requests', []))
    ttft_list.extend([15.0] * num_dropped)

    # 2. 處理模擬結束時死卡在 Queue 的殘留請求
    for c_name, cn in sim.control_nodes.items():
        for req in cn.pending_queue:
            wait_time_s = (sim.clock.now() - req.arrival_time_ms) / 1000.0
            ttft_list.append(wait_time_s)

    p95_ttft_sec = np.percentile(ttft_list, 95) if ttft_list else 0.0

    # 清理暫存檔
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    result_dict = {"p95_ttft": p95_ttft_sec}
    print(f"RESULT_JSON:{json.dumps(result_dict)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--exp_id", type=int)
    parser.add_argument("--rps", type=float)
    args = parser.parse_args()

    if args.worker:
        worker_simulation_logic(args.exp_id, args.rps)
    else:
        print("=" * 65)
        print("🚀 Starting P95 TTFT vs. RPS Analysis (Paper Table I Aligned)")
        print("=" * 65)
        
        target_rps_list = list(range(1, 21))
        target_exps = [1, 2, 3, 4, 5]
        
        tasks = [(exp_id, rps) for rps in target_rps_list for exp_id in target_exps]
        results = []

        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(run_worker_process, t[0], t[1]) for t in tasks]
            for future in futures:
                res = future.result()
                if res: results.append(res)

        os.makedirs("results", exist_ok=True)
        csv_path = "results/p95_ttft_vs_rps_results.csv"
        results.sort(key=lambda x: (x['exp_id'], x['rps']))
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Algorithm", "RPS", "P95_TTFT"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "Algorithm": EXP_LABELS[r['exp_id']],
                    "RPS": r['rps'],
                    "P95_TTFT": round(r['p95_ttft'], 4)
                })
        
        print(f"\n[Master] ✅ 數據已儲存至 {csv_path}")

        # ==========================================
        # 📊 繪製圖表 (對齊論文 Fig. 4 視覺風格)
        # ==========================================
        plt.figure(figsize=(9, 6), dpi=150)
        
        styles = {
            1: {"marker": "o", "color": "#d62728", "linestyle": "-", "label": "Ours"},
            2: {"marker": "^", "color": "#2ca02c", "linestyle": "-", "label": "w/o-Sem"},
            3: {"marker": "s", "color": "#ff7f0e", "linestyle": "-", "label": "w/o-SP2"},
            4: {"marker": "d", "color": "#9467bd", "linestyle": "--", "label": "dLoRA"},
            5: {"marker": "x", "color": "#1f77b4", "linestyle": "--", "label": "S-LoRA"}
        }
        
        for exp_id in target_exps:
            exp_data = [r for r in results if r['exp_id'] == exp_id]
            if not exp_data: continue
            
            x_vals = [r['rps'] for r in exp_data]
            y_vals = [min(r['p95_ttft'], 15.0) for r in exp_data] 

            st = styles[exp_id]
            plt.plot(
                x_vals, y_vals, 
                label=st["label"], marker=st["marker"], color=st["color"], 
                linestyle=st["linestyle"], linewidth=2.5, markersize=7
            )
                     
        plt.axhline(y=6.0, color='gray', linestyle='dashdot', label='SLO Target (6.0s)', linewidth=2)
        
        plt.xlabel("System Load (RPS)", fontsize=13, fontweight='bold')
        plt.ylabel("P95 TTFT (s)", fontsize=13, fontweight='bold')
        plt.xticks(target_rps_list)
        plt.ylim(0, 15.5) 
        plt.grid(True, which="major", ls="-", alpha=0.3)
        
        plt.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        
        img_path = "results/p95_ttft_vs_rps.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"[Master] 🎉 圖表已儲存至 {img_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""CLI entry point for discrete multi-lora simulation experiments.
(Multiprocessing Accelerated & Silent Output Version)
"""

import argparse
import json
import os
import sys
import concurrent.futures
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discrete_sim.sim_types import SimulationConfig, EXPERIMENT_CONFIGS
from discrete_sim.simulation import Simulation


def run_single(args, experiment_id: int):
    """執行單一實驗的核心函式，並屏蔽繁雜的輸出"""
    topology = json.loads(args.topology)

    config = SimulationConfig(
        experiment_id=experiment_id,
        cluster_topology=topology,
        start_offset=args.start_offset,
        duration_hours=args.duration_hours,
        target_clusters=json.loads(args.target_clusters) if args.target_clusters else None,
        seed=args.seed,
        output_dir=os.path.join(args.output_dir, f"experiment_single_cluster_2nodes{experiment_id}_logs"),
        trace_csv=args.trace_csv,
        metadata_dir=args.metadata_dir,
    )

    exp = EXPERIMENT_CONFIGS[experiment_id]
    
    # 僅印出開始提示，保持畫面乾淨
    print(f"[START] Experiment {experiment_id} | EFO: {exp.efo_type} | Control: {exp.control_type} | Disk: {exp.disk_capacity_gb} GB")

    sim = Simulation(config)
    
    # 【核心加速修改】：將 simulation 內部的所有 print 導向黑洞，避免 I/O 拖慢速度
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()
            
    # 印出完成提示
    print(f"[DONE ] Experiment {experiment_id} completed successfully.")
    return experiment_id


def run_plot(args):
    """Generate cost2.py chart from simulation outputs."""
    # Find all experiment log dirs
    log_files = []
    labels = []
    label_map = {
        1: "Experiment 1 (SP1+SP2)",
        2: "Experiment 2 (SP1+SP2 w/o semantic)",
        3: "Experiment 3 (SP1+Random)",
        4: "Experiment 4 (LRU+Random)",
        5: "Experiment 5 (Dlora)",
        6: "Experiment 6 (Slora)",
    }

    for i in range(1, 7):
        log_path = os.path.join(args.output_dir, f"experiment_single_cluster_2nodes{i}_logs", "efo_global_metrics.log")
        if os.path.exists(log_path):
            log_files.append(log_path)
            labels.append(label_map.get(i, f"Experiment {i}"))

    if not log_files:
        print("No experiment logs found!")
        return

    # Import cost2.py functions
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cost2 import plot_multiple_total_costs_with_simulation_bg

    start_days = args.start_offset // 86400
    plot_multiple_total_costs_with_simulation_bg(
        log_files=log_files,
        labels=labels,
        output_path=os.path.join(args.output_dir, "cost_per_request"),
        csv_path=args.trace_csv,
        target_clusters=json.loads(args.target_clusters) if args.target_clusters else list(json.loads(args.topology).keys()),
        speed_rate=1.0,
        start_offset_days=start_days,
        duration_hours=args.duration_hours,
        bin_minutes=5,
    )
    print(f"Chart saved to {os.path.join(args.output_dir, f'cost_per_request{start_days}.png')}")


def main():
    parser = argparse.ArgumentParser(description="Discrete Multi-LoRA Simulation")
    parser.add_argument("--experiment", type=int, choices=range(1, 7), help="Experiment ID (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all 6 experiments concurrently")
    parser.add_argument("--plot", action="store_true", help="Generate cost2.py chart from results")
    parser.add_argument("--topology", default='{"cluster_1": 2}', help='JSON topology: {"cluster_name": num_compute_nodes}')
    parser.add_argument("--start-offset", type=int, default=172800, help="CSV trace start offset in seconds")
    parser.add_argument("--duration-hours", type=int, default=8, help="Simulation duration in hours")
    parser.add_argument("--target-clusters", default=None, help="JSON list of target clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="./results/", help="Output directory")
    parser.add_argument("--trace-csv", default="./information/simulation_data.csv", help="Path to trace CSV")
    parser.add_argument("--metadata-dir", default="./information/", help="Path to metadata directory")

    args = parser.parse_args()

    if args.plot:
        run_plot(args)
    elif args.all:
        print("=" * 65)
        print("🚀 Starting Parallel Experiments 1 to 6")
        print("=" * 65)
        
        # 使用 ProcessPoolExecutor 讓 6 個實驗同時平行執行
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(run_single, args, i) for i in range(1, 7)]
            
            # 等待所有平行任務完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Error] Experiment execution failed: {e}")
                    
        print("\n🎉 All 6 experiments completed in parallel!")
        run_plot(args)  # 跑完自動畫圖
        
    elif args.experiment:
        run_single(args, args.experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
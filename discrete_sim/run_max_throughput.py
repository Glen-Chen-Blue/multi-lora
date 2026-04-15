#!/usr/bin/env python3
import os
import sys
import json
import csv
import concurrent.futures
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator
from discrete_sim.sim_network import generate_network_params, SimNetwork

SCALES = {
    "1c_2n": {"cluster_1": 2},
    "2c_3n": {"cluster_1": 1, "cluster_2": 2},
    "10c_50n": {f"cluster_{i}": 5 for i in range(1, 11)}
}

# 修正：對齊 sim_types.py 的演算法行為
EXP_LABELS = {
    1: "Ours (SP1+SP2)",
    2: "Ours w/o Sem",
    3: "Ours w/o SP2",
    4: "dLoRA",  
    5: "S-LoRA"    
}

def get_mapping_path(num_clusters):
    if num_clusters == 10:
        return "./information/lora_mapping_10c.json"
    return "./information/lora_mapping.json"

def run_single_pressure_test(exp_id: int, scale_name: str, total_rps: float):
    topology = SCALES[scale_name]
    target_clusters = list(topology.keys())
    num_clusters = len(target_clusters)
    rps_per_cluster = total_rps / num_clusters
    
    sim_config = SimulationConfig(
        experiment_id=exp_id,
        cluster_topology=topology,
        start_offset=0,
        duration_hours=0.5,
        target_clusters=target_clusters,
        seed=42,
        output_dir=f"./results/throughput/{scale_name}/exp_{exp_id}_rps_{int(total_rps)}",
        metadata_dir="./information/"
    )
    
    sim = Simulation(sim_config)
    sim.trace = SimSyntheticGenerator(
        lora_mapping_path=get_mapping_path(num_clusters),
        duration_s=1800,
        target_clusters=target_clusters,
        rps_per_cluster=rps_per_cluster,
        zipf_s=1.2
    )
    sim.TOTAL_REQUESTS = sim.trace.total_requests
    
    if num_clusters > 3:
        sim.network = SimNetwork(seed=42, params=generate_network_params(num_clusters))
        if hasattr(sim, 'efo'):
            sim.efo.network = sim.network

    # ==========================================
    # [關鍵修復] 將合成預測資料餵給 EFO
    # ==========================================
    if hasattr(sim, 'efo'):
        sim.efo.simulation_df = sim.trace.to_dataframe()
    # ==========================================

    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    total_sent = sim.stats["sent"]
    total_finished = sim.stats["finished"]
    actual_throughput = total_finished / 1800.0
    is_saturated = (total_finished < total_sent * 0.95)
    
    return total_sent, total_finished, actual_throughput, is_saturated

def search_worker(args):
    """
    這是在 ProcessPool 中執行的 Worker。
    [⚠️ 核心關鍵]：多進程會產生獨立記憶體，必須在這裡重新載入並覆寫 config！
    """
    exp_id, scale_name = args
    label = EXP_LABELS[exp_id]
    
    import config
    config.PENALTY_DROP_BASE = float('inf')
    config.BATCH_SIZE_MERGED = 25        
    config.BATCH_SIZE_UNMERGED_BASE = 10 
    config.DELTA_LOAD_S = 0.4            
    config.LORA_CACHE_CAPACITY = 8       

    num_nodes = sum(SCALES[scale_name].values())
    current_rps = 10.0 if num_nodes <= 5 else 100.0
    step = 10.0 if num_nodes <= 5 else 50.0
    
    max_recorded_throughput = 0.0
    
    print(f"🚀 [START] {scale_name} | {label}...")
    while True:
        sent, finished, throughput, is_saturated = run_single_pressure_test(exp_id, scale_name, current_rps)
        
        if is_saturated:
            final_throughput = max(max_recorded_throughput, throughput)
            print(f"💥 [DONE] {scale_name} | {label} SATURATED! Max: {final_throughput:.2f} req/s")
            return {
                "Scale": scale_name,
                "Algorithm": label,
                "Max_Throughput_RPS": round(final_throughput, 2)
            }
            
        max_recorded_throughput = throughput
        current_rps += step

def main():
    print("=" * 65)
    print("=== Parallel Maximum Throughput Search Experiment ===")
    print("=" * 65)

    target_scales = ["1c_2n", "2c_3n", "10c_50n"]
    target_exps = [1, 2, 3, 4, 5]
    
    tasks = [(exp_id, scale) for scale in target_scales for exp_id in target_exps]
    results = []

    # 使用 5 核心平行處理，極速飆車！
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(search_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 確保輸出排序整齊 (依 Scale 與 Algorithm 排序)
    results.sort(key=lambda x: (target_scales.index(x["Scale"]), x["Algorithm"]))

    output_csv = "max_throughput_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Scale", "Algorithm", "Max_Throughput_RPS"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("\n" + "=" * 65)
    print(f"🎉 All tests finished in parallel! Results saved to {output_csv}")
    print("=" * 65)
    
    for row in results:
        print(f"[{row['Scale']:<10}] {row['Algorithm']:<15} -> {row['Max_Throughput_RPS']} req/s")

if __name__ == "__main__":
    main()
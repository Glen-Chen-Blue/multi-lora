#!/usr/bin/env python3
"""
CLI entry point for Synthetic Multi-LoRA Simulation Experiments.
Uses Poisson distribution for arrival times and Zipf distribution for LoRA selection.
(Multiprocessing Accelerated Version - 25 Workers with Penalty Bug Fixes)
"""

import os
import gc
import sys
import json
import pandas as pd
import concurrent.futures  
import contextlib          
import argparse  # 新增這個模組

# 1. 自動定位專案根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from discrete_sim.sim_types import SimulationConfig
from discrete_sim.simulation import Simulation
from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

# ==========================================
# 實驗參數設定區
# ==========================================
SIMULATION_HOURS = 24                
NUM_CLUSTERS = 3                     
COMPUTE_NODES_PER_CLUSTER = 5        

# 設定你這次要跑的 RPS 區間
RPS_LIST = [i for i in range(26, 40)]
ZIPF_S_PARAMETER = 2               

LORA_MAPPING_PATH = os.path.join(PROJECT_ROOT, "information", "lora_mapping.json")
OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
TRACE_CSV_DUMMY = os.path.join(PROJECT_ROOT, "information", "simulation_data.csv")
METADATA_DIR = os.path.join(PROJECT_ROOT, "information")

BASELINE_STRATEGIES = [
    (1, "Ours (SP1+SP2)"),
    (2, "Ours w/o Sem"),
    (3, "Ours w/o SP2"),
    (4, "dLoRA"),
    (5, "S-LoRA")
]
# 對齊論文中的 Penalty 設定 (0.001 Base * 60 Multiplier)
PENALTY_WEIGHT = 0.06 
# ==========================================

def extract_final_average_cost_with_penalty(log_path: str, penalty_weight: float) -> float:
    """🚨 [修復點 1]：不再依賴 cost2.py，直接精準計算包含 Drop Penalty 的真實平均成本"""
    if not os.path.exists(log_path):
        return 0.0
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if not lines: return 0.0
            totals = json.loads(lines[-1]).get("efo_totals", {})
            
            total_req = max(1, totals.get("total_requests", 1))
            total_inference_time_ms = totals.get("total_inference_time", 0)
            offloads = totals.get("total_offloads", 0)
            downloads = totals.get("artifact_downloads", 0)
            total_drops = totals.get("total_drops", 0)  # 抓取 Drop 數量
            
            compute_cost = total_inference_time_ms * 0.001 
            network_cost = (offloads * 0.001) + (downloads * 3.0)
            penalty_cost = total_drops * penalty_weight  # 將巨大罰金加算進來
            
            # 平均成本 = (計算 + 網路 + 違約罰金) / 總請求數
            avg_cost = (compute_cost + network_cost + penalty_cost) / total_req
            return avg_cost
    except Exception as e:
        print(f"Error parsing log: {e}")
        return 0.0

def run_single_task(args):
    """獨立的任務函式，用來跑單一 (RPS, Baseline) 組合的模擬"""
    rps, cluster_rps, exp_id, exp_name, topology, target_clusters, duration_hours = args
    
    print(f"[START] RPS: {rps:2d} | {exp_name}")
    
    out_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs")
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # 🚨 [修復點 2]：在 Worker 進程內部強制注入 Config，啟動 Drop 與 SLO 超時機制
    # =========================================================================
    import config
    import discrete_sim.sim_control_node_ as scn
    import discrete_sim.sim_compute_node as s_compute
    import discrete_sim.sim_efo as sefo

    config.ENABLE_DROP = True
    config.MAX_WAITING_TIME = 6.0       # 嚴格執行 6.0s SLO 超時
    config.COST_DROP_PENALTY = PENALTY_WEIGHT
    config.PSI_DROP = PENALTY_WEIGHT

    if hasattr(scn, 'PSI_DROP'): scn.PSI_DROP = PENALTY_WEIGHT
    if hasattr(sefo, 'COST_DROP_PENALTY'): sefo.COST_DROP_PENALTY = PENALTY_WEIGHT
    scn.T_MAX = 6.0
    scn.ENABLE_DROP = True

    # 🚨 [修復點 3]：移植 run_TTFT_rps.py 的 Patch，確保 Baseline 節點不會無限期休眠
    if not hasattr(s_compute.SimComputeNode, '_original_full_reset'):
        s_compute.SimComputeNode._original_full_reset = s_compute.SimComputeNode.full_reset
        def patched_full_reset(self):
            self._original_full_reset()
            # 除非是具備動態伸縮能力的完整版 Ours，否則其他 Baseline 強制火力全開
            if exp_id != 1:
                self.status = s_compute.NodeStatus.ACTIVE 
        s_compute.SimComputeNode.full_reset = patched_full_reset
    # =========================================================================

    sim_config = SimulationConfig(
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
    
    sim = Simulation(sim_config)

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

    # 修復：將生成的合成事件轉為 DataFrame 讓 EFO 預測
    records = []
    for t_ms, reqs in synthetic_gen._events.items():
        arr_sec = t_ms / 1000.0
        for cluster, lora_id in reqs:
            records.append({"arrival_sec": arr_sec, "cluster": cluster, "lora_id": lora_id})
    
    sim.efo.simulation_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["arrival_sec", "cluster", "lora_id"])

    # 執行模擬
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    # 擷取真實成本 (包含 Drop Penalty)
    log_file_path = os.path.join(out_dir, "efo_global_metrics.log")
    avg_cost = extract_final_average_cost_with_penalty(log_file_path, PENALTY_WEIGHT)
    
    print(f"[DONE ] RPS: {rps:2d} | {exp_name:<15} -> Avg Cost: NT${avg_cost:.4f}")
    
    del sim, sim_config, synthetic_gen 
    gc.collect()

    return {"Global_RPS": rps, "Strategy": exp_name, "Average_Cost": avg_cost}


def rebuild_csv_from_logs():
    """直接從既有的 log 資料夾重建 CSV，不重新執行模擬"""
    base_dir = os.path.join(PROJECT_ROOT, "results", "synthetic")
    if not os.path.exists(base_dir):
        print(f"[Error] 找不到結果目錄: {base_dir}")
        return

    results_data = []
    strategy_dict = dict(BASELINE_STRATEGIES)

    print("=" * 65)
    print(f"🔍 開始掃描 {base_dir} 下的所有 Log 並重建 CSV...")
    print("=" * 65)

    # 掃描 RPS_* 資料夾
    for rps_folder in os.listdir(base_dir):
        if not rps_folder.startswith("RPS_"):
            continue
        try:
            rps = int(rps_folder.split("_")[1])
        except ValueError:
            continue

        rps_path = os.path.join(base_dir, rps_folder)

        # 掃描 Exp_*_logs 資料夾
        for exp_folder in os.listdir(rps_path):
            if not exp_folder.startswith("Exp_") or not exp_folder.endswith("_logs"):
                continue
            try:
                exp_id = int(exp_folder.split("_")[1])
            except ValueError:
                continue

            strategy_name = strategy_dict.get(exp_id, f"Unknown_{exp_id}")
            log_path = os.path.join(rps_path, exp_folder, "efo_global_metrics.log")

            if os.path.exists(log_path):
                avg_cost = extract_final_average_cost_with_penalty(log_path, PENALTY_WEIGHT)
                # 只有當 log 真的有解析出數字才記錄 (過濾掉空檔或失敗的 log)
                if avg_cost > 0:
                    results_data.append({
                        "Global_RPS": rps,
                        "Strategy": strategy_name,
                        "Average_Cost": avg_cost
                    })
                    print(f"✅ 讀取成功: RPS={rps:2d} | {strategy_name:<15} -> Avg Cost: NT${avg_cost:.4f}")
                else:
                    print(f"⚠️ 解析失敗或無效: {log_path}")

    if not results_data:
        print("⚠️ 沒有找到任何有效的 log 資料可以重建 CSV。")
        return

    # 轉換成 DataFrame 並排序
    df_new = pd.DataFrame(results_data)
    
    strategy_order = [s[1] for s in BASELINE_STRATEGIES]
    # 確保未知策略也能被處理不報錯
    categories = strategy_order + [s for s in df_new['Strategy'].unique() if s not in strategy_order]
    
    df_new['Strategy'] = pd.Categorical(df_new['Strategy'], categories=categories, ordered=True)
    df_new = df_new.sort_values(by=['Global_RPS', 'Strategy'])
    
    # 儲存
    df_new.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n" + "=" * 65)
    print(f"🎉 重建 CSV 完成！已從歷史資料收集了 {len(results_data)} 筆紀錄並寫入 {OUTPUT_CSV_FILE}")
    print("=" * 65)



def main():
    # 加入命令列參數解析
    parser = argparse.ArgumentParser(description="Synthetic Multi-LoRA Simulation Experiments")
    parser.add_argument("--rebuild-only", action="store_true", help="只掃描既有 log 資料夾重建 CSV，不執行新的模擬")
    args = parser.parse_args()

    if args.rebuild_only:
        rebuild_csv_from_logs()
        return

    # ================= 原本的模擬邏輯 =================
    duration_hours = SIMULATION_HOURS
    topology = {f"cluster_{i}": COMPUTE_NODES_PER_CLUSTER for i in range(1, NUM_CLUSTERS + 1)}
    target_clusters = list(topology.keys())
    
    tasks = []
    for rps in RPS_LIST:
        cluster_rps = rps / NUM_CLUSTERS
        for exp_id, exp_name in BASELINE_STRATEGIES:
            # 你也可以在這裡多加一層檢查，如果 log 已經存在就 continue 跳過，
            # 這樣連執行期的時間都省了 (Optional)
            log_path = os.path.join(PROJECT_ROOT, "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs", "efo_global_metrics.log")
            if os.path.exists(log_path):
                 print(f"⏭️  [SKIP] RPS: {rps:2d} | {exp_name} 已經跑過，跳過。")
                 continue
                 
            tasks.append((rps, cluster_rps, exp_id, exp_name, topology, target_clusters, duration_hours))

    if not tasks:
        print("所有設定在 RPS_LIST 中的任務都已經有對應的 Log 了，沒有需要執行的新模擬。")
        print("如果需要彙整 CSV，請執行: python script.py --rebuild-only")
        return

    print("=" * 65)
    print("🚀 Starting Parallel Synthetic Experiments (Average Cost vs RPS - PENALTY FIXED)")
    print("=" * 65)

    results_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(run_single_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results_data.append(result)
            except Exception as exc:
                print(f"[Error] 某個實驗執行時發生錯誤: {exc}")

    # (原本的 DataFrame 處理與儲存 CSV 邏輯保持不變...)
    df_new = pd.DataFrame(results_data)
    if not df_new.empty:
        if os.path.exists(OUTPUT_CSV_FILE):
            df_old = pd.read_csv(OUTPUT_CSV_FILE)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Global_RPS', 'Strategy'], keep='last')
        else:
            df_combined = df_new
            
        strategy_order = [s[1] for s in BASELINE_STRATEGIES]
        df_combined['Strategy'] = pd.Categorical(df_combined['Strategy'], categories=strategy_order, ordered=True)
        df_combined = df_combined.sort_values(by=['Global_RPS', 'Strategy'])
        df_combined.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n" + "=" * 65)
    print(f"🎉 All parallel experiments finished! Data safely saved to {OUTPUT_CSV_FILE}")
    print("=" * 65)



if __name__ == "__main__":
    main()
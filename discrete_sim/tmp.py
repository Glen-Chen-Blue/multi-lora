#!/usr/bin/env python3
"""
從現有的 log 檔案中提取 Average Cost 並重建 synthetic_results.csv，
無需重新執行模擬。
"""

import os
import sys
import pandas as pd

# 1. 自動定位專案根目錄 (保持與原本腳本相同的邏輯)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 2. 匯入必要的模組 (只需 cost2)
try:
    from cost2 import parse_logs
except ImportError:
    print("[Error] 無法匯入 cost2 模組，請確認您在正確的目錄下執行此腳本。")
    sys.exit(1)

# ==========================================
# 實驗參數設定區 (必須與原本跑實驗的設定一致)
# ==========================================
RPS_LIST = [i for i in range(1, 51)] 

BASELINE_STRATEGIES = [
    (1, "Experiment 1 (SP1+SP2)"),
    (2, "Experiment 2 (SP1+SP2 w/o semantic)"),
    (3, "Experiment 3 (SP1+Random)"),
    # (4, "Experiment 4 (LRU+Random)"),
    (4, "Experiment 4 (Dlora)"),
    (5, "Experiment 5 (Slora)")
]

OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
# ==========================================

def extract_final_average_cost(log_path: str) -> float:
    """使用 cost2.py 精準計算並提取最後一刻的 average cost"""
    if not os.path.exists(log_path):
        return None  # 改回 None 方便辨識缺失的資料
    try:
        df = parse_logs(log_path)
        if df.empty:
            return None
        return float(df['cost_per_request'].iloc[-1])
    except Exception as e:
        print(f"  [Warning] 解析 {log_path} 時發生錯誤: {e}")
        return None

def main():
    print("=" * 65)
    print("🔍 開始掃描現有的 Log 檔案並重建 CSV...")
    print("=" * 65)

    results_data = []
    missing_logs = 0

    for rps in RPS_LIST:
        for exp_id, exp_name in BASELINE_STRATEGIES:
            # 重建原本的資料夾路徑邏輯
            log_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs")
            log_file_path = os.path.join(log_dir, "efo_global_metrics.log")
            
            print(f"正在讀取 RPS: {rps:2d} | {exp_name} ...", end=" ")
            
            avg_cost = extract_final_average_cost(log_file_path)
            
            if avg_cost is not None:
                print(f"成功 (Avg Cost: {avg_cost:.4f})")
                results_data.append({
                    "Global_RPS": rps,
                    "Strategy": exp_name,
                    "Average_Cost": avg_cost
                })
            else:
                print("找不到檔案或無數據 (跳過)")
                missing_logs += 1

    # 將收集到的數據轉換成 DataFrame
    df_recovered = pd.DataFrame(results_data)
    
    if df_recovered.empty:
        print("\n❌ 找不到任何有效的 log 檔案，請確認您的 `results/synthetic/` 目錄是否存在且包含資料。")
        return

    # 依照 RPS 和策略排序，讓 CSV 乾淨整齊
    df_recovered = df_recovered.sort_values(by=['Global_RPS', 'Strategy'])
    
    # 寫入 CSV
    df_recovered.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n" + "=" * 65)
    print(f"🎉 復原完成！已從 {len(results_data)} 個實驗紀錄中提取數據。")
    if missing_logs > 0:
        print(f"⚠️ 注意：有 {missing_logs} 筆預期的實驗資料遺失或讀取失敗。")
    print(f"💾 資料已安全儲存至: {OUTPUT_CSV_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    main()
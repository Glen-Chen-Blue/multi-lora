#!/usr/bin/env python3
"""
只讀取現有的 Log 檔案，使用 cost2.py 重新計算成本，並輸出 synthetic_results.csv。
不需要重新執行模擬！
"""

import os
import sys
import pandas as pd

# 自動定位專案根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 載入你原本精準的算錢邏輯
from cost2 import parse_logs

# 設定與你剛才跑的模擬完全一致
RPS_LIST = [1]
BASELINE_STRATEGIES = [
    (1, "Experiment 1 (SP1+SP2)"),
    (2, "Experiment 2 (SP1+SP2 w/o semantic)"),
    (3, "Experiment 3 (SP1+Random)"),
    (4, "Experiment 4 (LRU+Random)"),
    (5, "Experiment 5 (Dlora)"),
    (6, "Experiment 6 (Slora)")
]

OUTPUT_CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")

def main():
    results_data = []
    print("=" * 65)
    print("📊 正在從現有日誌中提取並計算 Cost... (不需要重跑模擬)")
    print("=" * 65)

    for rps in RPS_LIST:
        for exp_id, exp_name in BASELINE_STRATEGIES:
            # 尋找你剛才輸出的 log 檔案
            # 為了避免你當時是在不同目錄執行的，這裡給予多個候選路徑自動尋找
            path1 = os.path.join(".", "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs", "efo_global_metrics.log")
            path2 = os.path.join(PROJECT_ROOT, "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs", "efo_global_metrics.log")
            path3 = os.path.join(PROJECT_ROOT, "discrete_sim", "results", "synthetic", f"RPS_{rps}", f"Exp_{exp_id}_logs", "efo_global_metrics.log")
            
            log_path = None
            if os.path.exists(path1):
                log_path = path1
            elif os.path.exists(path2):
                log_path = path2
            elif os.path.exists(path3):
                log_path = path3
                
            if log_path is None:
                print(f"[Warning] 找不到日誌: RPS {rps}, Exp {exp_id}。跳過...")
                continue

            try:
                # 呼叫 cost2.py 把 log 轉成 DataFrame 並乘上金額費率
                df = parse_logs(log_path)
                
                if df.empty:
                    print(f"[Warning] 日誌為空: RPS {rps}, Exp {exp_id}")
                    avg_cost = 0.0
                else:
                    # 抓取最後一個時間點的平均成本
                    avg_cost = float(df['cost_per_request'].iloc[-1])
                
                print(f"✅ 成功提取: {exp_name} @ {rps} RPS -> Final Avg Cost: {avg_cost:.4f}")
                
                results_data.append({
                    "Global_RPS": rps,
                    "Strategy": exp_name,
                    "Average_Cost": avg_cost
                })
            except Exception as e:
                print(f"[Error] 解析失敗 {log_path}: {e}")

    # 將所有結果存成 CSV
    if results_data:
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(OUTPUT_CSV_FILE, index=False)
        print("\n" + "=" * 65)
        print(f"🎉 所有數據提取完成！已重新覆寫：{OUTPUT_CSV_FILE}")
        print("👉 現在你可以直接執行 python draw_synthetic_cost.py 來畫圖了！")
        print("=" * 65)
    else:
        print("\n[Error] 沒有提取到任何數據，請確認你的 logs 檔案真的有產生。")

if __name__ == "__main__":
    main()
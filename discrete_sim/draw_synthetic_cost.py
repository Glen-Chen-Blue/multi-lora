#!/usr/bin/env python3
"""
繪製 RPS 對比 Average Cost 的折線圖。
讀取 run_synthetic_experiments.py 產生的 synthetic_results.csv。
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 自動定位專案根目錄 (往上推一層，和 run_synthetic_experiments.py 保持一致)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 使用絕對路徑
CSV_FILE = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
OUTPUT_IMAGE = os.path.join(PROJECT_ROOT, "rps_vs_cost.png")

def main():
    # 檢查數據檔案是否存在
    if not os.path.exists(CSV_FILE):
        print(f"[Error] 找不到數據檔案：{CSV_FILE}")
        print("請確認你已經成功執行完 run_synthetic_experiments.py！")
        return

    print(f"正在讀取 {CSV_FILE} 並開始繪圖...")
    
    # 讀取數據
    df = pd.read_csv(CSV_FILE)

    # 建立畫布：設定適當的大小與高解析度
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 定義不同策略的標記 (Markers)
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 取得所有的策略名稱
    strategies = df['Strategy'].unique()

    # 依序為每種策略畫線
    for idx, strategy in enumerate(strategies):
        subset = df[df['Strategy'] == strategy]
        subset = subset.sort_values(by='Global_RPS')
        
        plt.plot(
            subset['Global_RPS'], 
            subset['Average_Cost'], 
            # marker=markers[idx % len(markers)], 
            linewidth=2, 
            markersize=8,
            label=strategy
        )

    # 設定圖表標題與座標軸標籤
    plt.title('System Load vs Average Cost (3 Days Simulation)', fontsize=16, fontweight='bold')
    plt.xlabel('Global Requests Per Second (RPS)', fontsize=14)
    plt.ylabel('Average Cost per Request', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()

    # 儲存圖片
    plt.savefig(OUTPUT_IMAGE)
    print(f"✅ 圖表已成功繪製，並儲存為：{OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
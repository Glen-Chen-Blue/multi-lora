import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_throughput_barchart(csv_path="max_throughput_results.csv", output_path="max_throughput_barchart.png"):
    # 1. 讀取 CSV 資料
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到檔案 {csv_path}，請確認是否已經執行過壓力測試腳本。")
        return

    # 2. 定義順序 (可依據論文排版需求自由調整)
    # X 軸的演算法大群組順序
    algorithms_order = ["S-LoRA", "dLoRA", "Ours w/o SP2", "Ours w/o Sem", "Ours (SP1+SP2)"]
    # 每個大群組內部的 Bar 順序 (從小規模到大規模)
    scales_order = ["1c_2n", "2c_3n", "10c_50n"]
    
    # 圖例的顯示名稱
    scale_labels = {
        "1c_2n": "1 Cluster (2 Nodes)",
        "2c_3n": "2 Clusters (3 Nodes)",
        "10c_50n": "10 Clusters (50 Nodes)"
    }

    # 3. 資料樞紐分析 (Pivot)
    # 將資料轉換為列是 Algorithm，欄是 Scale 的表格
    pivot_df = df.pivot(index="Algorithm", columns="Scale", values="Max_Throughput_RPS")
    
    # 依照指定的順序重新排列
    pivot_df = pivot_df.reindex(index=algorithms_order, columns=scales_order)

    # 4. 開始繪圖設定
    x = np.arange(len(algorithms_order))  # 大群組的 X 軸基準位置
    width = 0.25  # 每個 Bar 的寬度

    fig, ax = plt.subplots(figsize=(12, 7))

    # 依序畫出三個規模的長條圖
    # Bar 1: 1c_2n (偏左)
    bars1 = ax.bar(x - width, pivot_df["1c_2n"], width, label=scale_labels["1c_2n"], color="#4C72B0", edgecolor="black")
    # Bar 2: 2c_3n (置中)
    bars2 = ax.bar(x, pivot_df["2c_3n"], width, label=scale_labels["2c_3n"], color="#DD8452", edgecolor="black")
    # Bar 3: 10c_50n (偏右)
    bars3 = ax.bar(x + width, pivot_df["10c_50n"], width, label=scale_labels["10c_50n"], color="#55A868", edgecolor="black")

    # 5. 美化與標籤設定
    ax.set_ylabel('Max Steady-state Throughput (Req/s)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scheduling & Deployment Methods', fontsize=14, fontweight='bold')
    # ax.set_title('Maximum Throughput Comparison Across System Scales', fontsize=16, fontweight='bold', pad=15)
    
    # 設定 X 軸的刻度與文字
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms_order, fontsize=12, rotation=15)
    
    # 加入 Y 軸網格線幫助對齊數值
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # 讓網格線在長條圖後方
    
    # 加入圖例
    ax.legend(title="Federation Scale", fontsize=11, title_fontsize=12, loc='upper left')

    # 6. 在每個 Bar 上方加上具體數值 (選用，可讓圖表更精確)
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 垂直向上偏移 3 點
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # 7. 儲存與顯示
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 長條圖已經繪製完成並儲存為: {output_path}")

if __name__ == "__main__":
    draw_throughput_barchart()
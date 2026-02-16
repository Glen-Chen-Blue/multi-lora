import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_from_data(json_file="experiment_data.json", output_image="replot_cost_vs_rps_with_limit.png"):
    # 1. 讀取數據
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file}' not found. Please ensure the JSON file exists.")
        return

    rps_steps = data["rps_steps"]
    results = data["results"]
    scenarios_config = data["scenarios_config"]

    # ==========================================
    # 設定臨界點 (RPS)
    # ==========================================
    # 這裡定義哪條線在那個 RPS 之後開始不準
    reliability_limits = {
        "Smart Mechanism (Ours)": 14,
        "No Semantic (Baseline 1)": 10,
        "Random/Full (Baseline 2)": 10
    }

    # 2. 開始繪圖
    print(f"📊 Plotting data from {json_file}...")
    plt.figure(figsize=(10, 6))

    for name, costs in results.items():
        # 讀取設定
        cfg = scenarios_config.get(name, {"color": "blue", "marker": "o"})
        
        # 畫原本的折線圖
        plt.plot(
            rps_steps, 
            costs, 
            marker=cfg.get("marker", "o"), 
            label=name, 
            color=cfg.get("color", "blue"), 
            linewidth=2
        )

        # 畫臨界點標記
        if name in reliability_limits:
            limit_rps = reliability_limits[name]
            
            # 確保這個 RPS 存在於數據中，才能抓到對應的 Cost
            if limit_rps in rps_steps:
                idx = rps_steps.index(limit_rps)
                limit_cost = costs[idx]
                
                # 在該點畫一個明顯的標記 (例如黑色的 X)
                plt.scatter(
                    [limit_rps], 
                    [limit_cost], 
                    color="black", 
                    marker="X", 
                    s=150,       # 大小
                    zorder=10    # 確保蓋在線上面
                )

    plt.title("Resource Cost vs Load (Computing Time Only)", fontsize=14)
    plt.xlabel("Request Rate (RPS)", fontsize=12)
    plt.ylabel("Cost (Busy Node-Seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # ==========================================
    # 處理圖例 (Legend)
    # ==========================================
    # 取得目前圖上已有的圖例 (三條線)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 手動增加一個圖例項目來說明 "X" 的意義
    limit_legend_item = Line2D(
        [0], [0], 
        color='w', 
        marker='X', 
        markerfacecolor='black', 
        markeredgecolor='black', 
        markersize=10, 
        label='Unable to Handle All Requests' # 這裡寫出意義
    )
    handles.append(limit_legend_item)

    # 顯示整合後的圖例
    plt.legend(handles=handles, loc="upper left")

    # 3. 儲存圖片
    plt.savefig(output_image)
    print(f"✅ Plot saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    plot_from_data()
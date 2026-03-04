import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation_requests(
    csv_path="./information/simulation_data.csv",
    target_clusters=["cluster_1"],
    speed_rate=1.0,
    start_offset_days=2,
    duration_hours=12,
    bin_minutes=10,
    output_filename="request_rate_plot.png"
):
    """
    讀取 CSV 並繪製 Request Rate (req/s) vs Time。
    
    Args:
        csv_path: CSV 檔案路徑
        target_clusters: 要繪製的 Cluster 列表
        speed_rate: 模擬加速倍率 (例如 2.0 代表把 24 小時的量壓縮進 12 小時)
        start_offset_days: 起始天數偏移 (預設 2 天 = 86400*2)
        duration_hours: 圖表要呈現的總長度 (模擬時間，預設 12 小時)
        bin_minutes: 統計顆粒度 (預設 10 分鐘)
    """
    
    # 1. 檢查檔案是否存在
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return

    print(f"📥 Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. 參數設定
    START_OFFSET = 86400 * start_offset_days
    BIN_SECONDS = bin_minutes * 60
    PLOT_DURATION_SECONDS = duration_hours * 3600
    
    # 為了填滿圖表的 duration_hours，我們需要從原始資料中讀取的時間長度
    # 例如 speed_rate=2, 圖表 12 小時，實際需要讀取 24 小時的資料
    REQUIRED_RAW_DURATION = PLOT_DURATION_SECONDS * speed_rate
    
    RAW_END_TIME = START_OFFSET + REQUIRED_RAW_DURATION

    # 3. 預先過濾資料 (只取需要的時間段)
    # 轉換時間欄位 (假設 CSV 欄位名稱是 'arrive_timestamp' 或 'arrival_sec')
    if 'arrive_timestamp' in df.columns:
        time_col = 'arrive_timestamp'
    elif 'arrival_sec' in df.columns:
        time_col = 'arrival_sec'
    else:
        print("❌ Error: Column 'arrive_timestamp' not found.")
        return

    df[time_col] = df[time_col].astype(float)
    
    # 過濾時間範圍
    mask = (df[time_col] >= START_OFFSET) & (df[time_col] < RAW_END_TIME)
    df_filtered = df[mask].copy()
    
    # 4. 計算模擬時間 (Simulation Time)
    # 公式: SimTime = (RealTime - StartOffset) / SpeedRate
    df_filtered['rel_time'] = df_filtered[time_col] - START_OFFSET
    df_filtered['sim_time'] = df_filtered['rel_time'] / speed_rate

    # 5. 準備繪圖
    plt.figure(figsize=(12, 6))
    
    # 建立時間區間 (Bins)
    # 從 0 到 PLOT_DURATION_SECONDS，步長為 BIN_SECONDS
    bins = np.arange(0, PLOT_DURATION_SECONDS + BIN_SECONDS, BIN_SECONDS)
    # 用於 X 軸標籤 (小時)
    bin_centers_hours = (bins[:-1] + BIN_SECONDS/2) / 3600 

    # 6. 針對每個 Cluster 處理並畫圖
    for cluster in target_clusters:
        # 取出該 Cluster 的資料
        cluster_df = df_filtered[df_filtered['cluster'] == cluster].copy()
        
        if cluster_df.empty:
            print(f"⚠️ Warning: No data found for {cluster} in the specified time range.")
            continue

        # 進行分組統計 (Cut & Groupby)
        # 這裡將 sim_time 切入我們定義好的 bins
        cluster_df['bin_idx'] = pd.cut(cluster_df['sim_time'], bins=bins, labels=False, include_lowest=True)
        
        # 統計每個 bin 的數量
        counts = cluster_df.groupby('bin_idx').size()
        
        # 重新索引以補零 (Reindex to ensure all time slots are present, even if 0 requests)
        # 完整的 index 應該是 0 到 len(bins)-2
        full_index = np.arange(len(bins) - 1)
        counts = counts.reindex(full_index, fill_value=0)
        
        # 計算 Request/s
        # Rate = Count / Bin_Duration
        rates = counts / BIN_SECONDS
        
        # 畫線
        plt.plot(bin_centers_hours, rates, label=f"{cluster}", linewidth=2)

    # 7. 圖表修飾
    plt.title(f"Request Rate over Time (Start: Day {start_offset_days}, Speed: {speed_rate}x)", fontsize=14)
    plt.xlabel("Simulation Time (Hours)", fontsize=12)
    plt.ylabel("Request Rate (req/s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xlim(0, duration_hours)
    plt.ylim(0, 10)
    
    # 顯示 Bin 資訊作為註解
    info_text = (f"Bin Size: {bin_minutes} min (Plot Time)\n"
                 f"Source Data Window: {bin_minutes * speed_rate:.1f} min")
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # 存檔或顯示
    output_filename = output_filename or "request_rate_plot.png"
    plt.savefig(output_filename)
    print(f"✅ Plot saved to {output_filename}")
    # plt.show() # 如果是在 Notebook 環境可開啟

# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    # 使用範例設定
    plot_simulation_requests(
        csv_path="./simulation_data.csv",
        target_clusters=["cluster_1", "cluster_2", "cluster_3"], # 可以加入更多 Cluster
        speed_rate=2.0,       # 2倍速 (24小時資料壓縮成12小時)
        start_offset_days=2,  # 從第 2 天開始 (86400*2)
        duration_hours=8,    # 畫 12 小時
        bin_minutes=5,        # 每 10 分鐘一點
        output_filename = "request_rate_plot_speed2.png"
    )
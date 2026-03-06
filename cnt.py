import pandas as pd
import json
import os
from config import SP1_INTERVAL_SECONDS

# 設定與原模擬器一致
TRACE_CSV = "./information/simulation_data.csv"
START_OFFSET = 86400 * 2
RUN_DURATION = SP1_INTERVAL_SECONDS * 8

def _load_json_env(name: str, default_json: str):
    raw = os.getenv(name, "").strip()
    if not raw:
        return json.loads(default_json)
    try:
        return json.loads(raw)
    except Exception as e:
        return json.loads(default_json)

# 取得目標 Cluster 名單
TARGET_CLUSTERS = _load_json_env("TARGET_CLUSTERS", '["cluster_1", "cluster_2", "cluster_3"]')

def count_simulation_requests():
    if not os.path.exists(TRACE_CSV):
        print(f"❌ 找不到檔案: {TRACE_CSV}")
        return

    print(f"📊 正在讀取並統計請求數量 (Offset: {START_OFFSET}s, Duration: {RUN_DURATION}s)...")
    
    # 讀取資料
    df = pd.read_csv(TRACE_CSV)

    # 1. 轉換時間與過濾區間
    df["arrival_sec"] = df["arrive_timestamp"].astype(float)
    df = df[(df["arrival_sec"] >= START_OFFSET) & 
            (df["arrival_sec"] <= START_OFFSET + RUN_DURATION)].copy()

    # 2. 過濾 Cluster
    df = df[df["cluster"].isin(TARGET_CLUSTERS)]

    # 3. 格式化 LoRA ID (補上 LoRA_ 字頭以利閱讀)
    df["lora_name"] = df["lora_id"].apply(lambda x: f"LoRA_{int(x)}")

    # 4. 進行群組統計
    # 以 Cluster 和 LoRA 分組，計算每一組的數量
    stats = df.groupby(['cluster', 'lora_name']).size().reset_index(name='count')

    # 5. 排序並列印結果
    stats = stats.sort_values(by=['cluster', 'count'], ascending=[True, False])

    print("\n" + "="*40)
    print(f"{'Cluster':<15} | {'LoRA ID':<10} | {'Count':<6}")
    print("-" * 40)
    
    total_count = 0
    for _, row in stats.iterrows():
        print(f"{row['cluster']:<15} | {row['lora_name']:<10} | {row['count']:<6}")
        total_count += row['count']
    
    print("-" * 40)
    print(f"{'TOTAL':<15} | {'':<10} | {total_count:<6}")
    print("="*40)

    # 額外統計：各 Cluster 的總量
    print("\n各 Cluster 總計:")
    cluster_totals = df.groupby('cluster').size()
    for cluster, count in cluster_totals.items():
        print(f"- {cluster}: {count}")

if __name__ == "__main__":
    count_requests = count_simulation_requests()
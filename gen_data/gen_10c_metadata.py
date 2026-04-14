#!/usr/bin/env python3
"""
Generate Metadata and Mapping for 10-Cluster Experiments.
(100 LoRAs: 70 Global, 30 Local -> 3 Local per cluster)
"""

import json
import os
import math
import random

def generate_10c_metadata():
    # 實驗設定 (參照論文 Section VI-B: Workload Description)
    NUM_CLUSTERS = 10
    NUM_GLOBAL_LORAS = 70
    LOCAL_LORAS_PER_CLUSTER = 3
    TOTAL_LORAS = NUM_GLOBAL_LORAS + (NUM_CLUSTERS * LOCAL_LORAS_PER_CLUSTER)  # 100

    # 確保輸出目錄存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "../information")
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "lora_metadata_10c.json")
    mapping_path = os.path.join(output_dir, "lora_mapping_10c.json")

    # 固定隨機種子，確保每次生成的 10c 測試資料一致
    random.seed(42)

    lora_metadata = {}
    
    # 1. 建立 70 個 Global LoRAs
    for i in range(1, NUM_GLOBAL_LORAS + 1):
        lora_id = f"LoRA_{i}"
        lora_metadata[lora_id] = {
            "type": "Global",
            "cluster": None,
            "embedding": [random.uniform(0, 1), random.uniform(0, 1)] # 2D 語意向量
        }

    # 2. 建立 30 個 Local LoRAs (每個 Cluster 分配 3 個)
    current_lora_idx = NUM_GLOBAL_LORAS + 1
    for c_idx in range(1, NUM_CLUSTERS + 1):
        cluster_name = f"cluster_{c_idx}"
        for _ in range(LOCAL_LORAS_PER_CLUSTER):
            lora_id = f"LoRA_{current_lora_idx}"
            lora_metadata[lora_id] = {
                "type": "Local",
                "cluster": cluster_name,
                "embedding": [random.uniform(0, 1), random.uniform(0, 1)]
            }
            current_lora_idx += 1

    # 3. 為每個 Cluster 定義一個「需求中心 (Demand Center)」
    # 需求中心決定了該叢集對哪些模型的偏好較高
    cluster_centers = {}
    for c_idx in range(1, NUM_CLUSTERS + 1):
        cluster_name = f"cluster_{c_idx}"
        cluster_centers[cluster_name] = [random.uniform(0, 1), random.uniform(0, 1)]

    # 4. 計算每個 Cluster 對所有 100 個 LoRA 的距離，並生成 Mapping (Rank)
    lora_mapping = {}
    
    for cluster_name, center in cluster_centers.items():
        distances = []
        for lora_id, meta in lora_metadata.items():
            emb = meta["embedding"]
            # 計算 Euclidean Distance (語意距離)
            dist = math.sqrt((center[0] - emb[0])**2 + (center[1] - emb[1])**2)
            distances.append((lora_id, dist))
        
        # 依照距離由小到大排序 (距離越近，熱度排名越前面)
        distances.sort(key=lambda x: x[1])
        
        # 建立排名字典 { "LoRA_X": rank, ... } (rank 從 1 到 100)
        cluster_ranking = {}
        for rank, (lora_id, dist) in enumerate(distances, start=1):
            cluster_ranking[lora_id] = str(rank) # 轉為字串以相容舊版格式
            
        lora_mapping[cluster_name] = cluster_ranking

    # 5. 將結果寫入 JSON 檔案
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(lora_metadata, f, indent=4)
        
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(lora_mapping, f, indent=4)

    print(f"✅ Successfully generated {TOTAL_LORAS} LoRAs for {NUM_CLUSTERS} clusters.")
    print(f"📦 Metadata saved to: {metadata_path}")
    print(f"📈 Mapping saved to : {mapping_path}")

if __name__ == "__main__":
    generate_10c_metadata()
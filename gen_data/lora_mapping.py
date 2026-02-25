import json
import math

def generate_mapping():
    # 1. 讀取 metadata
    print("📥 讀取 lora_metadata.json ...")
    with open("lora_metadata.json", "r", encoding="utf-8") as f:
        lora_metadata = json.load(f)

    # 準備每個 Cluster 擁有的 LoRA 列表 (70 global + 10 local)
    cluster_loras = {
        "cluster_1": [],
        "cluster_2": [],
        "cluster_3": []
    }

    # 2. 分類 LoRA
    for lora_id, info in lora_metadata.items():
        if info["type"] == "global":
            # Global LoRA 屬於所有 Cluster
            cluster_loras["cluster_1"].append(lora_id)
            cluster_loras["cluster_2"].append(lora_id)
            cluster_loras["cluster_3"].append(lora_id)
        elif info["type"] == "local":
            # Local LoRA 只屬於特定 Cluster
            c_name = info["cluster"]
            cluster_loras[c_name].append(lora_id)

    # 3. 找出每個 Cluster 的「中心點」
    # 這裡我們直接取該 Cluster 的第一個 Local LoRA 作為中心 (即生成時的 Seed)
    centers = {}
    for c_name in cluster_loras.keys():
        # 找出屬於該 cluster 的所有 local LoRA 中的第一個
        local_lora_id = next(l for l in cluster_loras[c_name] if lora_metadata[l]["type"] == "local")
        centers[c_name] = lora_metadata[local_lora_id]["pos"]
        print(f"🎯 {c_name} 的中心點為 {local_lora_id}，座標: {centers[c_name]}")

    # 4. 準備 Azure ID 的分配表 (1~240 依序發牌給 3 個 Cluster)
    # 數字越小代表用量越大 (來自 Azure 的特徵)
    azure_ids = {
        "cluster_1": [str(i) for i in range(1, 241, 3)],  # [1, 4, 7, ..., 238]
        "cluster_2": [str(i) for i in range(2, 241, 3)],  # [2, 5, 8, ..., 239]
        "cluster_3": [str(i) for i in range(3, 241, 3)]   # [3, 6, 9, ..., 240]
    }

    mapping = {}

    def calc_dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # 5. 依照距離分配 ID
    for c_name in cluster_loras.keys():
        center_pos = centers[c_name]
        loras_in_c = cluster_loras[c_name]
        
        # 計算距離並由近到遠排序 (距離中心越近，排越前面)
        loras_sorted = sorted(loras_in_c, key=lambda l_id: calc_dist(center_pos, lora_metadata[l_id]["pos"]))
        
        # 將排序好的 Azure ID 依序配對給排序好的 LoRA
        # 排最前面的 (最靠近中心) 會拿到數字最小的 ID (最高用量)
        mapping[c_name] = {}
        for i, l_id in enumerate(loras_sorted):
            mapping[c_name][l_id] = azure_ids[c_name][i]

    # 6. 輸出 JSON 檔案
    output_file = "lora_mapping.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    print(f"🎉 Mapping 建立完成！已儲存至 {output_file}")
    
    # 稍微印出前三個檢查一下
    print("\n🧐 檢查 Cluster 1 前三個用量最大的 LoRA:")
    for l_id, a_id in list(mapping["cluster_1"].items())[:3]:
        print(f"  - {l_id} 距離中心: {calc_dist(centers['cluster_1'], lora_metadata[l_id]['pos']):.3f} -> 配對 Azure ID: {a_id}")

if __name__ == "__main__":
    generate_mapping()
import json
import random
import math
import matplotlib.pyplot as plt

def generate_local_points(num_points, seed, distance_threshold=0.1, max_subs=5):
    """
    生成 Local LoRAs，並從指定的種子點 (seed) 開始生長
    """
    points = [seed]

    for i in range(1, num_points):
        placed = False
        for attempt in range(2000): 
            degrees = [0] * len(points)
            for p_i in range(len(points)):
                for p_j in range(p_i + 1, len(points)):
                    if math.hypot(points[p_i][0]-points[p_j][0], points[p_i][1]-points[p_j][1]) <= distance_threshold:
                        degrees[p_i] += 1
                        degrees[p_j] += 1

            candidates = []
            for idx, deg in enumerate(degrees):
                if deg == 0: candidates.extend([idx] * 20)
                elif deg == 1: candidates.extend([idx] * 15)
                elif deg == 2: candidates.extend([idx] * 8)
                elif deg == 3: candidates.extend([idx] * 2)
                elif deg == 4: candidates.extend([idx] * 1)

            if not candidates: break
            parent = points[random.choice(candidates)]

            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.065, 0.099)
            new_x, new_y = round(parent[0] + r * math.cos(angle), 3), round(parent[1] + r * math.sin(angle), 3)
            new_p = (new_x, new_y)

            new_degree = 0
            valid = True
            for idx, existing_p in enumerate(points):
                if math.hypot(new_p[0] - existing_p[0], new_p[1] - existing_p[1]) <= distance_threshold:
                    new_degree += 1
                    if degrees[idx] >= max_subs:
                        valid = False
                        break
            
            if valid and 1 <= new_degree <= 3:
                points.append(new_p)
                placed = True
                break
                
        if not placed:
            points.append((round(new_x + 0.1, 3), round(new_y + 0.1, 3)))

    return points

def generate_global_points(num_points, seed, all_local_points, distance_threshold=0.1, max_subs=5, max_global_near_local=4):
    """
    生成 Global LoRAs，並強制限制 Local LoRAs 附近不能有太多的 Global 點
    """
    points = [seed]
    # 紀錄每個 Local 點目前附近有幾個 Global 點
    local_global_counts = [0] * len(all_local_points)

    for i in range(1, num_points):
        placed = False
        for attempt in range(2000): 
            degrees = [0] * len(points)
            for p_i in range(len(points)):
                for p_j in range(p_i + 1, len(points)):
                    if math.hypot(points[p_i][0]-points[p_j][0], points[p_i][1]-points[p_j][1]) <= distance_threshold:
                        degrees[p_i] += 1
                        degrees[p_j] += 1

            candidates = []
            for idx, deg in enumerate(degrees):
                if deg == 0: candidates.extend([idx] * 20)
                elif deg == 1: candidates.extend([idx] * 15)
                elif deg == 2: candidates.extend([idx] * 8)
                elif deg == 3: candidates.extend([idx] * 2)
                elif deg == 4: candidates.extend([idx] * 1)

            if not candidates: break
            parent = points[random.choice(candidates)]

            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.065, 0.099)
            new_x, new_y = round(parent[0] + r * math.cos(angle), 3), round(parent[1] + r * math.sin(angle), 3)
            new_p = (new_x, new_y)

            new_degree = 0
            valid = True
            for idx, existing_p in enumerate(points):
                if math.hypot(new_p[0] - existing_p[0], new_p[1] - existing_p[1]) <= distance_threshold:
                    new_degree += 1
                    if degrees[idx] >= max_subs:
                        valid = False
                        break
            
            # 【關鍵】檢查跨屬性重疊 (不能讓 Local 點附近擠滿 Global 點)
            if valid:
                temp_counts = list(local_global_counts)
                for l_idx, lp in enumerate(all_local_points):
                    if math.hypot(new_p[0] - lp[0], new_p[1] - lp[1]) <= distance_threshold:
                        temp_counts[l_idx] += 1
                        # 每個 Local 點周圍最多只能容忍 2 個 Global 點
                        if temp_counts[l_idx] > max_global_near_local:
                            valid = False
                            break
                            
                if valid and 1 <= new_degree <= 3:
                    points.append(new_p)
                    local_global_counts = temp_counts # 狀態確認，更新紀錄
                    placed = True
                    break
                
        if not placed:
            # 如果周圍真的太擠，強制把點往外推
            points.append((round(new_x + 0.15, 3), round(new_y + 0.15, 3)))

    return points

def generate_lora_system():
    NUM_LORAS = 100
    CLUSTERS = ["cluster_1", "cluster_2", "cluster_3"]
    LOCAL_PER_CLUSTER = 10
    GLOBAL_COUNT = NUM_LORAS - (len(CLUSTERS) * LOCAL_PER_CLUSTER)
    DISTANCE_THRESHOLD = 0.1

    loras = {}

    print("🌱 Growing Local LoRAs (Far apart from each other)...")
    # 把三個 Cluster 綁死在地圖的左下、右下、正上方，形成大三角
    local_points = {
        "cluster_1": generate_local_points(LOCAL_PER_CLUSTER, seed=(0.3, 0.3)),
        "cluster_2": generate_local_points(LOCAL_PER_CLUSTER, seed=(0.7, 0.3)),
        "cluster_3": generate_local_points(LOCAL_PER_CLUSTER, seed=(0.3, 0.5))
    }

    # 收集所有的 local points 以供後續密度檢查
    all_local_points = []
    for pts in local_points.values():
        all_local_points.extend(pts)

    print("🌱 Growing Global LoRAs (Avoiding dense local areas)...")
    # Global 從中心點開始長，往外蔓延，但會避開 Local 密集區
    global_points = generate_global_points(GLOBAL_COUNT, seed=(0.5, 0.4), all_local_points=all_local_points)

    # 1. 組合 JSON 結構並分配座標
    for i in range(1, NUM_LORAS + 1):
        lora_id = f"LoRA_{i}"
        loras[lora_id] = {"substitutes": []}
        
        if i <= GLOBAL_COUNT:
            loras[lora_id]["type"] = "global"
            loras[lora_id]["pos"] = list(global_points.pop(0))
        else:
            cluster_idx = (i - GLOBAL_COUNT - 1) // LOCAL_PER_CLUSTER
            cluster_name = CLUSTERS[cluster_idx]
            loras[lora_id]["type"] = "local"
            loras[lora_id]["cluster"] = cluster_name
            loras[lora_id]["pos"] = list(local_points[cluster_name].pop(0))

    # 2. 計算 Substitutes (同屬性且距離 < 0.1)
    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    for id1, info1 in loras.items():
        for id2, info2 in loras.items():
            if id1 >= id2: continue
            
            if info1["type"] != info2["type"]: continue
            if info1["type"] == "local" and info1.get("cluster") != info2.get("cluster"): continue
            
            if dist(info1["pos"], info2["pos"]) <= DISTANCE_THRESHOLD:
                info1["substitutes"].append(id2)
                info2["substitutes"].append(id1)

    degrees = [len(info["substitutes"]) for info in loras.values()]
    print(f"📊 統計：平均取代數量 {sum(degrees)/len(degrees):.2f}, 最小 {min(degrees)}, 最大 {max(degrees)}")

    # 3. 匯出 JSON
    with open("lora_metadata.json", "w", encoding="utf-8") as f:
        json.dump(loras, f, indent=4)
    print("✅ Saved JSON to 'lora_metadata.json'")

    # 4. 繪製散布圖
    colors = {"global": "gray", "cluster_1": "red", "cluster_2": "blue", "cluster_3": "green"}
    plt.figure(figsize=(12, 12))
    
    for id1, info1 in loras.items():
        for id2 in info1["substitutes"]:
            if id1 < id2: 
                p1, p2 = info1["pos"], loras[id2]["pos"]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linestyle='-', linewidth=1.5, zorder=1)

    for lora_id, info in loras.items():
        color_key = info["type"] if info["type"] == "global" else info["cluster"]
        size = 120 if info["type"] == "local" else 50
        plt.scatter(info["pos"][0], info["pos"][1], color=colors[color_key], s=size, zorder=2, edgecolors='black', alpha=0.8)
        num_str = lora_id.split("_")[1]
        plt.text(info["pos"][0] + 0.012, info["pos"][1] + 0.012, num_str, fontsize=8, zorder=3, fontweight='bold' if info["type"] == "local" else 'normal')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k], markersize=10, label=k.capitalize()) for k in colors]
    plt.legend(handles=handles, loc='best', title="LoRA Category")

    plt.title("LoRA Semantic Distribution (Clusters Separated & Global Density Controlled)")
    plt.xlabel("Semantic Dimension X")
    plt.ylabel("Semantic Dimension Y")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig("lora_distribution.png", dpi=200, bbox_inches='tight')
    print("✅ Saved plot to 'lora_distribution.png'")

if __name__ == "__main__":
    generate_lora_system()
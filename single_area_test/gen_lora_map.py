import os
import json
import random
import math

# ================= Configuration =================
SOURCE_DIR = "./testLoRA"       
OUTPUT_FILE = "lora_mapping.json"
NUM_VIRTUAL = 100               
SIMILARITY_THRESHOLD = 0.90     # 門檻值
# =================================================

def generate_map():
    print(f"🔨 Generating FIXED LoRA Mapping (Cluster 1, 2, 3)...")
    
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR, exist_ok=True)

    # 指向同一個實體檔案，節省空間
    default_source = "./testLoRA/LoRA_1" 
    # 確保該資料夾存在，避免 EFO 報錯
    if not os.path.exists(default_source):
        os.makedirs(default_source, exist_ok=True)
        # 建立一個假的 safetensors 防止載入錯誤
        with open(os.path.join(default_source, "adapter_model.safetensors"), "wb") as f:
            f.write(b"dummy_content_for_test")
    
    lora_map = {}
    
    # === 設定向量 (固定邏輯) ===
    # LoRA 1, 2, 3: 強制聚類 (角度非常接近, 在 0 度附近)
    cluster_angles = {
        "1": 0.0,
        "2": 0.02,  # 約 1.1 度
        "3": -0.02  # 約 -1.1 度
    }
    
    # 定義核心群組 ID
    core_cluster_ids = {"1", "2", "3"}

    for i in range(1, NUM_VIRTUAL + 1):
        vid = str(i)
        
        if vid in cluster_angles:
            # 這是我們的主角群，使用指定角度
            angle = cluster_angles[vid]
            l_type = "global"
        else:
            # 其他 LoRA (4~100): 
            # [修改 1] 為了物理上遠離 1,2,3 (0度)，我們將隨機範圍設在背對它們的地方
            # 0.9 的相似度大約需要角度差 < 0.45 弧度
            # 我們讓隨機角度從 1.0 開始到 2pi - 1.0，確保絕不會因為隨機而靠近 0 度
            random.seed(i) 
            angle = random.uniform(1.0, 2 * math.pi - 1.0)
            l_type = "global"

        vec = [math.cos(angle), math.sin(angle)]
        
        lora_map[vid] = {
            "name": vid,
            "type": l_type,
            "source_path": default_source,
            "embedding": vec,
            "substitutes": [] 
        }

    # === 計算 Affinity ===
    print("🧮 Calculating Affinity...")
    all_ids = sorted(list(lora_map.keys()), key=lambda x: int(x))
    
    for target_id in all_ids:
        target_vec = lora_map[target_id]["embedding"]
        substitutes = []
        
        for cand_id in all_ids:
            if target_id == cand_id: continue
            
            # [修改 2] 強制邏輯隔離：確保 1,2,3 與其他群組絕對不互通
            is_target_core = target_id in core_cluster_ids
            is_cand_core = cand_id in core_cluster_ids
            
            # 如果一個是核心群組，另一個不是，直接跳過，不計算相似度
            if is_target_core != is_cand_core:
                continue

            cand_vec = lora_map[cand_id]["embedding"]
            # Cosine Similarity
            dot = sum(a*b for a, b in zip(target_vec, cand_vec))
            norm_a = math.sqrt(sum(a*a for a in target_vec))
            norm_b = math.sqrt(sum(b*b for b in cand_vec))
            
            if norm_a == 0 or norm_b == 0: score = 0
            else: score = dot / (norm_a * norm_b)
            
            if score >= SIMILARITY_THRESHOLD:
                substitutes.append(cand_id)
        
        lora_map[target_id]["substitutes"] = substitutes

    # === 輸出 ===
    output_data = {"lora_map": lora_map}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated {OUTPUT_FILE}")
    print(f"   Cluster Check (Core Group):")
    print(f"   - LoRA 1 subs: {lora_map['1']['substitutes']}")
    print(f"   - LoRA 2 subs: {lora_map['2']['substitutes']}")
    print(f"   - LoRA 3 subs: {lora_map['3']['substitutes']}")
    
    # 檢查 4 號有沒有意外混入
    if "4" in lora_map:
         print(f"   - LoRA 4 subs: {lora_map['4']['substitutes']} (Should NOT contain 1, 2, 3)")

if __name__ == "__main__":
    generate_map()
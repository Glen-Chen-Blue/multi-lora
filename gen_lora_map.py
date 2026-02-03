import os
import json
import random
import math

# ================= Configuration =================
SOURCE_DIR = "./testLoRA"       
OUTPUT_FILE = "lora_mapping.json"
NUM_VIRTUAL = 100               
SIMILARITY_THRESHOLD = 0.90     # 設定為 0.9 以確保 1,2,3 互相涵蓋
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
    # LoRA 1, 2, 3: 強制聚類 (角度非常接近)
    cluster_angles = {
        "1": 0.0,
        "2": 0.02,  # 約 1.1 度
        "3": -0.02  # 約 -1.1 度
    }

    for i in range(1, NUM_VIRTUAL + 1):
        vid = str(i)
        
        if vid in cluster_angles:
            # 這是我們的主角群，使用指定角度
            angle = cluster_angles[vid]
            l_type = "global"
        else:
            # 其他 LoRA (4~100): 隨機散落在遠處 (避開 0.0 附近)
            # 使用固定種子確保每次 generate 結果一致 (雖然 exp_runner 會設種子，這裡也設一下保險)
            random.seed(i) 
            angle = random.uniform(0.5, 2 * math.pi)
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
    print(f"   Cluster Check:")
    print(f"   - LoRA 1 subs: {lora_map['1']['substitutes']} (Should include 2, 3)")
    print(f"   - LoRA 2 subs: {lora_map['2']['substitutes']} (Should include 1, 3)")
    print(f"   - LoRA 3 subs: {lora_map['3']['substitutes']} (Should include 1, 2)")

if __name__ == "__main__":
    generate_map()
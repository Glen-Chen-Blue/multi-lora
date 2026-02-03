import os
import json
import random
import math

# ================= Configuration =================
SOURCE_DIR = "./testLoRA"       # 實體 LoRA 來源資料夾
OUTPUT_FILE = "lora_mapping.json"
NUM_VIRTUAL = 100               # 模擬 100 個 LoRA
EMBED_DIM = 2                   # 使用 2 維向量
SIMILARITY_THRESHOLD = 0.99      # 相似度閾值
NUM_AREAS = 1                   # [新增] 定義總共有幾個 Edge Area
# =================================================

def cosine_similarity(v1, v2):
    dot_product = sum(a*b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a*a for a in v1))
    norm_b = math.sqrt(sum(b*b for b in v2))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (norm_a * norm_b)

def generate_map():
    print(f"🔍 Scanning physical adapters in {SOURCE_DIR}...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: {SOURCE_DIR} does not exist. Please create it and add some LoRA models.")
        return

    # 1. 取得實體來源
    sources = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    if not sources:
        print("❌ Error: No physical LoRA directories found.")
        return

    print(f"✅ Found {len(sources)} physical LoRAs. Generating {NUM_VIRTUAL} virtual LoRAs (2D)...")

    lora_map = {}
    embeddings = {}

    # 2. 生成虛擬 LoRA 與 Embeddings
    for i in range(1, NUM_VIRTUAL + 1):
        vid = str(i)
        phy_source = random.choice(sources)
        
        angle = random.uniform(0, 2 * math.pi)
        vec = [math.cos(angle), math.sin(angle)]

        # [新增] 決定 Type: "global" 或是 具體的 Area ID ("1" ~ "n")
        # 假設 20% 是區域專用，其餘是 Global
        if random.random() < 0.2:
            lora_type = str(random.randint(1, NUM_AREAS)) # 例如 "1", "2", "3"
        else:
            lora_type = "global"

        embeddings[vid] = vec
        lora_map[vid] = {
            "name": vid,
            "type": lora_type,  # [新增] 寫入類型
            "source_path": os.path.join(SOURCE_DIR, phy_source),
            "embedding": vec,
            "substitutes": [] 
        }

    # 3. 計算 Affinity 並直接寫入 lora_map
    print("🧮 Calculating Affinity (injecting into lora_map)...")
    
    all_ids = sorted(list(lora_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
    total_subs = 0
    
    for target_id in all_ids:
        target_vec = embeddings[target_id]
        substitutes = []
        
        for cand_id in all_ids:
            if target_id == cand_id: continue 
            
            score = cosine_similarity(target_vec, embeddings[cand_id])
            if score >= SIMILARITY_THRESHOLD:
                substitutes.append(cand_id)
        
        # 寫入 substitues，這裡我們暫時不根據 Type 過濾，因為 EFO/ControlNode 會在 Runtime 處理
        lora_map[target_id]["substitutes"] = substitutes
        total_subs += len(substitutes)

    # 4. 輸出 JSON
    output_data = {
        "lora_map": lora_map
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated {OUTPUT_FILE}")
    print(f"   - Virtual LoRAs: {len(all_ids)}")
    print(f"   - Avg Substitutes per LoRA: {total_subs / len(all_ids):.2f}")
    
    # 範例檢查
    example_id = "1"
    if example_id in lora_map:
        info = lora_map[example_id]
        print(f"   - Example: ID '1' (Type: {info['type']}) has substitutes: {info['substitutes'][:5]}...")

if __name__ == "__main__":
    generate_map()
import os
import torch
from safetensors.torch import save_file

def generate_fake_lora(lora_name, base_dir="../testLoRA", r=16, num_layers=32):
    # LLaMA-3-8B 的維度常數
    hidden_size = 4096
    intermediate_size = 14336
    kv_dim = 1024  # 8 KV heads * 128 dim
    
    # 定義各層的 (in_features, out_features)
    layer_shapes = {
        "self_attn.q_proj": (hidden_size, hidden_size),
        "self_attn.k_proj": (hidden_size, kv_dim),
        "self_attn.v_proj": (hidden_size, kv_dim),
        "self_attn.o_proj": (hidden_size, hidden_size),
        "mlp.gate_proj": (hidden_size, intermediate_size),
        "mlp.up_proj": (hidden_size, intermediate_size),
        "mlp.down_proj": (intermediate_size, hidden_size),
    }
    
    tensors = {}
    
    # 建立 32 層的假權重
    for i in range(num_layers):
        for suffix, (in_dim, out_dim) in layer_shapes.items():
            # 這是對應 self.model.named_modules() 裡面的名稱 n
            n = f"model.layers.{i}.{suffix}"
            
            # 這是 multilora_system.py 中對應尋找的 Key 格式
            key_A = f"base_model.model.{n}.lora_A.weight"
            key_B = f"base_model.model.{n}.lora_B.weight"
            
            # PEFT 標準中：
            # lora_A.weight 的 shape 為 (r, in_features)
            # lora_B.weight 的 shape 為 (out_features, r)
            # 使用 float16 確保檔案大小與真實訓練出來的接近 (約 40~50 MB)
            tensors[key_A] = torch.zeros((r, in_dim), dtype=torch.float16)
            tensors[key_B] = torch.zeros((out_dim, r), dtype=torch.float16)
            
    # 建立目錄
    out_dir = os.path.join(base_dir, lora_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 儲存 safetensors
    out_file = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(tensors, out_file)
    
    # 印出建立結果與大小
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"✅ 成功建立: {out_file} (檔案大小: {size_mb:.2f} MB)")

if __name__ == "__main__":
    # 在這裡設定你想要產生的 LoRA 名稱
    loras_to_create = ["LoRA_1", "LoRA_2", "LoRA_3"]
    
    print("開始產生假的 LoRA safetensors...")
    for lora in loras_to_create:
        generate_fake_lora(lora)
    print("🎉 全部產生完畢！")
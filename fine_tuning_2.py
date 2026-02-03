import os
import gc
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# =========================================================
# 🔒 GPU 設定
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 修改 1: 使用 4bit 量化版本模型
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_ROOT = "./testLoRA"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "LoRA_test2")

# =========================================================
# 🧠 LoRA 設定
# =========================================================
LORA_CONFIG = dict(
    r=16,                # 修改 2: Rank 改為 64
    lora_alpha=128,      # 建議 alpha 設為 r 的 2 倍
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
)

# =========================================================
# 📚 簡易 Dataset builder (只用 Alpaca)
# =========================================================
def build_quick_dataset(max_rows=100):
    # 修改 4: 只讀取前 100 筆資料
    print(f"📚 Loading dataset (limit: {max_rows} rows)...")
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{max_rows}]")

    def _map(ex):
        return {
            "text":
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}"
        }

    return ds.map(_map, remove_columns=ds.column_names)

# =========================================================
# 🚀 主訓練流程
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. Load base model ----------
    print(f"🚀 Loading base model: {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,  # 修改 1: 開啟 4bit 載入
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 套用 LoRA
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # ---------- 2. Prepare Data ----------
    # 修改 4: 只取 100 筆
    train_ds = build_quick_dataset(max_rows=100)

    # ---------- 3. Fast Training ----------
    print(f"\n🛠️ Starting Quick Training for {OUTPUT_DIR}...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,       # 修改: 只要快速產生，1 epoch 即可
            per_device_train_batch_size=4, # 4bit 模型省顯存，batch 可以大一點
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            optim="adamw_8bit",       # 使用 8bit 優化器更省顯存
            lr_scheduler_type="linear",
            logging_steps=5,
            max_seq_length=512,       # 縮短長度以加速
            dataset_text_field="text",
            report_to="none",
        ),
    )

    trainer.train()

    # ---------- 4. Save ----------
    print(f"💾 Saving to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ---------- Cleanup ----------
    del trainer
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("\n🎉 Quick LoRA generation finished!")

if __name__ == "__main__":
    main()
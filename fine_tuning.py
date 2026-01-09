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

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
OUTPUT_ROOT = "./testLoRA"

# =========================================================
# 🧠 LoRA 設定
# =========================================================
LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
)

# =========================================================
# 🏋️ SFT 訓練設定
# =========================================================
SFT_BASE_CONFIG = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    save_strategy="no",
    report_to="none",
    logging_steps=10,
)

# =========================================================
# 📚 Dataset builders（全部用穩定來源）
# =========================================================
def build_math_dataset(max_rows=4000):
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def _map(ex):
        return {
            "text":
            f"### Instruction:\n{ex['question']}\n\n"
            f"### Response:\n{ex['answer']}"
        }

    return ds.map(_map, remove_columns=ds.column_names).select(range(max_rows))


def build_code_dataset(max_rows=4000):
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")

    def _map(ex):
        return {
            "text":
            f"### Instruction:\n{ex['prompt']}\n\n"
            f"### Response:\n{ex['completion']}"
        }

    return ds.map(_map, remove_columns=ds.column_names).select(range(max_rows))


def build_chat_dataset(max_rows=4000):
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    def _map(ex):
        return {
            "text":
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}"
        }

    return ds.map(_map, remove_columns=ds.column_names).select(range(max_rows))


# =========================================================
# 🔍 Dataset sanity check（關鍵）
# =========================================================
def preload_and_check_datasets():
    print("\n🔍 Preloading datasets (sanity check)...")

    builders = {
        "math": build_math_dataset,
        "code": build_code_dataset,
        "chat": build_chat_dataset,
    }

    datasets = {}
    for name, builder in builders.items():
        print(f"  • Loading {name} dataset...")
        ds = builder(max_rows=100)  # 少量先檢查
        print(f"    ✓ {name}: {len(ds)} samples loaded")
        print(f"    ✓ sample text:\n{ds[0]['text'][:200]}...\n")
        datasets[name] = builder  # 確認沒問題才存

    print("✅ All datasets loaded successfully!\n")
    return datasets


# =========================================================
# 🚀 主訓練流程
# =========================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---------- 1. Dataset sanity check ----------
    dataset_builders = preload_and_check_datasets()

    # ---------- 2. Load base model ----------
    print("🚀 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    TRAINING_PLAN = [
        ("LoRA_math", dataset_builders["math"]),
        ("LoRA_code", dataset_builders["code"]),
        ("LoRA_chat", dataset_builders["chat"]),
    ]

    # ---------- 3. Train each LoRA ----------
    for lora_name, builder in TRAINING_PLAN:
        print(f"\n🛠️ Training {lora_name}")
        out_dir = os.path.join(OUTPUT_ROOT, lora_name)
        os.makedirs(out_dir, exist_ok=True)

        train_ds = builder(max_rows=6000)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            args=SFTConfig(
                output_dir=out_dir,
                num_train_epochs=3,
                max_seq_length=1024,
                dataset_text_field="text",
                **SFT_BASE_CONFIG,
            ),
        )

        trainer.train()

        print(f"💾 Saving {lora_name}")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # ---------- cleanup ----------
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        model.base_model.unload()
        model = FastLanguageModel.get_peft_model(
            model.get_base_model(), **LORA_CONFIG
        )

    print("\n🎉 All LoRA training finished successfully!")


if __name__ == "__main__":
    main()

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "unsloth/Meta-Llama-3.1-8B"

# 只下載 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 下載 model（會進 cache）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"  # 避免一開始就吃 GPU
)

print("✅ Model & tokenizer downloaded!")
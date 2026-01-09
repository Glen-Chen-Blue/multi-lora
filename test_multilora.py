from multilora_system import MultiLoRAEngine
import time

# ============================================================
# 1. 初始化 Engine
# ============================================================
model_id = "unsloth/Meta-Llama-3.1-8B"
engine = MultiLoRAEngine(
    model_id,
    adapter_slots=1,
    max_batch_size=2,
)

# ============================================================
# 2. 載入 LoRA 到 CPU RAM
# ============================================================
engine.load_adapters_to_cpu("./testLoRA")

# ============================================================
# 3. 加入請求（全部用同一個 adapter）
# ============================================================
N_REQ = 2
PROMPT = "Explain Transformer self-attention in one sentence."
ADAPTER_ID = "chat"

for _ in range(N_REQ):
    engine.add_request(PROMPT, adapter_id=ADAPTER_ID)

print("\n🚀 開始調度測試...")

# ============================================================
# 4. 執行生成循環
# ============================================================
start = time.time()
step_count = 0
MAX_STEPS = 3000

while len(engine.finished_results) < N_REQ and step_count < MAX_STEPS:
    has_running = engine.step()
    step_count += 1

    # 沒有在跑、也沒有待處理請求，就可以停
    if not has_running and not engine.request_queue:
        break

end = time.time()
print(f"調度結束，總共花費時間: {end - start:.2f} 秒")
print(f"總 step 數: {step_count}")

# ============================================================
# 5. 結果驗證（CPU-only）
# ============================================================
print("\n" + "=" * 50)
for i, res in enumerate(engine.finished_results):
    text = engine.tokenizer.decode(res["tokens"], skip_special_tokens=True)
    aid = res["adapter_id"]
    reason = res["reason"]

    print(f"完成 {i+1:02d} [LoRA={aid}, reason={reason}]:")
    print(text)
    print("-" * 50)

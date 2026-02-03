import time
import os
# 引入修改後的 MultiLoRAEngine (基於 vLLM)
from test import MultiLoRAEngine

# ============================================================
# 1. 設定與初始化
# ============================================================
# 確保這裡有一個真實存在的 Base Model，dLoRA 通常需要與權重匹配
MODEL_ID = "unsloth/Meta-Llama-3.1-8B"  # 範例使用小模型，請換成您的真實模型如 "meta-llama/Llama-2-7b-hf"
ADAPTER_DIR = "./testLoRA"      # 存放 LoRA 的根目錄

# 初始化引擎
# 這會掃描 ADAPTER_DIR 下的所有子目錄作為 Adapter
print(f"🚀 Initializing MultiLoRAEngine with model: {MODEL_ID}")
engine = MultiLoRAEngine(
    model_id=MODEL_ID,
    adapter_dirs=[ADAPTER_DIR],
    max_batch_size=4,
    device="cuda"
)

# 列出已載入的 Adapters
print(f"📋 Available Adapters: {engine.adapter_to_id}")

# 檢查是否有 'chat' adapter，如果沒有則使用第一個掃描到的
target_adapter = "chat"
if target_adapter not in engine.adapter_to_id:
    if engine.adapter_to_id:
        target_adapter = list(engine.adapter_to_id.keys())[0]
        print(f"⚠️ 'chat' adapter not found, using '{target_adapter}' instead.")
    else:
        raise RuntimeError("❌ No adapters found in ./testLoRA!")

# ============================================================
# 2. 定義推理與監控函式
# ============================================================
def run_inference_loop(description: str, steps: int = 100):
    print(f"\n▶️  Running: {description}")
    start_time = time.time()
    
    # 簡單的迴圈來驅動引擎
    # 在真實應用中，這通常是一個背景無窮迴圈
    active_steps = 0
    while active_steps < steps:
        has_work = engine.step()
        
        # 檢查是否有完成的結果
        while engine.finished_results:
            res = engine.finished_results.popleft()
            print(f"   ✅ Finished Request [{res['request_id']}]: {res['text'][:50]}...")
            
        if not has_work and engine.is_idle():
            break
            
        if has_work:
            active_steps += 1
            
    end_time = time.time()
    print(f"⏹️  Done. Duration: {end_time - start_time:.2f}s")

# ============================================================
# 3. 測試情境 A: 一般混合模式 (Mixed Mode)
# ============================================================
print("\n" + "="*60)
print("🧪 Test A: Mixed Mode (Default)")
print("="*60)

# 加入請求
prompts = [
    ("Explain quantum physics in simple terms.", "req_A1"),
    ("Write a poem about rust.", "req_A2")
]

for p, rid in prompts:
    print(f"📥 Adding request: {rid}")
    engine.add_request(prompt=p, adapter_id=target_adapter, request_id=rid, max_new_tokens=32)

# 執行
run_inference_loop("Processing Mixed Requests")

# ============================================================
# 4. 測試情境 B: Merge 模式 (Merged/Exclusive Mode)
# ============================================================
print("\n" + "="*60)
print(f"🧪 Test B: Merge Mode -> Merging '{target_adapter}'")
print("="*60)
print("ℹ️  This forces the engine to optimize for this specific LoRA.")

# 執行 Merge
engine.merge_adapter(target_adapter, force=True)

# 加入請求 (注意：此時若加入其他 Adapter 的請求可能會被擋下或變慢，視實作而定)
prompts_merge = [
    ("What is the capital of France?", "req_B1"),
    ("Python code for fibonacci.", "req_B2")
]

for p, rid in prompts_merge:
    print(f"📥 Adding request: {rid} (Optimized)")
    engine.add_request(prompt=p, adapter_id=target_adapter, request_id=rid, max_new_tokens=32)

# 執行
run_inference_loop("Processing Merged Requests")

# ============================================================
# 5. 測試情境 C: Unmerge (恢復混合模式)
# ============================================================
print("\n" + "="*60)
print("🧪 Test C: Unmerge -> Back to Mixed Mode")
print("="*60)

engine.unmerge_all()

# 加入請求
print(f"📥 Adding final request.")
engine.add_request(prompt="Say hello!", adapter_id=target_adapter, request_id="req_C1", max_new_tokens=10)

run_inference_loop("Processing Final Requests")

print("\n✨ All tests completed.")
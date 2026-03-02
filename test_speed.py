import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from multilora_system import MultiLoRAEngine
from config import MODEL_ID

# =====================================================================
# 模擬 Adapter Fetcher: 不管傳什麼 ID，都回傳同一個真實的 safetensors
# =====================================================================
def mock_adapter_fetcher(adapter_id: str) -> bytes:
    file_path = "./testLoRA/LoRA_1/adapter_model.safetensors"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到測試檔案: {file_path}")
    with open(file_path, "rb") as f:
        return f.read()

def measure_performance(engine, prompt, batch_size, is_merged=False, num_trials=5, decode_steps=20):
    """測量給定 Batch Size 下的 Prefill 與 Decode 時間"""
    prefill_times = []
    decode_step_times = []

    aid = "merged_lora" if is_merged else "test_lora_1"

    for trial in range(num_trials):
        # 確保佇列清空
        with engine.lock:
            engine.request_queue.clear()
            engine.running_queue.clear()
        
        # 塞入指定數量的 Request (皆使用同一種 LoRA，避免超過 capacity)
        for i in range(batch_size):
            engine.add_request(prompt, aid, request_id=f"req_{batch_size}_{trial}_{i}")

        torch.cuda.synchronize()
        
        # ===============================
        # 1. 測量 Prefill 階段 (第一個 step)
        # ===============================
        t0 = time.time()
        engine.step()
        torch.cuda.synchronize()
        prefill_times.append(time.time() - t0)

        # ===============================
        # 2. 測量 Decode 階段 (後續 steps)
        # ===============================
        t1 = time.time()
        for _ in range(decode_steps):
            engine.step()
        torch.cuda.synchronize()
        
        decode_total = time.time() - t1
        # 計算生成「單一 Token」(即一個 step) 的平均時間
        decode_step_times.append(decode_total / decode_steps)

    return np.mean(prefill_times), np.mean(decode_step_times)


def run_experiment():
    print("🚀 初始化 MultiLoRAEngine 中...")
    engine = MultiLoRAEngine(
        model_id=MODEL_ID,
        adapter_fetcher=mock_adapter_fetcher
    )

    # 構造一個超長的 Prompt 來確保被截斷為 FIXED_INPUT_LEN (512 tokens)
    prompt = "This is a performance measurement test for the edge federation system. " * 100

    print("\n🔥 正在進行 GPU 暖機 (Warm-up)... (忽略前幾次的延遲)")
    _ = measure_performance(engine, prompt, batch_size=2, is_merged=False, num_trials=2, decode_steps=10)
    print("✅ 暖機完成！開始正式測量。")

    # =================================================================
    # 測試 1: 測量 SIM_LOAD_DELAY (Disk -> Host Memory 延遲)
    # =================================================================
    print("\n[測試 1] 測量 SIM_LOAD_DELAY (重複 10 次取平均)...")
    load_times = []
    for i in range(10):
        aid = f"load_test_{i}"
        t0 = time.time()
        engine._ensure_cpu_loaded(aid)
        load_times.append(time.time() - t0)
        # 測完馬上從快取清掉，確保下次依然是冷啟動
        with engine.lock:
            if aid in engine.cpu_cache:
                del engine.cpu_cache[aid]

    sim_load_delay = np.mean(load_times)
    print(f"✅ SIM_LOAD_DELAY 測量結果: {sim_load_delay:.4f} 秒")

    # =================================================================
    # 測試 2 & 3: 測量 Unmerged 模式下的 Prefill, Decode 與 Slope
    # =================================================================
    print("\n[測試 2 & 3] 測量 Unmerged 模式 (Batch Size = 1, 2, 4, 8, 10)...")
    # Unmerged max capacity 預設為 12，為避免 (batch + unique loras) 超出 12 導致 Request 被拒絕，最高測到 10
    batch_sizes = [1, 2, 4, 8, 10]
    unmerged_prefills = []
    unmerged_decodes = []
    
    for bs in batch_sizes:
        print(f"  -> 正在測量 Batch Size {bs}...")
        p_time, d_time = measure_performance(engine, prompt, bs, is_merged=False, num_trials=5)
        unmerged_prefills.append(p_time)
        unmerged_decodes.append(d_time)

    # Prefill base time (以 BS=1 時為基準)
    sim_prefill_base_time = unmerged_prefills[0]

    # 使用線性迴歸計算 Decode Base Time 與 Slope (Batch 干擾斜率)
    slope, intercept, r_value, p_value, std_err = linregress(batch_sizes, unmerged_decodes)
    sim_decode_base_time = intercept
    sim_decode_slope = slope

    print(f"✅ SIM_PREFILL_BASE_TIME (BS=1): {sim_prefill_base_time:.4f} 秒")
    print(f"✅ SIM_DECODE_BASE_TIME (Intercept): {sim_decode_base_time:.4f} 秒")
    print(f"✅ SIM_DECODE_SLOPE: {sim_decode_slope:.6f} 秒/Req")
    print(f"   (迴歸 R-squared: {r_value**2:.4f})")

    # =================================================================
    # 測試 4: 測量 Merged 模式與 MERGE_SPEED_MULTIPLIER
    # =================================================================
    print("\n[測試 4] 切換至 Merged 模式並計算 MERGE_SPEED_MULTIPLIER...")
    engine.merge_adapter("merged_lora", force=True)
    
    # 在 Merged 模式下測量 BS=1
    merged_p_time, merged_d_time = measure_performance(engine, prompt, batch_size=1, is_merged=True, num_trials=5)
    
    # 計算乘數 (Merged / Unmerged 的 Decode 時間比值)
    merge_speed_multiplier = merged_d_time / unmerged_decodes[0]
    
    print(f"  -> Unmerged BS=1 Decode Time: {unmerged_decodes[0]:.4f} 秒")
    print(f"  -> Merged BS=1 Decode Time: {merged_d_time:.4f} 秒")
    print(f"✅ MERGE_SPEED_MULTIPLIER 測量結果: {merge_speed_multiplier:.4f}")

    # =================================================================
    # 輸出最終供 config.py 使用的常數結果
    # =================================================================
    print("\n" + "="*50)
    print("🎯 請將以下測量結果複製到 config.py 中：")
    print("="*50)
    print(f"SIM_LOAD_DELAY = {sim_load_delay:.3f}           # 從 Disk 載入到 Host Memory 的 I/O 延遲 (秒)")
    print(f"SIM_PREFILL_BASE_TIME = {sim_prefill_base_time:.3f}    # Prefill 階段處理單一請求的基礎時間 (秒)")
    print(f"SIM_DECODE_BASE_TIME = {sim_decode_base_time:.3f}     # Decode 階段產生單一 Token 的基礎時間 (秒)")
    print(f"SIM_DECODE_SLOPE = {sim_decode_slope:.4f}        # Decode 階段受 Batch Size 影響的斜率 (增加的干擾時間)")
    print(f"MERGE_SPEED_MULTIPLIER = {merge_speed_multiplier:.3f}     # 當處於 Merged 模式時的運算加速/時間折扣乘數")
    print("="*50)

    # =================================================================
    # 繪製結果圖表
    # =================================================================
    print("\n📊 正在繪製並儲存分析圖表...")
    plt.figure(figsize=(14, 5))

    # 子圖 1：Decode 時間對 Batch Size 的線性迴歸
    plt.subplot(1, 3, 1)
    plt.scatter(batch_sizes, unmerged_decodes, color='blue', label='Measured Data', zorder=5)
    
    x_range = np.linspace(0, max(batch_sizes)+1, 100)
    plt.plot(x_range, sim_decode_base_time + sim_decode_slope * x_range, 
             color='red', linestyle='--', label=f'Fit: y={sim_decode_slope:.4f}x + {sim_decode_base_time:.4f}')
    
    plt.title('Unmerged Mode:\nDecode Time vs Batch Size')
    plt.xlabel('Batch Size ($|\mathcal{B}|$)', fontsize=12)
    plt.ylabel('Time per Token Step (sec)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 子圖 2：Prefill 時間對 Batch Size (觀察是否線性增加)
    plt.subplot(1, 3, 2)
    plt.plot(batch_sizes, unmerged_prefills, marker='o', color='purple', linewidth=2)
    plt.title('Unmerged Mode:\nPrefill Total Time vs Batch Size')
    plt.xlabel('Batch Size ($|\mathcal{B}|$)', fontsize=12)
    plt.ylabel('Prefill Phase Time (sec)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 子圖 3：Merged vs Unmerged 速度比較 (Batch Size 1)
    plt.subplot(1, 3, 3)
    categories = ['Prefill Time', 'Decode Time (per token)']
    unmerged_vals = [unmerged_prefills[0], unmerged_decodes[0]]
    merged_vals = [merged_p_time, merged_d_time]

    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width/2, unmerged_vals, width, label='Unmerged', color='orange', edgecolor='black')
    plt.bar(x + width/2, merged_vals, width, label='Merged', color='green', edgecolor='black')

    plt.title('Performance Comparison\n(Batch Size = 1)')
    plt.xticks(x, categories)
    plt.ylabel('Time (sec)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hardware_latency_measurements.png', dpi=150)
    print("✅ 圖表已儲存為 'hardware_latency_measurements.png'。實驗完成！")

if __name__ == "__main__":
    run_experiment()
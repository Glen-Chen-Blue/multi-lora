import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import gc
from typing import List, Dict
from sklearn.linear_model import LinearRegression

# Import your system
from multilora_system import MultiLoRAEngine

# =========================================================================
# Configuration
# =========================================================================
MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
LORA_PATH = "./testLoRA/LoRA_1/adapter_model.safetensors"

INPUT_LEN = 512
OUTPUT_LEN = 256

# Constants for Slot Logic (Must match multilora_system.py)
CAPACITY_MERGED = 15  # Updated based on new optimization
CAPACITY_UNMERGED = 12

# =========================================================================
# Helper: Adapter Fetcher Simulation
# =========================================================================
def get_adapter_bytes(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Adapter file not found: {path}")
    with open(path, 'rb') as f:
        return f.read()

_LORA_BYTES = None
def mock_adapter_fetcher(adapter_id: str) -> bytes:
    global _LORA_BYTES
    if _LORA_BYTES is None:
        _LORA_BYTES = get_adapter_bytes(LORA_PATH)
    return _LORA_BYTES

# =========================================================================
# Test Engine
# =========================================================================
def run_speed_test():
    print("\n" + "="*60)
    print(f"🚀 Initializing Speed Test (Robust Mode)")
    print(f"Model: {MODEL_ID}")
    print(f"LoRA: {LORA_PATH}")
    print(f"Input/Output: {INPUT_LEN}/{OUTPUT_LEN}")
    print(f"Runs per Batch Size: 3 (Taking Median)")
    print("="*60)

    # Initialize Engine
    engine = MultiLoRAEngine(
        model_id=MODEL_ID, 
        device="cuda", 
        enable_monitor=False,
        adapter_fetcher=mock_adapter_fetcher
    )

    # Generate Prompt
    base_prompt = "Hello " * 200 
    tokens = engine.tokenizer.encode(base_prompt)
    while len(tokens) < INPUT_LEN:
        base_prompt += " world"
        tokens = engine.tokenizer.encode(base_prompt)
    base_prompt = engine.tokenizer.decode(tokens[:INPUT_LEN])
    
    print(f"✅ Generated prompt length: {len(engine.tokenizer.encode(base_prompt))} tokens")

    # Define Scenarios
    scenarios = [
        {"name": "Merged (1 LoRA)", "type": "merged", "n_unique_lora": 1},
        {"name": "Unmerged (1 LoRA)", "type": "unmerged", "n_unique_lora": 1},
        # {"name": "Unmerged (2 LoRA)", "type": "unmerged", "n_unique_lora": 2},
        # {"name": "Unmerged (4 LoRA)", "type": "unmerged", "n_unique_lora": 4},
        # {"name": "Unmerged (8 LoRA)", "type": "unmerged", "n_unique_lora": 8},
    ]

    results = []

    for sc in scenarios:
        mode_name = sc["name"]
        is_merged = (sc["type"] == "merged")
        n_lora = sc["n_unique_lora"]

        if is_merged:
            max_batch = CAPACITY_MERGED
        else:
            max_batch = CAPACITY_UNMERGED - n_lora
        
        if max_batch <= 0:
            continue

        print(f"\n🧪 Scenario: {mode_name} | Max Batch: {max_batch}")

        adapter_ids = [f"lora_{i}" for i in range(n_lora)]
        engine.update_known_adapters(adapter_ids)

        if is_merged:
            engine.merge_adapter(adapter_ids[0], force=True)
        else:
            engine.unmerge_all()
            engine._ensure_adapters_resident(adapter_ids)

        batch_sizes = list(range(1, max_batch + 1))
        
        for bs in batch_sizes:
            # Storage for multiple runs
            run_prefills = []
            run_decodes = []
            
            # Robustness: Run 3 times, take median
            for run_idx in range(3):
                # Clean State
                engine.request_queue = []
                engine.running_queue = []
                torch.cuda.empty_cache()
                gc.collect()

                # Add Requests
                for i in range(bs):
                    aid = adapter_ids[i % n_lora]
                    rid = f"req_{mode_name}_{bs}_{i}_run{run_idx}"
                    engine.add_request(
                        prompt=base_prompt,
                        adapter_id=aid,
                        request_id=rid,
                        max_new_tokens=OUTPUT_LEN
                    )

                # --- Measure Prefill ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                has_work = engine.step()
                
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                if not has_work:
                    print(f"❌ Batch {bs} Run {run_idx}: Failed start")
                    break
                
                run_prefills.append((t1 - t0) * 1000)

                # --- Measure Decode ---
                steps = 0
                torch.cuda.synchronize()
                t_dec_start = time.perf_counter()
                
                while True:
                    if not engine.step(): break
                    steps += 1
                
                torch.cuda.synchronize()
                t_dec_end = time.perf_counter()
                
                avg_dec = ((t_dec_end - t_dec_start) * 1000) / steps if steps > 0 else 0
                run_decodes.append(avg_dec)

            # Calculate Median
            if len(run_prefills) == 3:
                median_prefill = np.median(run_prefills)
                median_decode = np.median(run_decodes)
                
                print(f"   Batch {bs:2d} | Prefill (Med): {median_prefill:6.2f} ms | Decode (Med): {median_decode:6.2f} ms/token")

                results.append({
                    "Scenario": mode_name,
                    "Is_Merged": is_merged,
                    "N_LoRA": n_lora,
                    "Batch_Size": bs,
                    "Prefill_Latency": median_prefill,
                    "Decode_Latency": median_decode
                })
            else:
                print(f"⚠️ Batch {bs} skipped due to failures.")

    # Save Data
    df = pd.DataFrame(results)
    df.to_csv("speed_test_results.csv", index=False)
    print("\n💾 Data saved to speed_test_results.csv")

    analyze_and_plot(df)

# =========================================================================
# Analysis & Plotting
# =========================================================================
def analyze_and_plot(df):
    """
    Calculates coefficients excluding Batch_Size=1.
    Formula: T_prefill = theta * (tau + beta * Batch_Size)
    """
    print("\n" + "="*60)
    print("🧮 Calculating Coefficients (Robust Fit)")
    print("Excluding Batch_Size = 1 from regression.")
    print("="*60)

    # 1. Fit Base Line (Merged Mode)
    merged_data = df[df["Scenario"] == "Merged (1 LoRA)"]
    
    if merged_data.empty:
        print("❌ No Merged data found.")
        return

    # Filter out Batch Size 1
    mask = merged_data["Batch_Size"] > 1
    valid_data = merged_data[mask]
    
    if valid_data.empty:
        print("❌ Not enough data points > 1.")
        return

    X_merged = valid_data["Batch_Size"].values.reshape(-1, 1)
    y_merged = valid_data["Prefill_Latency"].values

    reg = LinearRegression()
    reg.fit(X_merged, y_merged)
    
    tau_pre = reg.intercept_
    beta_new = reg.coef_[0]

    print(f"🔹 Base Parameters (Merged Mode):")
    print(f"   tau_pre (Base Constant Overhead)  : {tau_pre:.4f} ms")
    print(f"   beta_new (Per-Request Latency)    : {beta_new:.4f} ms/request") 
    print(f"   (Note: This beta includes computing 512 tokens + 1st gen token)")

    # 2. Calculate Theta (Unmerged Overhead)
    unmerged_data = df[df["Scenario"] == "Unmerged (1 LoRA)"]
    theta_unmerged = 1.0 
    
    if not unmerged_data.empty:
        # Also filter batch 1
        mask_un = unmerged_data["Batch_Size"] > 1
        valid_un = unmerged_data[mask_un]
        
        if not valid_un.empty:
            X_un = valid_un["Batch_Size"].values
            y_un = valid_un["Prefill_Latency"].values
            
            # y_un = theta * (tau + beta * X)
            expected_base = tau_pre + beta_new * X_un
            theta_ratios = y_un / expected_base
            theta_unmerged = np.mean(theta_ratios)

            print(f"🔸 Mode Overhead:")
            print(f"   theta_merged                 : 1.0000")
            print(f"   theta_unmerged (Calculated)  : {theta_unmerged:.4f}")
    
    # Save coeffs
    with open("latency_coefficients.txt", "w") as f:
        f.write(f"tau_pre={tau_pre}\n")
        f.write(f"beta_new={beta_new}\n")
        f.write(f"theta_unmerged={theta_unmerged}\n")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Prefill
    for name, group in df.groupby("Scenario"):
        ax1.plot(group["Batch_Size"], group["Prefill_Latency"], marker='o', label=name)
    
    # Plot Fit Line
    x_range = np.linspace(1, CAPACITY_MERGED, 50)
    y_pred = tau_pre + beta_new * x_range
    ax1.plot(x_range, y_pred, 'k--', alpha=0.5, label=f"Fit (Batch>1): {tau_pre:.0f} + {beta_new:.1f}*B")

    ax1.set_title(f"Prefill Latency (Median of 3)\nInput={INPUT_LEN}")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Decode
    for name, group in df.groupby("Scenario"):
        ax2.plot(group["Batch_Size"], group["Decode_Latency"], marker='s', label=name)
    
    ax2.set_title(f"Decode Latency (Median of 3)\nOutput={OUTPUT_LEN}")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Latency (ms/token)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("latency_analysis.png")
    print("\n📈 Plot saved to latency_analysis.png")

if __name__ == "__main__":
    try:
        run_speed_test()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
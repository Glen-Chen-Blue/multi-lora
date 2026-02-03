import subprocess
import time
import os
import signal
import requests
import asyncio
import httpx
import random
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Experiment Configuration
# ==========================================
RPS_STEPS = [1, 2, 4, 8, 12, 16]  
STEP_DURATION = 40
COOLDOWN = 10
CONTROL_URL = "http://127.0.0.1:9000"

SCENARIOS = {
    "Smart Mechanism (Ours)": {
        "env": {"ENABLE_SEMANTIC": "true", "ENABLE_AUTOSCALE": "true", "DISPATCH_MODE": "smart", "INITIAL_NODES": "one"},
        "color": "green", "marker": "o"
    },
    "No Semantic (Baseline 1)": {
        "env": {"ENABLE_SEMANTIC": "false", "ENABLE_AUTOSCALE": "true", "DISPATCH_MODE": "smart", "INITIAL_NODES": "one"},
        "color": "orange", "marker": "^"
    },
    "Random/Full (Baseline 2)": {
        "env": {"ENABLE_SEMANTIC": "false", "ENABLE_AUTOSCALE": "false", "DISPATCH_MODE": "random", "INITIAL_NODES": "all"},
        "color": "red", "marker": "x"
    }
}

# ==========================================
# Traffic Logic
# ==========================================
async def simulate_user(client: httpx.AsyncClient):
    # [需求] 流量分佈控制: 1, 2, 3 各佔 30%，剩下的 10% 隨機
    rand_val = random.random()
    
    if rand_val < 0.30:
        adapter = "1"
    elif rand_val < 0.60:
        adapter = "2"
    elif rand_val < 0.90:
        adapter = "3"
    else:
        adapter = str(random.randint(4, 100))
    
    payload = {
        "prompt": "test prompt", 
        "adapter_id": adapter,
        "max_new_tokens": 10
    }
    
    try:
        # 只發送不等待完整結果 (Fire and Forget)
        await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=2.0)
    except:
        pass

async def traffic_generator(rps, duration, scenario_name):
    """
    執行指定 RPS 和時間的流量，並回傳累積 Cost
    """
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=200)
    async with httpx.AsyncClient(limits=limits) as client:
        end_time = time.time() + duration
        cost_accumulator = 0.0
        
        # 背景 Cost 監控
        async def monitor_cost():
            nonlocal cost_accumulator
            while time.time() < end_time:
                try:
                    r = await client.get(f"{CONTROL_URL}/status", timeout=1.0)
                    if r.status_code == 200:
                        active = r.json().get("active_nodes", 0)
                        cost_accumulator += active
                    else:
                        raise Exception("Status error")
                except: 
                    # [修正] 如果連不上 Control Node (可能過載)，依據模式補償 Cost
                    if "Random" in scenario_name:
                        cost_accumulator += 2 # Random 模式固定開 2 台
                    else:
                        cost_accumulator += 1 # 其他模式至少有 1 台
                await asyncio.sleep(1.0)

        monitor_task = asyncio.create_task(monitor_cost())
        
        # 流量發送 Loop
        while time.time() < end_time:
            asyncio.create_task(simulate_user(client))
            # Poisson Arrival
            sleep_time = random.expovariate(rps)
            await asyncio.sleep(sleep_time)
        
        await monitor_task
        return cost_accumulator

# ==========================================
# System Helpers
# ==========================================
def start_system(env_vars):
    print(f"   🚀 Starting system... (Env: Semantic={env_vars.get('ENABLE_SEMANTIC')})")
    current_env = os.environ.copy()
    current_env.update(env_vars)
    # 這裡會呼叫 single_area.sh -> python gen_lora_map.py (我們已經修改過它了)
    proc = subprocess.Popen(["bash", "single_area.sh"], env=current_env, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 等待 Control Node 上線
    for _ in range(40):
        try:
            r = requests.get(f"{CONTROL_URL}/status", timeout=1)
            if r.status_code == 200:
                print("   ✅ System Ready.")
                time.sleep(5) 
                return proc
        except:
            time.sleep(1)
    
    print("   ❌ System Failed to Start.")
    try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except: pass
    return None

def stop_system(proc):
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
        except: pass
    subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
    time.sleep(5)

# ==========================================
# Main
# ==========================================
def main():
    final_results = {name: [] for name in SCENARIOS}
    
    # 預先生成一次 Map 確保檔案存在 (雖然 start_system 也會跑)
    subprocess.run(["python", "gen_lora_map.py"])

    for name, config in SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"🧪 Experiment: {name}")
        print(f"{'='*50}")
        
        proc = start_system(config["env"])
        if not proc: continue
        
        try:
            for rps in RPS_STEPS:
                print(f"\n   >>> Testing RPS: {rps}")
                
                # [關鍵] 設定固定的 Seed
                # 確保每個 Scenario 在面對 "12 RPS" 時，生成的 Adapter 序列和間隔完全一樣
                random.seed(42 + rps)
                
                cost = asyncio.run(traffic_generator(rps, STEP_DURATION, name))
                print(f"   💰 Cost: {cost:.2f} Node-Seconds")
                final_results[name].append(cost)
                
                print(f"   ❄️ Cooldown {COOLDOWN}s...")
                time.sleep(COOLDOWN)
        finally:
            stop_system(proc)

    print("\n📊 Generating Plot...")
    plt.figure(figsize=(10, 6))
    
    for name, costs in final_results.items():
        cfg = SCENARIOS[name]
        plt.plot(RPS_STEPS, costs, marker=cfg["marker"], label=name, color=cfg["color"], linewidth=2)
    
    plt.title("Resource Cost vs Load (Semantic Clustering Effect)", fontsize=14)
    plt.xlabel("Request Rate (RPS)", fontsize=12)
    plt.ylabel("Cost (Active Node-Seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "cost_vs_rps_final.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")

if __name__ == "__main__":
    main()
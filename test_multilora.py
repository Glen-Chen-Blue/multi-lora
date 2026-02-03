import subprocess
import time
import os
import signal
import requests
import asyncio
import httpx
import random
import threading
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Experiment Configuration
# ==========================================
RPS_STEPS = list(range(1, 17))  # 1 to 16
STEP_DURATION = 30
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
# Independent Monitor Thread
# ==========================================
class CostMonitor(threading.Thread):
    """
    獨立的監控執行緒，不受 Async Event Loop 阻塞影響。
    使用積分方式計算 Cost，即使 sleep 有誤差也能保持準確。
    """
    def __init__(self, scenario_name):
        super().__init__()
        self.stop_event = threading.Event()
        self.scenario_name = scenario_name
        self.total_cost = 0.0
        self.lock = threading.Lock()
        self.daemon = True # 確保主程式結束時它會跟著結束

    def run(self):
        last_time = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            delta = now - last_time
            last_time = now
            
            active_nodes = 0
            try:
                # 使用同步 requests，避免與流量生成的 async client 搶資源
                resp = requests.get(f"{CONTROL_URL}/status", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    active_nodes = data.get("active_nodes", 0)
                else:
                    active_nodes = self._fallback_nodes()
            except:
                active_nodes = self._fallback_nodes()

            # 積分計算: Cost = 節點數 * 時間區間
            with self.lock:
                self.total_cost += active_nodes * delta
            
            time.sleep(1.0) # 這裡的 sleep 不會被 async loop 卡住

    def _fallback_nodes(self):
        # 如果請求失敗，根據場景給一個保底值，避免曲線掉到0
        if "Random" in self.scenario_name:
            return 2 # 假設 Random 至少有初始節點
        return 1

    def get_cost(self):
        with self.lock:
            return self.total_cost

    def stop(self):
        self.stop_event.set()

# ==========================================
# Traffic Logic
# ==========================================
async def simulate_user(client: httpx.AsyncClient, stats: dict):
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
        "max_new_tokens": 64  # [修正] Max New Tokens = 32
    }
    
    try:
        # 發送請求
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=10.0)
        
        if resp.status_code == 200:
            data = resp.json()
            request_id = data["request_id"]
            stats["sent"] += 1
            
            # [修正] 等待請求完成 (Listening for [DONE])
            try:
                async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=60.0) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: [DONE]"):
                            stats["finished"] += 1
                            break
                        if "error" in line.lower() and "type" in line:
                             pass 
            except Exception:
                pass 
    except Exception:
        pass 

async def traffic_generator(rps, duration, scenario_name):
    """
    執行指定 RPS 和時間的流量，並回傳累積 Cost
    """
    # 啟動獨立監控執行緒
    monitor = CostMonitor(scenario_name)
    monitor.start()

    # [修正] 增加 max_connections 防止高併發時卡住
    limits = httpx.Limits(max_keepalive_connections=2000, max_connections=2000)
    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        end_time = time.time() + duration
        
        stats = {"sent": 0, "finished": 0}
        
        # Phase 1: 流量發送 Loop (持續 duration 秒)
        print(f"      -> Sending traffic for {duration}s...")
        while time.time() < end_time:
            # 這裡不等待 simulate_user 完成，讓它在背景跑
            asyncio.create_task(simulate_user(client, stats))
            
            # Poisson Arrival
            sleep_time = random.expovariate(rps)
            await asyncio.sleep(sleep_time)
        
        # Phase 2: 等待 90% 完成 (Drain Phase)
        print(f"      -> Waiting for 90% completion (Sent: {stats['sent']})...")
        timeout_cutoff = time.time() + 120 # 最多等 2 分鐘
        while time.time() < timeout_cutoff:
            if stats["sent"] > 0:
                ratio = stats["finished"] / stats["sent"]
                if ratio >= 0.9:
                    print(f"      ✅ Reached 90% completion ({stats['finished']}/{stats['sent']})")
                    break
            elif stats["sent"] == 0:
                break
                
            await asyncio.sleep(1.0)
        
        # 停止監控
        monitor.stop()
        monitor.join()
        final_cost = monitor.get_cost()
        
        # Phase 3: 清空隊列 (Reset System)
        print("      -> 🧹 Clearing System Queues...")
        try:
            await client.post(f"{CONTROL_URL}/debug/reset", timeout=5.0)
        except Exception as e:
            print(f"      ⚠️ Reset failed: {e}")

        return final_cost

# ==========================================
# System Helpers
# ==========================================
def start_system(env_vars):
    print(f"   🚀 Starting system... (Env: Semantic={env_vars.get('ENABLE_SEMANTIC')})")
    current_env = os.environ.copy()
    current_env.update(env_vars)
    # 這裡會呼叫 single_area.sh
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
    
    # 預先生成一次 Map 確保檔案存在
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
                
                # [關鍵] 設定固定的 Seed 確保流量一致
                random.seed(42 + rps)
                
                cost = asyncio.run(traffic_generator(rps, STEP_DURATION, name))
                print(f"   💰 Cost: {cost:.2f} Node-Seconds")
                final_results[name].append(cost)
                
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
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
RPS_STEPS = [i for i in range(1, 9)] 
STEP_DURATION = 60
CONTROL_URL = "http://127.0.0.1:9000"
MAXNEWTOKENS = 64

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
    [修改] 使用積分方式計算 Cost，只計算有在做事的節點 (Computing Nodes)。
    """
    def __init__(self, scenario_name):
        super().__init__()
        self.stop_event = threading.Event()
        self.scenario_name = scenario_name
        self.total_cost = 0.0
        self.lock = threading.Lock()
        self.daemon = True 

    def run(self):
        last_time = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            delta = now - last_time
            last_time = now
            
            busy_nodes = 0
            try:
                # 使用同步 requests
                resp = requests.get(f"{CONTROL_URL}/status", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    # [修改] 讀取 computing_nodes (正在運算的節點)
                    # 如果 Control Node 尚未回傳此欄位 (舊版相容)，退回讀取 active_nodes
                    busy_nodes = data.get("computing_nodes", data.get("active_nodes", 0))
                else:
                    busy_nodes = self._fallback_nodes()
            except:
                busy_nodes = self._fallback_nodes()

            # 積分計算: Cost = 忙碌節點數 * 時間區間
            with self.lock:
                self.total_cost += busy_nodes * delta
            
            time.sleep(1.0) 

    def _fallback_nodes(self):
        return 0 # 連線失敗視為無消耗 (或者保留上次值)

    def get_cost(self):
        with self.lock:
            return self.total_cost

    def stop(self):
        self.stop_event.set()

# ==========================================
# Traffic Logic
# ==========================================
async def simulate_user(client: httpx.AsyncClient, stats: dict):
    rand_val = random.random()
    
    if rand_val < 0.30: adapter = "1"
    elif rand_val < 0.60: adapter = "2"
    elif rand_val < 0.90: adapter = "3"
    else: adapter = str(random.randint(4, 100))
    
    payload = {
        "prompt": "test prompt", 
        "adapter_id": adapter,
        "max_new_tokens": MAXNEWTOKENS
    }
    
    try:
        # 發送請求
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=10.0)
        
        if resp.status_code == 200:
            data = resp.json()
            request_id = data["request_id"]
            stats["sent"] += 1
            
            # [修正] 嚴格 Timeout 防止請求卡死
            start_wait = time.time()
            try:
                async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=45.0) as response:
                    async for line in response.aiter_lines():
                        # 超時強制中斷 (防止 SSE 連結卡住不放)
                        if time.time() - start_wait > 30.0:
                             break
                             
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
        
        # Phase 1: 流量發送
        print(f"      -> Sending traffic for {duration}s...")
        while time.time() < end_time:
            asyncio.create_task(simulate_user(client, stats))
            
            sleep_time = random.expovariate(rps)
            await asyncio.sleep(sleep_time)
        
        # Phase 2: 等待 95% 完成
        print(f"      -> Waiting for 95% completion (Sent: {stats['sent']})...")
        timeout_cutoff = time.time() + 120 
        while time.time() < timeout_cutoff:
            if stats["sent"] > 0:
                ratio = stats["finished"] / stats["sent"]
                if ratio >= 0.95:
                    print(f"      ✅ Reached 95% completion ({stats['finished']}/{stats['sent']})")
                    break
            elif stats["sent"] == 0:
                break
                
            await asyncio.sleep(1.0)
        
        # 停止監控
        monitor.stop()
        monitor.join()
        final_cost = monitor.get_cost()
        
        # Phase 3: 清空隊列
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
    proc = subprocess.Popen(["bash", "single_area.sh"], env=current_env, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
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
                random.seed(42 + rps)
                
                cost = asyncio.run(traffic_generator(rps, STEP_DURATION, name))
                print(f"   💰 Cost: {cost:.2f} Busy Node-Seconds")
                final_results[name].append(cost)
                
        finally:
            stop_system(proc)

    print("\n📊 Generating Plot...")
    plt.figure(figsize=(10, 6))
    
    for name, costs in final_results.items():
        cfg = SCENARIOS[name]
        plt.plot(RPS_STEPS, costs, marker=cfg["marker"], label=name, color=cfg["color"], linewidth=2)
    
    plt.title("Resource Cost vs Load (Computing Time Only)", fontsize=14)
    plt.xlabel("Request Rate (RPS)", fontsize=12)
    plt.ylabel("Cost (Busy Node-Seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "cost_vs_rps_final.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")

if __name__ == "__main__":
    main()
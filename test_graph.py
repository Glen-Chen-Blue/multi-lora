import asyncio
import httpx
import random
import time
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# Benchmark Configuration
# ==========================================
CONTROL_URL = "http://localhost:9000"
ADAPTERS = ["1", "2", "3", "chat", "math", "code"]
TRAFFIC_PATTERN = "1"
TARGET_ADAPTER = "1"
MAX_NEW_TOKENS = 128

# 實驗參數設定
RPS_STEPS = [4, 8, 12, 16, 20]  # 要測試的 RPS 級距
TEST_DURATION_SEC = 60               # 每個 RPS 級距測試幾秒 (建議 60s 以上以觀察長期穩定性)
COOLDOWN_SEC = 30                    # 級距間的冷卻時間 (讓系統縮編或清空 Queue)

# ==========================================
# Utilities
# ==========================================
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

PROMPTS = ["test"]

def format_alpaca_prompt(user_prompt):
    return f"### Instruction:\n{user_prompt}\n\n### Response:\n"

# ==========================================
# Core Load Test Logic (Encapsulated)
# ==========================================
async def simulate_user(client: httpx.AsyncClient, req_id_seq: int, adapter_prob: float, target_rps: float, stats: dict, ttft_records: list):
    """
    模擬單一使用者的請求行為
    """
    # 決定 Adapter
    if TRAFFIC_PATTERN == "1":
        if random.random() < adapter_prob:
            adapter = TARGET_ADAPTER
        else:
            others = [a for a in ADAPTERS if a != TARGET_ADAPTER]
            adapter = TARGET_ADAPTER if not others else random.choice(others)
    else:
        adapter = random.choice(ADAPTERS)

    raw_prompt = random.choice(PROMPTS)
    formatted_prompt = format_alpaca_prompt(raw_prompt)
    
    payload = {
        "prompt": formatted_prompt, 
        "adapter_id": adapter,
        "max_new_tokens": MAX_NEW_TOKENS
    }
    
    start_ts = time.time()
    ttft = 0.0 
    
    try:
        # 發送請求
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        request_id = data["request_id"]
        stats["sent"] += 1

        # 接收串流
        async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=120.0) as response:
            async for line in response.aiter_lines():
                if not line: continue
                if line.startswith("data: [DONE]"): break
                
                if line.startswith("data:"):
                    raw_content = line[len("data:"):].rstrip("\n")
                    if raw_content.strip() == "ok": continue 

                    # 記錄 TTFT
                    if ttft == 0.0:
                        ttft = time.time() - start_ts

                    # 檢查錯誤
                    try:
                        content = json.loads(raw_content)
                    except json.JSONDecodeError:
                        content = raw_content
                    
                    if isinstance(content, dict) and content.get("type") == "error":
                        print(f"{RED}[Error] {content['message']}{RESET}")
                        if "aborted by system merge" in content.get("message", ""):
                             stats["errors"] += 1
                             return # 這裡選擇不中斷整個測試，只記為錯誤
        
        elapsed = time.time() - start_ts
        stats["finished"] += 1
        
        final_ttft = ttft if ttft > 0 else elapsed
        ttft_records.append(final_ttft)

    except Exception as e:
        stats["errors"] += 1
        # print(f"{RED}[ERROR] Req #{req_id_seq} Failed: {repr(e)}{RESET}")

async def run_test_phase(target_rps: int):
    """
    執行單一 RPS 階段的測試
    """
    total_requests = int(target_rps * TEST_DURATION_SEC)
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}🚀 Starting Phase: {target_rps} RPS{RESET}")
    print(f"{CYAN}   Duration: {TEST_DURATION_SEC}s | Total Requests: {total_requests}{RESET}")
    print(f"{CYAN}=================================================={RESET}")

    stats = {"sent": 0, "finished": 0, "errors": 0}
    ttft_records = []
    background_tasks = set()
    
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=200)

    async with httpx.AsyncClient(limits=limits) as client:
        # 檢查連線
        try:
            await client.get(f"{CONTROL_URL}/status")
        except Exception:
            print(f"{RED}❌ Cannot connect to Control Node. Is it running?{RESET}")
            return None

        start_time = time.time()
        
        # 根據 RPS 發送請求
        for i in range(1, total_requests + 1):
            task = asyncio.create_task(simulate_user(client, i, 0.8, target_rps, stats, ttft_records))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

            # 控制發送頻率 (Poisson Process)
            sleep_time = random.expovariate(target_rps)
            await asyncio.sleep(sleep_time)

        # 等待剩餘任務完成
        print(f"{YELLOW}⏳ Waiting for remaining tasks to finish...{RESET}")
        if background_tasks:
            await asyncio.wait(background_tasks)
            
    # 計算統計數據
    if ttft_records:
        avg_ttft = sum(ttft_records) / len(ttft_records)
        sorted_ttft = sorted(ttft_records)
        p95_ttft = sorted_ttft[int(len(sorted_ttft) * 0.95)] if len(sorted_ttft) > 0 else 0
    else:
        avg_ttft = 0
        p95_ttft = 0
        
    print(f"{GREEN}✅ Phase Done. Sent: {stats['sent']}, Errors: {stats['errors']}")
    print(f"   Avg TTFT: {avg_ttft:.4f}s | P95 TTFT: {p95_ttft:.4f}s{RESET}")
    
    return {
        "rps": target_rps,
        "avg_ttft": avg_ttft,
        "p95_ttft": p95_ttft,
        "error_rate": stats["errors"] / total_requests if total_requests > 0 else 0
    }

# ==========================================
# Main Benchmark Loop
# ==========================================
async def main():
    results = []

    print(f"=== 🔥 Multi-LoRA Scalability Benchmark 🔥 ===")
    print(f"Steps (RPS): {RPS_STEPS}")
    print(f"Phase Duration: {TEST_DURATION_SEC}s")
    
    for rps in RPS_STEPS:
        res = await run_test_phase(rps)
        if res:
            results.append(res)
        
        if rps != RPS_STEPS[-1]:
            print(f"{YELLOW}❄️  Cooling down for {COOLDOWN_SEC}s...{RESET}")
            await asyncio.sleep(COOLDOWN_SEC)

    # 輸出最終結果
    print("\n\n=== 📊 Benchmark Summary ===")
    print(f"{'RPS':<10} {'Avg TTFT':<15} {'P95 TTFT':<15} {'Error Rate':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['rps']:<10} {r['avg_ttft']:<15.4f} {r['p95_ttft']:<15.4f} {r['error_rate']:<15.2%}")

    # 繪圖
    plot_results(results)

def plot_results(results):
    try:
        rps_list = [r['rps'] for r in results]
        avg_ttft_list = [r['avg_ttft'] for r in results]
        p95_ttft_list = [r['p95_ttft'] for r in results]

        plt.figure(figsize=(10, 6))
        
        plt.plot(rps_list, avg_ttft_list, marker='o', label='Average TTFT', color='b', linewidth=2)
        plt.plot(rps_list, p95_ttft_list, marker='x', label='P95 TTFT', color='r', linestyle='--', linewidth=2)
        
        plt.title(f'Latency vs Load (Duration per step: {TEST_DURATION_SEC}s)')
        plt.xlabel('Requests Per Second (RPS)')
        plt.ylabel('Time To First Token (s)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_file = 'benchmark_rps_ttft.png'
        plt.savefig(output_file)
        print(f"\n{GREEN}📈 Plot saved to {output_file}{RESET}")
        
    except Exception as e:
        print(f"{RED}Failed to plot results: {e}{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
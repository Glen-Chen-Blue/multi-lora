import asyncio
import httpx
import random
import time
import sys
import json 
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
CONTROL_URL = "http://localhost:9000"

# 請確認這些 Adapter ID 與資料夾名稱一致 (大小寫敏感)
# 如果您之前遇到 400 錯誤，請確保這些 ID 在 ./testLoRA 中真的存在
ADAPTERS = ["1", "2", "3", "chat", "math", "code"] 

# [新增] 流量分佈模式切換
# "uniform" : 所有 Adapter 機率均等 (原本的模式)
# "skewed"  : 集中流量，TARGET_ADAPTER 佔 80%，其他平分 20%
TRAFFIC_PATTERN = "skewed"  # <--- 修改這裡切換模式
TARGET_ADAPTER = "chat"     # <--- 設定 "skewed" 模式下的熱點 Adapter

TOTAL_REQUESTS = 100
AVG_RPS = 30.0 
MAX_NEW_TOKENS = 128

# 真實的 Prompts
PROMPTS = [
    "Explain the theory of relativity in simple terms for a 5-year-old.",
    "Write a Python function to calculate the Fibonacci sequence using recursion.",
    "Solve the quadratic equation: x^2 - 5x + 6 = 0.",
    "What are the three laws of thermodynamics?",
    "Translate 'The quick brown fox jumps over the lazy dog' into French and Spanish.",
    "List 5 benefits of regular exercise and explain one in detail.",
    "Create a SQL query to find the second highest salary from an Employee table.",
    "Summarize the plot of 'Romeo and Juliet' in three sentences.",
    "Explain the difference between TCP and UDP protocols.",
    "Write a haiku about artificial intelligence.",
    "If I have 3 apples and you take away 2, how many do you have?",
    "Debug this code: `def add(a,b): return a * b` - it should add numbers.",
]

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
GREY = "\033[90m"

stats = {"sent": 0, "finished": 0, "errors": 0}
ttft_records = [] 

def format_alpaca_prompt(user_prompt):
    return (
        f"### Instruction:\n{user_prompt}\n\n"
        f"### Response:\n"
    )

async def simulate_user(client: httpx.AsyncClient, req_id_seq: int):
    # [修改] 根據 TRAFFIC_PATTERN 選擇 Adapter
    if TRAFFIC_PATTERN == "skewed":
        # 80% 機率選中 TARGET_ADAPTER
        if random.random() < 0.8:
            adapter = TARGET_ADAPTER
        else:
            # 20% 機率從剩下的裡面選
            others = [a for a in ADAPTERS if a != TARGET_ADAPTER]
            if not others: # 防呆：如果只有一個 Adapter
                adapter = TARGET_ADAPTER
            else:
                adapter = random.choice(others)
    else:
        # 均勻分佈
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
    full_response_text = []
    
    try:
        # 1. 發送請求
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        request_id = data["request_id"]
        
        stats["sent"] += 1
        short_prompt = (raw_prompt[:30] + '..') if len(raw_prompt) > 30 else raw_prompt
        print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq}/{TOTAL_REQUESTS} SENT -> {adapter} (ID: {request_id[:8]}...) Q: {short_prompt}{RESET}")

        # 2. 接收串流
        async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=120.0) as response:
            async for line in response.aiter_lines():
                if not line: continue

                if line.startswith("data: [DONE]"):
                    break
                
                if line.startswith("data:"):
                    raw_content = line[len("data:"):].rstrip("\n")
                    if raw_content.strip() == "ok": continue 

                    try:
                        content = json.loads(raw_content)
                    except json.JSONDecodeError:
                        content = raw_content

                    if isinstance(content, str) and (content.startswith("[ERROR]") or "Processing aborted" in content):
                         full_response_text.append(f"{RED}{content}{RESET}")
                         continue

                    if ttft == 0.0: ttft = time.time() - start_ts
                    full_response_text.append(str(content))

        elapsed = time.time() - start_ts
        stats["finished"] += 1
        
        final_ttft = ttft if ttft > 0 else elapsed
        ttft_records.append(final_ttft)

        answer = "".join(full_response_text).strip()
        
        print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq} DONE <- {adapter} (Time: {elapsed:.2f}s, TTFT: {final_ttft:.2f}s){RESET}")
        
        if answer:
            preview = (answer) if len(answer) > 100 else answer
            # print(f"    {GREY}>> {preview.replace(chr(10), ' ')}{RESET}")
        else:
            if elapsed < 0.1:
                print(f"    {RED}>> [Request Failed Immediately - Check Server Logs]{RESET}")
            else:
                print(f"    {RED}>> [No Output generated]{RESET}")

    except Exception as e:
        stats["errors"] += 1
        print(f"{RED}[ERROR] #{req_id_seq} Failed: {repr(e)}{RESET}")

async def main():
    print(f"=== Traffic Simulator ===")
    print(f"Mode: {TRAFFIC_PATTERN.upper()}")
    if TRAFFIC_PATTERN == "skewed":
        print(f"Target: {TARGET_ADAPTER} (80%)")
    
    background_tasks = set()
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=200)
    
    async with httpx.AsyncClient(limits=limits) as client:
        try:
            await client.get(f"{CONTROL_URL}/status")
        except Exception as e:
            print(f"{RED}Cannot connect to Control Node: {e}{RESET}")
            return

        try:
            for i in range(1, TOTAL_REQUESTS + 1):
                task = asyncio.create_task(simulate_user(client, i))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

                if i < TOTAL_REQUESTS:
                    sleep_time = random.expovariate(AVG_RPS)
                    await asyncio.sleep(sleep_time)

            print(f"\n{YELLOW}=== Waiting for remaining tasks... ==={RESET}")
            if background_tasks:
                await asyncio.wait(background_tasks)

        except KeyboardInterrupt:
            print("\nStopping...")
            for t in background_tasks: t.cancel()

    print(f"\n=== Summary: Sent {stats['sent']} / Fin {stats['finished']} / Err {stats['errors']} ===")
    
    if ttft_records:
        avg_ttft = sum(ttft_records) / len(ttft_records)
        sorted_ttft = sorted(ttft_records)
        p95_index = int(len(sorted_ttft) * 0.95)
        if p95_index >= len(sorted_ttft):
            p95_index = len(sorted_ttft) - 1
        p95_ttft = sorted_ttft[p95_index]

        print(f"{CYAN}=== Latency Statistics ({TRAFFIC_PATTERN}) ==={RESET}")
        print(f"Average TTFT : {avg_ttft:.4f} s")
        print(f"P95 TTFT     : {p95_ttft:.4f} s")
    else:
        print(f"{RED}No successful requests to calculate stats.{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
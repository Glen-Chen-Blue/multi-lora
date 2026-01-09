import asyncio
import httpx
import random
import time
import sys
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
CONTROL_URL = "http://localhost:9000"
ADAPTERS = ["chat","chat","chat", "chat", "chat", "math", '1', '2'] 
TOTAL_REQUESTS = 50

# 如果你的電腦出現 Reaper 誤殺，可以稍微調低 RPS (例如 2.0 或 3.0)
AVG_RPS = 4.0 

MAX_NEW_TOKENS = 128
PROMPTS = ["Hello", "Test", "Math", "Code", "Joke"]

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

stats = {"sent": 0, "finished": 0, "errors": 0}

async def simulate_user(client: httpx.AsyncClient, req_id_seq: int):
    adapter = random.choice(ADAPTERS)
    prompt = random.choice(PROMPTS)
    
    payload = {
        "prompt": prompt,
        "adapter_id": adapter,
        "max_new_tokens": MAX_NEW_TOKENS
    }
    
    start_ts = time.time()
    ttft = 0.0 # [NEW] 初始化 TTFT
    
    try:
        # 1. 發送請求 [FIX] 修正 Endpoint 為 /add_request
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        request_id = data["request_id"]
        
        stats["sent"] += 1
        print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq}/{TOTAL_REQUESTS} SENT -> {adapter} (ID: {request_id[:8]}...){RESET}")

        # 2. 接收串流
        async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=120.0) as response:
            async for line in response.aiter_lines():
                if not line: continue

                # 檢查結束訊號
                if line.startswith("data: [DONE]"):
                    break
                
                # [NEW] 計算 TTFT
                # 邏輯：如果 ttft 還沒被記錄，且收到 data 開頭的行
                if ttft == 0.0 and line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    # 忽略握手訊號 "ok"，只記錄真正的 token
                    if content and content != "ok":
                        ttft = time.time() - start_ts

        elapsed = time.time() - start_ts
        stats["finished"] += 1
        
        # [MODIFIED] 增加 TTFT 顯示
        # 如果 ttft 還是 0 (例如瞬間完成或只有 handshake)，就顯示為 elapsed
        final_ttft = ttft if ttft > 0 else elapsed
        
        print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq} DONE <- {adapter} (Time: {elapsed:.2f}s, TTFT: {final_ttft:.2f}s){RESET}")

    except Exception as e:
        stats["errors"] += 1
        print(f"{RED}[ERROR] #{req_id_seq} Failed: {repr(e)}{RESET}")

async def main():
    print(f"=== Traffic Simulator ===")
    
    background_tasks = set()
    # 提高連線池上限，避免在高 RPS 下塞車
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
            # Cancel remaining tasks to exit cleanly
            for t in background_tasks: t.cancel()

    print(f"\n=== Summary: Sent {stats['sent']} / Fin {stats['finished']} / Err {stats['errors']} ===")

if __name__ == "__main__":
    asyncio.run(main())
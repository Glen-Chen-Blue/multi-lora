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

# [Modified] Use virtual IDs 1-100
ADAPTERS = [str(i) for i in range(1, 101)]

# 流量分佈模式
TRAFFIC_PATTERN = "1"  
TARGET_ADAPTER = "1"     

TOTAL_REQUESTS = 500
AVG_RPS = 100.0 

PROMPTS = ["test"]

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
GREY = "\033[90m"

stats = {"sent": 0, "finished": 0, "errors": 0}
ttft_records = [] 

resource_stats = {
    "node_seconds": 0.0,
    "max_nodes": 0,
    "samples": 0
}
is_test_running = True

def format_alpaca_prompt(user_prompt):
    return (
        f"### Instruction:\n{user_prompt}\n\n"
        f"### Response:\n"
    )

async def monitor_cluster_usage(client: httpx.AsyncClient):
    print(f"{YELLOW}[Monitor] Started tracking cluster resource usage...{RESET}")
    while is_test_running:
        try:
            start_check = time.time()
            resp = await client.get(f"{CONTROL_URL}/status", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                active_nodes = data.get("active_nodes", 0)
                resource_stats["node_seconds"] += active_nodes * 1.0 
                if active_nodes > resource_stats["max_nodes"]:
                    resource_stats["max_nodes"] = active_nodes
                resource_stats["samples"] += 1
            
            elapsed = time.time() - start_check
            sleep_time = max(0.0, 1.0 - elapsed)
            await asyncio.sleep(sleep_time)
            
        except Exception:
            await asyncio.sleep(1)

async def simulate_user(client: httpx.AsyncClient, req_id_seq: int):
    # Traffic Pattern: 80% to TARGET, 20% Uniform to others
    if TRAFFIC_PATTERN == "1":
        if random.random() < 0.8:
            adapter = TARGET_ADAPTER
        else:
            others = [a for a in ADAPTERS if a != TARGET_ADAPTER]
            adapter = random.choice(others) if others else TARGET_ADAPTER
    else:
        adapter = random.choice(ADAPTERS)

    raw_prompt = random.choice(PROMPTS)
    formatted_prompt = format_alpaca_prompt(raw_prompt)
    current_max_tokens = random.randint(64, 128)
    
    payload = {
        "prompt": formatted_prompt, 
        "adapter_id": adapter,
        "max_new_tokens": current_max_tokens
    }
    
    start_ts = time.time()
    ttft = 0.0 
    full_response_text = []
    
    try:
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        request_id = data["request_id"]
        
        stats["sent"] += 1
        short_prompt = (raw_prompt[:30] + '..') if len(raw_prompt) > 30 else raw_prompt
        print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq}/{TOTAL_REQUESTS} SENT -> {adapter} (T:{current_max_tokens}) (ID: {request_id[:8]}...){RESET}")

        async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=120.0) as response:
            async for line in response.aiter_lines():
                if not line: continue
                if line.startswith("data: [DONE]"): break
                
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

        token_count = len(full_response_text)
        print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq} DONE <- {adapter} (Time: {elapsed:.2f}s, TTFT: {final_ttft:.2f}s, Tokens: {token_count}){RESET}")
        
    except Exception as e:
        stats["errors"] += 1
        print(f"{RED}[ERROR] #{req_id_seq} Failed: {repr(e)}{RESET}")

async def main():
    global is_test_running
    print(f"=== Traffic Simulator (1-100 Virtual IDs) ===")
    print(f"Mode: {TRAFFIC_PATTERN.upper()}")
    
    background_tasks = set()
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=200)
    
    async with httpx.AsyncClient(limits=limits) as client:
        try:
            await client.get(f"{CONTROL_URL}/status")
        except Exception as e:
            print(f"{RED}Cannot connect to Control Node: {e}{RESET}")
            return

        monitor_task = asyncio.create_task(monitor_cluster_usage(client))

        try:
            start_time = time.time()
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
        finally:
            is_test_running = False
            await monitor_task
            total_duration = time.time() - start_time

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
        
        node_seconds = resource_stats['node_seconds']
        avg_nodes = node_seconds / total_duration if total_duration > 0 else 0
        
        print(f"\n{YELLOW}=== Resource Consumption Stats ==={RESET}")
        print(f"Test Duration        : {total_duration:.2f} s")
        print(f"Total Resource Usage : {node_seconds:.2f} Node-Seconds")
        print(f"Average Active Nodes : {avg_nodes:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
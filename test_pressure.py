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

ADAPTERS = ["1", "2", "3", "chat", "math", "code"] 

# 流量分佈模式
TRAFFIC_PATTERN = "skewed"  
TARGET_ADAPTER = "chat"     

TOTAL_REQUESTS = 100
AVG_RPS = 15.0 
MAX_NEW_TOKENS = 128

PROMPTS = ["test"]

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
GREY = "\033[90m"

stats = {"sent": 0, "finished": 0, "errors": 0}
ttft_records = [] 

# [新增] 資源監控統計
resource_stats = {
    "node_seconds": 0.0,      # 累積的 (節點 * 秒數)
    "max_nodes": 0,           # 曾達到的最大節點數
    "samples": 0              # 取樣次數
}
is_test_running = True        # 控制監控迴圈的旗標

def format_alpaca_prompt(user_prompt):
    return (
        f"### Instruction:\n{user_prompt}\n\n"
        f"### Response:\n"
    )

async def monitor_cluster_usage(client: httpx.AsyncClient):
    """
    [新增] 背景任務：每秒查詢一次 Control Node，計算資源消耗積分
    """
    print(f"{YELLOW}[Monitor] Started tracking cluster resource usage...{RESET}")
    while is_test_running:
        try:
            start_check = time.time()
            resp = await client.get(f"{CONTROL_URL}/status", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                active_nodes = data.get("active_nodes", 0)
                
                # 積分計算：假設這 1 秒內節點數維持不變 (Riemann Sum)
                # 這裡簡單使用採樣間隔作為權重
                resource_stats["node_seconds"] += active_nodes * 1.0 
                
                # 更新最大值
                if active_nodes > resource_stats["max_nodes"]:
                    resource_stats["max_nodes"] = active_nodes
                
                resource_stats["samples"] += 1
            
            # 補償延遲，盡量維持 1.0 秒的採樣頻率
            elapsed = time.time() - start_check
            sleep_time = max(0.0, 1.0 - elapsed)
            await asyncio.sleep(sleep_time)
            
        except Exception:
            # 忽略監控錯誤，不影響主測試
            await asyncio.sleep(1)

async def simulate_user(client: httpx.AsyncClient, req_id_seq: int):
    if TRAFFIC_PATTERN == "skewed":
        if random.random() < 0.8:
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
    full_response_text = []
    
    try:
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        request_id = data["request_id"]
        
        stats["sent"] += 1
        short_prompt = (raw_prompt[:30] + '..') if len(raw_prompt) > 30 else raw_prompt
        print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq}/{TOTAL_REQUESTS} SENT -> {adapter} (ID: {request_id[:8]}...) Q: {short_prompt}{RESET}")

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

        answer = "".join(full_response_text).strip()
        print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] #{req_id_seq} DONE <- {adapter} (Time: {elapsed:.2f}s, TTFT: {final_ttft:.2f}s){RESET}")
        print(f"{GREY}--- Response Start ---{RESET}\n{answer}\n{GREY}--- Response End ---{RESET}")

    except Exception as e:
        stats["errors"] += 1
        print(f"{RED}[ERROR] #{req_id_seq} Failed: {repr(e)}{RESET}")

async def main():
    global is_test_running
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

        # [新增] 啟動資源監控
        monitor_task = asyncio.create_task(monitor_cluster_usage(client))

        try:
            start_time = time.time()
            
            # 發送請求迴圈
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
            # [新增] 停止資源監控
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
        
        # [新增] 資源消耗報告
        node_seconds = resource_stats['node_seconds']
        avg_nodes = node_seconds / total_duration if total_duration > 0 else 0
        
        print(f"\n{YELLOW}=== Resource Consumption Stats ==={RESET}")
        print(f"Test Duration        : {total_duration:.2f} s")
        print(f"Total Resource Usage : {node_seconds:.2f} Node-Seconds")
        print(f"Average Active Nodes : {avg_nodes:.2f}")
        print(f"Peak Active Nodes    : {resource_stats['max_nodes']}")
        print(f"Cost Estimate        : {node_seconds / 3600:.4f} Node-Hours")

    else:
        print(f"{RED}No successful requests to calculate stats.{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
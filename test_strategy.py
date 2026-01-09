import asyncio
import httpx
import json
import time
import sys
from typing import List, Dict

# ==========================================
# Configuration
# ==========================================
CONTROL_NODE_URL = "http://localhost:9000"

# 請修改這裡以符合你 ./testLoRA 資料夾中的實際名稱
ADAPTER_A = "chat"  
ADAPTER_B = "math" 

# 顏色輸出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def log(msg, color=RESET):
    print(f"{color}[TEST] {msg}{RESET}")

async def get_system_status(client: httpx.AsyncClient) -> Dict:
    try:
        resp = await client.get(f"{CONTROL_NODE_URL}/status")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log(f"Failed to get status: {e}", RED)
        return {}

async def run_inference(client: httpx.AsyncClient, prompt: str, adapter_id: str, tokens: int = 20, verbose: bool = True) -> str:
    """
    發送請求並透過 SSE 接收結果
    """
    start_time = time.time()
    payload = {
        "prompt": prompt,
        "adapter_id": adapter_id,
        "max_new_tokens": tokens
    }

    # 1. Submit Request
    try:
        resp = await client.post(f"{CONTROL_NODE_URL}/send_request", json=payload, timeout=5.0)
        resp.raise_for_status()
        request_id = resp.json()["request_id"]
        if verbose:
            log(f"Request submitted. ID: {request_id} (Adapter: {adapter_id})", CYAN)
    except Exception as e:
        log(f"Submission failed: {e}", RED)
        return ""

    # 2. Listen to SSE Stream
    full_text = ""
    try:
        async with client.stream("GET", f"{CONTROL_NODE_URL}/stream/{request_id}", timeout=60.0) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content == "[DONE]":
                        break
                    if content == "ok":
                        continue
                    if content == "[ERROR]":
                        log(f"Stream Error for {request_id}", RED)
                        break
                    full_text += content
    except Exception as e:
        log(f"Streaming failed: {e}", RED)
        return ""

    elapsed = time.time() - start_time
    if verbose:
        log(f"Finished {request_id} in {elapsed:.2f}s. Output len: {len(full_text)}", GREEN)
    return full_text

# ==========================================
# Test Cases
# ==========================================

async def test_health_check(client: httpx.AsyncClient):
    log("=== TEST 1: System Health Check ===", YELLOW)
    status = await get_system_status(client)
    
    healthy_nodes = status.get("healthy_nodes", [])
    log(f"Healthy Nodes: {len(healthy_nodes)}")
    
    if len(healthy_nodes) == 0:
        log("FAIL: No healthy compute nodes found!", RED)
        sys.exit(1)
    
    for node in healthy_nodes:
        node_info = status["nodes"][node]
        log(f"Node {node}: Mode={node_info.get('mode')} | Load={node_info.get('metrics', {}).get('load', {})}")
    
    log("PASS: Health Check", GREEN)

async def test_basic_inference(client: httpx.AsyncClient):
    log("\n=== TEST 2: Basic Single Inference ===", YELLOW)
    prompt = "Explain quantum computing briefly."
    
    result = await run_inference(client, prompt, ADAPTER_A, tokens=30)
    
    if len(result) > 5:
        log(f"Output snippet: {result[:50]}...", CYAN)
        log("PASS: Basic Inference", GREEN)
    else:
        log("FAIL: No output generated", RED)

async def test_multi_lora_concurrency(client: httpx.AsyncClient):
    log("\n=== TEST 3: Multi-LoRA Concurrency (Interleaved) ===", YELLOW)
    # 同時發送 Adapter A 和 Adapter B 的請求
    tasks = []
    log(f"Sending 2 requests for {ADAPTER_A} and 2 for {ADAPTER_B} simultaneously...")
    
    tasks.append(run_inference(client, "Q1", ADAPTER_A, 20, False))
    tasks.append(run_inference(client, "Q2", ADAPTER_B, 20, False))
    tasks.append(run_inference(client, "Q3", ADAPTER_A, 20, False))
    tasks.append(run_inference(client, "Q4", ADAPTER_B, 20, False))
    
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if len(r) > 0)
    if success_count == 4:
        log("PASS: All concurrent Multi-LoRA requests finished.", GREEN)
    else:
        log(f"FAIL: Only {success_count}/4 requests finished.", RED)

async def test_merge_mechanism(client: httpx.AsyncClient):
    log("\n=== TEST 4: Dynamic Merge Triggering (Stress Test) ===", YELLOW)
    
    # 為了觸發 Merge，我們需要讓 Queue 積壓
    # 條件通常是: Queue > QMIN_MULT * Node_Count
    # 預設 QMIN_MULT=4, 若有 2 個節點，需要 > 8 個請求排隊
    
    req_count = 20
    log(f"Spamming {req_count} requests for {ADAPTER_A} to trigger merge...", CYAN)
    
    # 1. 啟動非同步請求轟炸
    tasks = []
    for i in range(req_count):
        # 使用較長的 token 生成讓 GPU 忙碌，確保 Queue 堆積
        tasks.append(run_inference(client, f"Stress test {i}", ADAPTER_A, tokens=64, verbose=False))
    
    # 不要等待全部完成，先讓它們在背景跑，我們同時監控狀態
    futures = asyncio.gather(*tasks)
    
    # 2. 監控狀態變化
    merge_triggered = False
    merged_node_url = None
    
    for i in range(20): # 監控 10 秒 (0.5s * 20)
        status = await get_system_status(client)
        assignments = status.get("merged_assignment", {})
        
        # 檢查 EFO/Control Node 是否指派了 Merge
        if ADAPTER_A in assignments:
            merged_node_url = assignments[ADAPTER_A]
            log(f"Control Node assigned MERGE for {ADAPTER_A} to {merged_node_url}", GREEN)
            
            # 進一步檢查該 Node 的實際狀態
            node_status = status["nodes"].get(merged_node_url, {})
            mode = node_status.get("mode")
            actual_merged = node_status.get("metrics", {}).get("lora_state", {}).get("merged_adapter")
            
            log(f"Status poll {i}: Node Mode={mode}, Actual Merged={actual_merged}", CYAN)
            
            if mode == "MERGED" and actual_merged == ADAPTER_A:
                merge_triggered = True
                log(">>> SUCCESS: Node successfully entered MERGED mode!", GREEN)
                break
        else:
            log(f"Status poll {i}: Waiting for merge trigger... (Queue total: {status.get('queue_total')})", CYAN)
        
        await asyncio.sleep(0.5)
        
    # 等待所有請求完成
    await futures
    
    if merge_triggered:
        log("PASS: Merge Mechanism Verified", GREEN)
    else:
        log("FAIL: Merge did not trigger within timeout. (Queue might have drained too fast)", RED)
        # 注意：如果你的 GPU 非常快或者網路延遲高，可能在 polling 到之前就處理完了

    # 3. 驗證 Unmerge (Drain)
    log("\n=== TEST 5: Idle Unmerge Check ===", YELLOW)
    log("Waiting for system to idle and auto-unmerge...", CYAN)
    
    for i in range(20):
        status = await get_system_status(client)
        assignments = status.get("merged_assignment", {})
        
        if ADAPTER_A not in assignments:
            log("System unmerged successfully (Assignment cleared).", GREEN)
            return
            
        await asyncio.sleep(1.0)
        
    log("WARN: System did not unmerge automatically (this assumes check logic in scheduler)", RED)


async def main():
    async with httpx.AsyncClient() as client:
        # 0. 檢查連線
        try:
            await client.get(f"{CONTROL_NODE_URL}/status")
        except:
            log(f"Cannot connect to {CONTROL_NODE_URL}. Is the server running?", RED)
            return

        await test_health_check(client)
        await test_basic_inference(client)
        await test_multi_lora_concurrency(client)
        await test_merge_mechanism(client)
        
        log("\nAll Tests Completed.", GREEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
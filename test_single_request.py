import asyncio
import httpx
import time
import random
import json
import logging
from datetime import datetime
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
CONTROL_URL = "http://127.0.0.1:9000"
TEST_DURATION = 30       # 測試持續時間 (秒)
RPS = 16                  # 每秒請求數 (壓力不用太大，重點是抓錯)
TIMEOUT_THRESHOLD = 20.0 # 超過幾秒視為卡住 (Stuck)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Debugger")

# Stats
stats = {
    "sent": 0,
    "success": 0,
    "error": 0,
    "stuck": 0,
    "stuck_details": []
}

async def get_system_snapshot(client: httpx.AsyncClient):
    """
    當發生錯誤時，抓取系統當下的詳細狀態
    """
    try:
        # 1. Get Control Node Status
        resp = await client.get(f"{CONTROL_URL}/status", timeout=2.0)
        cn_data = resp.json()
        
        logger.error(f"\n{'='*20} [SYSTEM SNAPSHOT] {'='*20}")
        logger.error(f"Control Node Queues: {json.dumps(cn_data.get('queues', {}), indent=2)}")
        logger.error(f"Active Nodes: {cn_data.get('active_nodes')}, Busy Nodes: {cn_data.get('computing_nodes')}")
        
        # 2. (Optional) 如果你知道 Compute Node IP，也可以嘗試抓 Compute Node Metrics
        # 這裡假設 Control Node status 內沒有包含詳細 Compute Node URL，如果有可以遍歷抓取
        
        logger.error(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Failed to take snapshot: {e}")

async def send_monitored_request(client: httpx.AsyncClient, req_index: int):
    """
    發送單一請求並全程監控
    """
    # 模擬 80% 熱門模型, 20% 冷門模型 (容易觸發 Loading/Unloading)
    if random.random() < 0.8:
        adapter_id = random.choice(["1", "2", "3"]) # 假設這些是常用的
    else:
        adapter_id = f"{random.randint(4, 10)}"

    payload = {
        "prompt": f"Debug req {req_index}",
        "adapter_id": adapter_id,
        "max_new_tokens": 32
    }
    
    start_time = time.time()
    req_id = "UNKNOWN"
    
    try:
        # Phase 1: 發送請求
        resp = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=5.0)
        
        if resp.status_code != 200:
            logger.error(f"❌ [Req {req_index}] Submission Failed: {resp.status_code}")
            stats["error"] += 1
            return

        data = resp.json()
        req_id = data.get("request_id")
        stats["sent"] += 1
        
        # Phase 2: 等待 SSE 串流
        # 我們設定一個比一般 Timeout 短一點的閥值來判定是否 "Stuck"
        async with client.stream("GET", f"{CONTROL_URL}/stream/{req_id}", timeout=TIMEOUT_THRESHOLD + 5) as response:
            first_token_time = None
            
            async for line in response.aiter_lines():
                now = time.time()
                elapsed = now - start_time
                
                # 檢查是否已經逾時
                if elapsed > TIMEOUT_THRESHOLD:
                    logger.error(f"⚠️ [Req {req_index} | {req_id}] STUCK! Elapsed: {elapsed:.2f}s (Adapter: {adapter_id})")
                    stats["stuck"] += 1
                    stats["stuck_details"].append(req_id)
                    
                    # 觸發快照 (只觸發一次，避免洗版)
                    if stats["stuck"] == 1: 
                        await get_system_snapshot(client)
                    
                    # 這裡不 break，繼續看它會不會死透，或者你可以選擇 break
                    # break 
                
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    
                    if content == "[DONE]":
                        total_time = time.time() - start_time
                        if total_time > TIMEOUT_THRESHOLD:
                             logger.warning(f"🐢 [Req {req_index}] Finished but SLOW. Time: {total_time:.2f}s")
                        # else:
                        #      logger.info(f"✅ [Req {req_index}] Finished in {total_time:.2f}s")
                        stats["success"] += 1
                        return
                        
                    if not first_token_time and content != "ok":
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        # logger.info(f"   [Req {req_index}] TTFT: {ttft:.2f}s")

    except httpx.ReadTimeout:
        logger.error(f"💀 [Req {req_index} | {req_id}] CLIENT TIMEOUT (Strict) > {TIMEOUT_THRESHOLD}s")
        stats["stuck"] += 1
        stats["stuck_details"].append(req_id)
        await get_system_snapshot(client)
        
    except Exception as e:
        logger.error(f"❌ [Req {req_index}] Exception: {e}")
        stats["error"] += 1

async def traffic_generator():
    logger.info(f"🚀 Starting Reliability Test (Duration: {TEST_DURATION}s, RPS: {RPS})")
    
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        
        start_test_time = time.time()
        req_counter = 0
        tasks = []
        
        while time.time() - start_test_time < TEST_DURATION:
            # 發送一批請求
            for _ in range(RPS):
                req_counter += 1
                tasks.append(asyncio.create_task(send_monitored_request(client, req_counter)))
            
            # 簡單的速率控制
            await asyncio.sleep(1.0)
            
        logger.info("⏳ Waiting for pending requests to finish...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
    logger.info("\n" + "="*30)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*30)
    logger.info(f"Total Sent:    {stats['sent']}")
    logger.info(f"Success:       {stats['success']}")
    logger.info(f"Errors:        {stats['error']}")
    logger.info(f"Stuck (> {TIMEOUT_THRESHOLD}s): {stats['stuck']}")
    if stats["stuck"] > 0:
        logger.error(f"Stuck IDs: {stats['stuck_details'][:10]} ...")
        logger.info("💡 Tip: Search these IDs in Control Node logs to see if they were dispatched.")

if __name__ == "__main__":
    try:
        asyncio.run(traffic_generator())
    except KeyboardInterrupt:
        print("\n🛑 Test Interrupted")
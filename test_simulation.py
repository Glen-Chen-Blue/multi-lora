import asyncio
import httpx
import pandas as pd
import json
import time
from datetime import datetime

# 匯入常數以確保區間對齊
from config import SP1_INTERVAL_SECONDS

TRACE_CSV = "./information/simulation_data.csv"
START_OFFSET = 86400 * 2
RUN_DURATION = 3600
TIMEOUT = 120

EFO_URL = "http://localhost:9900"

# ===== NEW =====
# SPEEDUP > 1.0 => replay faster (more req/sec).
# SPEEDUP = 2.0 means timeline is compressed by 2x.
SPEEDUP = 1.0

# 只有在這個列表中的 Cluster 請求兩會被發送，其餘會被忽略
TARGET_CLUSTERS = ["cluster_1", "cluster_2", "cluster_3"] 
# ==============

CLUSTER_PORT_MAP = {
    "cluster_1": 9000,
    "cluster_2": 9001,
    "cluster_3": 9002
}

GREEN = "\033[92m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; RED = "\033[91m"
RESET = "\033[0m"

stats = {"sent": 0, "finished": 0, "dropped": 0, "errors": 0}
ttft_records = []

# ====================
# Load Trace
# ====================
if SPEEDUP <= 0:
    raise ValueError("SPEEDUP must be > 0")

print("📥 Loading CSV trace...")
df = pd.read_csv(TRACE_CSV)

df["arrival_sec"] = df["arrive_timestamp"].astype(float)
df = df[df["arrival_sec"] >= START_OFFSET].copy()
df["arrival_sec"] -= START_OFFSET
df = df[df["arrival_sec"] <= RUN_DURATION]

# 過濾出目標 Cluster 的請求
df = df[df["cluster"].isin(TARGET_CLUSTERS)]

df = df.sort_values("arrival_sec").reset_index(drop=True)

TOTAL_REQUESTS = len(df)

print(f"⏱ Replay Duration={RUN_DURATION}s (speedup={SPEEDUP}x, effective duration={RUN_DURATION / SPEEDUP:.2f}s)")
print(f"🎯 Target Clusters: {TARGET_CLUSTERS}")
print(f"📦 Requests={TOTAL_REQUESTS}")

# ====================
async def simulate_trace_req(client, row, idx):
    cluster = row["cluster"]
    lora_id = int(row["lora_id"])
    port = CLUSTER_PORT_MAP[cluster]
    CONTROL_URL = f"http://localhost:{port}"
    adapter = f"LoRA_{lora_id}"

    payload = {
        "prompt": "test",
        "adapter_id": adapter,
        "max_new_tokens": 256
    }

    stats["sent"] += 1
    print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] #{idx}/{TOTAL_REQUESTS} SENDING -> {adapter}@{cluster}{RESET}")

    start = time.time()
    ttft = 0
    tokens = []
    is_dropped = False
    reason = ""

    try:
        r = await client.post(f"{CONTROL_URL}/send_request", json=payload, timeout=30)
        r.raise_for_status()
        rid = r.json()["request_id"]

        async with client.stream("GET", f"{CONTROL_URL}/stream/{rid}", timeout=TIMEOUT) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: [DONE]"):
                    break
                if line.startswith("data:"):
                    raw = line[5:].rstrip()

                    if raw.strip() in ["ok", "connected"]:
                        continue
                    try:
                        content = json.loads(raw)
                    except Exception:
                        content = raw

                    if isinstance(content, dict) and content.get("type") == "error":
                        is_dropped = True
                        reason = content.get("message", "Unknown")
                        break

                    if isinstance(content, str) and (
                        content.startswith("[ERROR]") or
                        "Processing aborted" in content
                    ):
                        is_dropped = True
                        reason = content
                        break

                    if ttft == 0:
                        ttft = time.time() - start
                    tokens.append(content)

        elapsed = time.time() - start

        if is_dropped:
            stats["dropped"] += 1
            print(f"{RED}[{datetime.now().strftime('%H:%M:%S')}] #{idx} DROPPED <- {adapter} (Reason:{reason}){RESET}")
        else:
            stats["finished"] += 1
            final_ttft = ttft if ttft > 0 else elapsed
            ttft_records.append(final_ttft)
            print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] #{idx} DONE <- {adapter} (Time:{elapsed:.2f}s TTFT:{final_ttft:.2f}s Tokens:{len(tokens)}){RESET}")

    except Exception as e:
        stats["errors"] += 1
        print(f"{RED}[ERROR] #{idx} Failed:{repr(e)}{RESET}")

# ====================
async def main():
    limits = httpx.Limits(max_connections=300, max_keepalive_connections=300)

    async with httpx.AsyncClient(limits=limits) as client:
        print(f"{YELLOW}=== Trace Replay Pressure Simulator ==={RESET}")

        # 1. 初始觸發 SP1 (第 0 區間)
        print(f"\n{YELLOW}🚀 Triggering initial SP1 /time_edge (Step 0)...{RESET}")
        try:
            resp = await client.post(f"{EFO_URL}/time_edge", timeout=600.0)
            print(f"{GREEN}✅ Initial SP1 complete: {resp.json()}{RESET}\n")
        except Exception as e:
            print(f"{RED}❌ Initial SP1 failed: {e}{RESET}")
            return

        start_time = time.time()
        tasks = []
        current_interval = 0

        for i, row in df.iterrows():
            arrival_sec = float(row["arrival_sec"])
            req_interval = int(arrival_sec // SP1_INTERVAL_SECONDS)

            # 2. 跨越時間區間，暫停並觸發新的 SP1
            if req_interval > current_interval:
                print(f"\n{YELLOW}⏳ Reached interval {req_interval} (Arrival Sec: {arrival_sec:.1f}). Triggering /time_edge...{RESET}")
                edge_start_time = time.time()
                try:
                    resp = await client.post(f"{EFO_URL}/time_edge", timeout=600.0)
                    print(f"{GREEN}✅ SP1 Interval {req_interval} complete: {resp.json()}{RESET}\n")
                except Exception as e:
                    print(f"{RED}❌ SP1 Interval {req_interval} failed: {e}{RESET}\n")
                
                # 時鐘補償：把等待排空的時間加回 start_time，避免後續 Request 認為遲到而瞬間暴衝
                edge_duration = time.time() - edge_start_time
                start_time += edge_duration
                current_interval = req_interval

            scheduled_offset = arrival_sec / SPEEDUP
            send_time = start_time + scheduled_offset

            delay = send_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

            task = asyncio.create_task(simulate_trace_req(client, row, i + 1))
            tasks.append(task)

        print(f"\n{YELLOW}=== Waiting for remaining tasks ==={RESET}")
        await asyncio.gather(*tasks)

    print(f"\n=== Summary: Sent {stats['sent']} / Fin {stats['finished']} / Drop {stats['dropped']} / Err {stats['errors']} ===")

    if ttft_records:
        avg = sum(ttft_records) / len(ttft_records)
        p95 = sorted(ttft_records)[int(len(ttft_records) * 0.95)]
        print(f"{CYAN}Average TTFT:{avg:.4f}s{RESET}")
        print(f"{CYAN}P95 TTFT:{p95:.4f}s{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
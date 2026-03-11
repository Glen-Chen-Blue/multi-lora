import asyncio
import httpx
import pandas as pd
import json
import time
import os
from datetime import datetime

# 匯入常數以確保區間對齊
from config import SP1_INTERVAL_SECONDS, START_OFFSET

TRACE_CSV = "./information/simulation_data.csv"
RUN_DURATION = SP1_INTERVAL_SECONDS * 8
TIMEOUT = 120

# ====================
# ✅ Read from env (set by your .sh)
# ====================
EFO_URL = os.getenv("EFO_URL", "http://localhost:9900")

def _load_json_env(name: str, default_json: str):
    raw = os.getenv(name, "").strip()
    if not raw:
        return json.loads(default_json)
    try:
        return json.loads(raw)
    except Exception as e:
        raise ValueError(f"Env var {name} is not valid JSON: {raw!r}, error={e}")

# 只有在這個列表中的 Cluster 請求才會被發送，其餘會被忽略
TARGET_CLUSTERS = _load_json_env("TARGET_CLUSTERS", '["cluster_1"]')

# cluster -> control node port
CLUSTER_PORT_MAP = _load_json_env(
    "CLUSTER_PORT_MAP",
    '{"cluster_1": 9000, "cluster_2": 9001, "cluster_3": 9002}'
)

# normalize port type
CLUSTER_PORT_MAP = {k: int(v) for k, v in CLUSTER_PORT_MAP.items()}
# ====================

# ===== NEW =====
# SPEEDUP > 1.0 => replay faster (more req/sec).
# SPEEDUP = 2.0 means timeline is compressed by 2x.
SPEEDUP = 1.0
# ==============

# 統計資料
stats = {"sent": 0, "finished": 0, "dropped": 0, "errors": 0}
ttft_records = []

# 取得當前時間字串的輔助函數
def get_ts():
    return datetime.now().strftime('%H:%M:%S')

# ====================
# Load Trace
# ====================
if SPEEDUP <= 0:
    raise ValueError("SPEEDUP must be > 0")

print("[INFO] 📥 Loading CSV trace...")
df = pd.read_csv(TRACE_CSV)

df["arrival_sec"] = df["arrive_timestamp"].astype(float)
df = df[df["arrival_sec"] >= START_OFFSET].copy()
df["arrival_sec"] -= START_OFFSET
df = df[df["arrival_sec"] <= RUN_DURATION]

# 過濾出目標 Cluster 的請求
df = df[df["cluster"].isin(TARGET_CLUSTERS)]
df = df.sort_values("arrival_sec").reset_index(drop=True)

TOTAL_REQUESTS = len(df)
PAD_LEN = len(str(TOTAL_REQUESTS)) # 動態計算請求編號對齊寬度

print("-" * 65)
print(f"[INFO] ✅ Using EFO_URL      : {EFO_URL}")
print(f"[INFO] 🎯 Target Clusters    : {TARGET_CLUSTERS}")
print(f"[INFO] 🧭 Cluster Port Map   : {CLUSTER_PORT_MAP}")
print(f"[INFO] ⏱  Replay Duration    : {RUN_DURATION}s (speedup={SPEEDUP}x, effective={RUN_DURATION / SPEEDUP:.2f}s)")
print(f"[INFO] 📦 Requests Count     : {TOTAL_REQUESTS}")
print("-" * 65)

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
    
    # 格式化輸出字串
    req_str = f"{idx:>{PAD_LEN}}/{TOTAL_REQUESTS}"
    adapter_str = f"{adapter:^8}"
    
    print(f"[{get_ts()}] [SEND] Req:{req_str} | Target:{adapter_str} @ {cluster}")

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
        ts = get_ts()

        if is_dropped:
            stats["dropped"] += 1
            print(f"[{ts}] [DROP] Req:{req_str} | Target:{adapter_str} | Reason: {reason}")
        else:
            stats["finished"] += 1
            final_ttft = ttft if ttft > 0 else elapsed
            ttft_records.append(final_ttft)
            print(f"[{ts}] [DONE] Req:{req_str} | Target:{adapter_str} | Time: {elapsed:>6.2f}s | TTFT: {final_ttft:>5.2f}s | Tokens: {len(tokens)}")

    except Exception as e:
        stats["errors"] += 1
        print(f"[{get_ts()}] [FAIL] Req:{req_str} | Target:{adapter_str} | Exception: {repr(e)}")

# ====================
async def main():
    limits = httpx.Limits(max_connections=300, max_keepalive_connections=300)

    async with httpx.AsyncClient(limits=limits) as client:
        print("=" * 65)
        print("=== Trace Replay Pressure Simulator ===")
        print("=" * 65)

        # 1. 初始觸發 SP1 (第 0 區間)
        print(f"\n[{get_ts()}] [SYS ] 🚀 Triggering initial SP1 /time_edge (Step 0)...")
        try:
            resp = await client.post(f"{EFO_URL}/time_edge", timeout=600.0)
            await asyncio.sleep(3.0)
            print(f"[{get_ts()}] [SYS ] ✅ Initial SP1 complete: {resp.json()}\n")
        except Exception as e:
            print(f"[{get_ts()}] [SYS ] ❌ Initial SP1 failed: {e}")
            return

        start_time = time.time()
        tasks = []
        current_interval = 0

        for i, row in df.iterrows():
            arrival_sec = float(row["arrival_sec"])
            req_interval = int(arrival_sec // SP1_INTERVAL_SECONDS)

            # 2. 跨越時間區間，暫停並觸發新的 SP1
            if req_interval > current_interval:
                print(f"[{get_ts()}] [SYS ] ⏳ Reached Interval {req_interval}. Pausing to trigger /time_edge...")
                sleep_start = time.time()
                await asyncio.sleep(10.0)

                await client.post(f"{EFO_URL}/time_edge", timeout=600.0)

                total_pause = time.time() - sleep_start
                start_time += total_pause
                current_interval = req_interval
                print(f"[{get_ts()}] [SYS ] ▶️ Resuming simulation for Interval {req_interval}.")

            scheduled_offset = arrival_sec / SPEEDUP
            send_time = start_time + scheduled_offset

            delay = send_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

            task = asyncio.create_task(simulate_trace_req(client, row, i + 1))
            tasks.append(task)

        print(f"\n[{get_ts()}] [SYS ] === Waiting for remaining tasks to complete ===")
        await asyncio.gather(*tasks)

    print("\n" + "=" * 65)
    print(f"=== SUMMARY: Sent: {stats['sent']} | Finished: {stats['finished']} | Dropped: {stats['dropped']} | Errors: {stats['errors']} ===")
    print("=" * 65)

    if ttft_records:
        avg = sum(ttft_records) / len(ttft_records)
        p95 = sorted(ttft_records)[int(len(ttft_records) * 0.95)]
        print(f"[STAT] Average TTFT : {avg:.4f} s")
        print(f"[STAT] P95 TTFT     : {p95:.4f} s")
    print("-" * 65 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
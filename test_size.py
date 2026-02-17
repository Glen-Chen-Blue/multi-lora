import time
import httpx
import os

# 設定目標 Control Node (預設為 single_area.sh 的配置)
CONTROL_NODE_URL = os.environ.get("CONTROL_NODE_URL", "http://localhost:9000")
# 測試用的 Adapter ID (對應您之前測試用的 1~100)
ADAPTER_ID = "1" 

def test_fetch():
    url = f"{CONTROL_NODE_URL}/fetch_adapter/{ADAPTER_ID}"
    print(f"🚀 Testing fetch latency from: {url}")
    print(f"   Target Adapter ID: {ADAPTER_ID}")
    print("-" * 60)

    # --- 第一次嘗試 (模擬 Cold Start) ---
    # 如果 Control Node 本地沒有，它會去 EFO 下載，這會包含 (EFO->Control) + (Control->Client) 的時間
    start_time = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            print("⏳ Attempt 1 (Requesting)...")
            resp = client.get(url)
            
            if resp.status_code != 200:
                print(f"❌ Attempt 1 Failed: Status {resp.status_code} - {resp.text}")
                return
                
            content = resp.content
            
        duration = time.time() - start_time
        size_bytes = len(content)
        size_mb = size_bytes / (1024 * 1024)
        speed = size_mb / duration if duration > 0 else 0
        
        print(f"✅ Attempt 1 Success!")
        print(f"   Time Taken : {duration:.4f} s")
        print(f"   Size       : {size_mb:.2f} MB")
        print(f"   Throughput : {speed:.2f} MB/s")
        if duration > 1.0:
            print("   (ℹ️  Likely fetched from EFO or disk I/O latency)")
            
    except Exception as e:
        print(f"❌ Attempt 1 Error: {e}")
        return

    print("-" * 60)

    # --- 第二次嘗試 (模擬 Cached / Hot Start) ---
    # 此時 Control Node 硬碟已經有檔案，理論上只剩 HTTP 傳輸時間
    start_time = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            print("⏳ Attempt 2 (Requesting again - assume cached)...")
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            
        duration = time.time() - start_time
        size_mb = len(content) / (1024 * 1024)
        speed = size_mb / duration if duration > 0 else 0
        
        print(f"✅ Attempt 2 Success!")
        print(f"   Time Taken : {duration:.4f} s")
        print(f"   Size       : {size_mb:.2f} MB")
        print(f"   Throughput : {speed:.2f} MB/s")
        
    except Exception as e:
        print(f"❌ Attempt 2 Error: {e}")

if __name__ == "__main__":
    test_fetch()
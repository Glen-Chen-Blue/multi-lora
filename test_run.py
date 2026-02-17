import requests
import json
import sseclient # pip install sseclient-py
import threading

CONTROL_URL = "http://localhost:9000"

def send_and_listen(adapter_id, prompt):
    print(f"Sending request for LoRA {adapter_id}...")
    
    # 1. 發送請求
    resp = requests.post(f"{CONTROL_URL}/send_request", json={
        "prompt": prompt,
        "adapter_id": adapter_id,
        "max_new_tokens": 50
    })
    
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return

    req_id = resp.json()["request_id"]
    print(f"Request ID: {req_id}. Listening for stream...")

    # 2. 監聽 SSE 串流
    url = f"{CONTROL_URL}/stream/{req_id}"
    headers = {'Accept': 'text/event-stream'}
    response = requests.get(url, stream=True, headers=headers)
    client = sseclient.SSEClient(response)

    full_text = ""
    for event in client.events():
        if event.data == "[DONE]":
            print(f"\n[DONE] LoRA {adapter_id} finished.")
            break
        
        # 嘗試解析 JSON (Compute Node 回傳的是 JSON string)
        try:
            # 這裡要注意：control_node 直接把 compute_node 的 "data: ..." 內容轉發
            # compute_node 回傳的 data 可能是一個 json string，例如 "{\"token\": \"Hello\"}"
            # 或者如果是錯誤訊息 "{\"type\": \"error\"...}"
            # 我們的 control node 簡單轉發 string
            data_str = event.data
            # 去除可能的前後引號 (如果是 raw json dump)
            if data_str.startswith('"') and data_str.endswith('"'):
                data_str = json.loads(data_str) # 解一次 json string -> string/obj
            
            print(data_str, end="", flush=True)
            full_text += str(data_str)
        except:
            print(event.data, end="", flush=True)

    print(f"\nFinal Output Length: {len(full_text)}")

if __name__ == "__main__":
    # 測試情境：同時發送兩個請求給不同的 LoRA
    # 預期：
    # 1. Control Node 收到請求
    # 2. Compute Node 發現本地沒有 LoRA (lora_repo/cn_x 空的)
    # 3. Compute Node 向 Control Node 下載 LoRA (Lazy Load)
    # 4. 開始推論並回傳
    
    t1 = threading.Thread(target=send_and_listen, args=("LoRA_1", "Test Prompt for LoRA 1"))
    t2 = threading.Thread(target=send_and_listen, args=("LoRA_2", "Test Prompt for LoRA 2"))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
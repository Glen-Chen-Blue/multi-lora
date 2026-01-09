import asyncio
import httpx
import sys
import json # [Added] Import json

# ==========================================
# 設定
# ==========================================
CONTROL_URL = "http://localhost:9000"
ADAPTER_ID = "chat"  # 請確保這個資料夾存在於 compute node 的 ./testLoRA 中
PROMPT_TEXT = "List 5 benefits of regular exercise."
MAX_NEW_TOKENS = 128
# ==========================================
# Alpaca 格式化函式 (對應你的 Fine-tuning)
# ==========================================
def format_alpaca_prompt(user_prompt):
    return (
        f"### Instruction:\n{user_prompt}\n\n"
        f"### Response:\n"
    )

async def main():
    # 1. 準備 Payload
    formatted_prompt = format_alpaca_prompt(PROMPT_TEXT)
    payload = {
        "prompt": formatted_prompt,
        "adapter_id": ADAPTER_ID,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    print(f"🚀 Sending request to {CONTROL_URL}...")
    print(f"📝 Adapter: {ADAPTER_ID}")
    print(f"❓ Prompt: {PROMPT_TEXT}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # 2. 發送請求取得 Request ID
            resp = await client.post(f"{CONTROL_URL}/send_request", json=payload)
            resp.raise_for_status()
            
            data = resp.json()
            request_id = data["request_id"]
            print(f"✅ Request ID: {request_id}")
            print(f"waiting for stream...\n")
            print(">> ", end="", flush=True)

            # 3. 訂閱串流 (Streaming)
            async with client.stream("GET", f"{CONTROL_URL}/stream/{request_id}", timeout=120.0) as response:
                async for line in response.aiter_lines():
                    if not line: continue

                    # 處理結束訊號 (通常由 Control Node 發送)
                    if line.startswith("data: [DONE]"):
                        print("\n\n[DONE] Stream finished.")
                        break
                    
                    # 處理資料
                    if line.startswith("data:"):
                        # [Modified] 移除 strip(), 改用 json 解析
                        content = line[len("data:"):].rstrip("\n")
                        
                        if content == "ok": continue # Handshake
                        
                        try:
                            # 嘗試解析 JSON (因為 Compute Node 現在送 JSON)
                            text_token = json.loads(content)
                        except json.JSONDecodeError:
                            # Fallback: 如果不是 JSON (例如 legacy error)，直接用 raw string
                            text_token = content
                        
                        # 處理錯誤訊息
                        if text_token.startswith("[ERROR]"):
                            print(f"\n❌ Server Error: {text_token}")
                            break

                        # 即時印出 Token (不換行)
                        print(text_token, end="", flush=True)

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"\n❌ Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
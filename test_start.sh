#!/usr/bin/env bash

PIDS=()

stop() {
  echo ""
  echo "🛑 Stopping services..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
  
  # 強制殺死殘留 (Double Tap)
  sleep 1
  pkill -f "uvicorn control_node_server"
  pkill -f "uvicorn compute_node_server"
  
  echo "✅ Stopped."
}

trap stop INT TERM

start() {
  echo "=== Phase 1: Infrastructure Test (Dynamic Registration Mode) ==="
  echo "📂 Using existing LoRA files from ./testLoRA"

  echo "🚀 Starting Control Node (Port 9000)..."
  # [修改] 移除了 COMPUTE_NODES 環境變數，讓 Compute Node 自己來報到
  LORA_PATH="./testLoRA" \
  uvicorn control_node_server:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)
  
  # 等待 Control Node 啟動
  sleep 2

  echo "🚀 Starting Compute Nodes..."
  
  # [修改] 明確加上 PORT=800x，讓 Python 內部能拿到正確的 URL 去註冊
  NODE_ID=cn-1 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8001 \
  uvicorn compute_node_server:app --port 8001 &
  PIDS+=($!)

  NODE_ID=cn-2 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8002 \
  uvicorn compute_node_server:app --port 8002 &
  PIDS+=($!)

  NODE_ID=cn-3 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8003 \
  uvicorn compute_node_server:app --port 8003 &
  PIDS+=($!)

  echo "✅ All services started. Press Ctrl+C to stop."
  wait
}

start
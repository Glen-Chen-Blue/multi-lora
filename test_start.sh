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
  echo "=== Phase 1: Infrastructure Test (Real LoRA Mode) ==="
  echo "📂 Using existing LoRA files from ./testLoRA"

  echo "🚀 Starting Control Node (Port 9000)..."
  # 注意：LORA_PATH 指向存放真實 LoRA 的目錄
  LORA_PATH="./testLoRA" \
  COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003" \
  uvicorn control_node_server:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)
  
  # 等待 Control Node 啟動
  sleep 2

  # 3. 啟動 Compute Nodes
  echo "🚀 Starting Compute Nodes..."
  
  # CUDA_VISIBLE_DEVICES=0 \
  NODE_ID=cn-1 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  uvicorn compute_node_server:app --port 8001 &
  PIDS+=($!)

  # CUDA_VISIBLE_DEVICES=1 \
  NODE_ID=cn-2 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  uvicorn compute_node_server:app --port 8002 &
  PIDS+=($!)

  NODE_ID=cn-3 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  uvicorn compute_node_server:app --port 8003 &
  PIDS+=($!)

  echo "✅ All services started. Press Ctrl+C to stop."
  wait
}

start
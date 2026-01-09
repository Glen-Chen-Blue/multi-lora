#!/usr/bin/env bash
trap stop INT TERM
set -e

PIDS=()

start() {
  echo "=== Mode 1: 1 EFO -> 1 Control -> 2 Compute ==="

  # 1. 啟動 EFO Server (Port 9090)
  echo "Starting EFO Server..."
  uvicorn EFO_server:app --host 0.0.0.0 --port 9080 &
  PIDS+=($!)
  sleep 2

  # 2. 啟動 Compute Node 1 (Port 8001)
  echo "Starting Compute Node 1..."
  CUDA_VISIBLE_DEVICES=0 NODE_ID=cn-1 uvicorn compute_node_server:app --port 8001 &
  PIDS+=($!)

  # 3. 啟動 Compute Node 2 (Port 8002)
  echo "Starting Compute Node 2..."
  CUDA_VISIBLE_DEVICES=1 NODE_ID=cn-2 uvicorn compute_node_server:app --port 8002 &
  PIDS+=($!)

  echo "Waiting 5 seconds for compute nodes to warm up..."
  sleep 5

  # 4. 啟動 Control Node (Port 9000)
  # 它負責管理 cn-1 和 cn-2，並指向 EFO
  echo "Starting Control Node..."
  EFO_URL="http://127.0.0.1:9080" \
  MY_NODE_URL="http://127.0.0.1:9000" \
  COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002" \
  uvicorn control_node_server:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)

  echo "✅ All services started. Press Ctrl+C to stop."
  wait
}

stop() {
  echo "Stopping all services..."

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing PID $pid"
      kill "$pid"
    fi
  done

  # 強制殺死殘留進程
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing PID $pid"
      kill -9 "$pid"
    fi
  done

  echo "All services stopped."
}

# 執行啟動
start
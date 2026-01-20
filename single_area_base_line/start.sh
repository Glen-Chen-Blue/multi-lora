#!/usr/bin/env bash
trap stop INT TERM
set -e

PIDS=()

start() {
  echo "=== Baseline Mode: 2 Compute (Fixed Batch) + 1 Random Control ==="

  # 1. 啟動 Compute Node 1 (Port 8001)
  echo "Starting Compute Node 1..."
  # [Fix] 加入 PORT=8001
  CUDA_VISIBLE_DEVICES=0 NODE_ID=cn-1 MAX_BATCH_SIZE=32 PORT=8001 \
  python test_compute_node.py &
  PIDS+=($!)

  # 2. 啟動 Compute Node 2 (Port 8002)
  echo "Starting Compute Node 2..."
  # [Fix] 加入 PORT=8002
  CUDA_VISIBLE_DEVICES=1 NODE_ID=cn-2 MAX_BATCH_SIZE=32 PORT=8002 \
  python test_compute_node.py &
  PIDS+=($!)

  echo "Waiting 8 seconds for compute nodes to load all LoRAs..."
  sleep 8

  # 3. 啟動 Control Node (Port 9000)
  echo "Starting Control Node (Random Dispatch)..."
  COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002" \
  python test_control_node.py &
  PIDS+=($!)
#   echo "Starting Control Node (Random Dispatch)..."
#   COMPUTE_NODES="http://127.0.0.1:8001" \
#   python test_control_node.py &
#   PIDS+=($!)

  echo "✅ All baseline services started. Press Ctrl+C to stop."
  wait
}

stop() {
  echo "Stopping all services..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
  echo "All services stopped."
}

start
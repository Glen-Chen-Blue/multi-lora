#!/usr/bin/env bash
trap stop INT TERM
set -e

PIDS=()

start() {
  echo "Starting compute node 1..."
  CUDA_VISIBLE_DEVICES=0 NODE_ID=cn-1 uvicorn compute_node_server:app --port 8001 &
  PIDS+=($!)

  echo "Starting compute node 2..."
  CUDA_VISIBLE_DEVICES=1 NODE_ID=cn-2 uvicorn compute_node_server:app --port 8002 &
  PIDS+=($!)

  echo "Waiting 10 seconds before starting control node..."
  sleep 10

  echo "Starting control node..."
  COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002" QMIN_MULT=2 \
    uvicorn control_node:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)

  echo "All services started."
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

  # 再保險：如果還活著，強殺
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing PID $pid"
      kill -9 "$pid"
    fi
  done

  echo "All services stopped."
}

trap stop INT TERM

case "$1" in
  start)
    start
    ;;
  *)
    echo "Usage: $0 start"
    ;;
esac

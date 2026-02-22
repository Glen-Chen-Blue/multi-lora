#!/usr/bin/env bash

PIDS=()

# ===============================
# 🧨 Kill port helper function
# ===============================
kill_port() {
  PORT=$1
  PID=$(lsof -ti tcp:$PORT)

  if [ ! -z "$PID" ]; then
    echo "⚠️  Port $PORT in use → Killing PID $PID"
    kill -9 $PID
    sleep 0.3
  fi
}

# ===============================
# 🛑 Stop handler
# ===============================
stop() {
  echo ""
  echo "🛑 Stopping services..."

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
  
  sleep 1

  pkill -f "uvicorn EFO_server"
  pkill -f "uvicorn control_node_server"
  pkill -f "uvicorn compute_node_server"
  
  echo "✅ Stopped."
}

trap stop INT TERM

# ===============================
# 🚀 Start
# ===============================
start() {

  echo "🧹 Cleaning target ports..."

  kill_port 9100
  kill_port 9000
  kill_port 8001
  kill_port 8002
  kill_port 8003

  echo "=== Phase 1: Infrastructure Test (Dynamic Registration Mode) ==="
  echo "📂 Using existing LoRA files from ./testLoRA"

  echo "🚀 Starting EFO (Port 9100)..."
  PORT=9100 \
  LORA_PATH="./testLoRA" \
  LORA_METADATA="./lora_metadata.json" \
  CLUSTERS='{"cluster_1":"http://127.0.0.1:9000","cluster_2":"http://127.0.0.1:9001","cluster_3":"http://127.0.0.1:9002"}' \
  uvicorn EFO_server:app --host 0.0.0.0 --port 9100 &
  PIDS+=($!)

  sleep 1

  echo "🚀 Starting Control Node (Port 9000)..."
  CLUSTER_NAME="cluster_1" \
  EFO_URL="http://127.0.0.1:9100" \
  LORA_PATH="./testLoRA" \
  uvicorn control_node_server:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)
  
  sleep 2

  echo "🚀 Starting Compute Nodes..."

  NODE_ID=cn-1 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8001 \
  uvicorn compute_node_server:app --host 0.0.0.0 --port 8001 &
  PIDS+=($!)

  NODE_ID=cn-2 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8002 \
  uvicorn compute_node_server:app --host 0.0.0.0 --port 8002 &
  PIDS+=($!)

  NODE_ID=cn-3 \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  PORT=8003 \
  uvicorn compute_node_server:app --host 0.0.0.0 --port 8003 &
  PIDS+=($!)

  echo "✅ All services started. Press Ctrl+C to stop."
  wait
}

start
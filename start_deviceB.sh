#!/usr/bin/env bash
set -euo pipefail

PIDS=()

# ==================================================
# 📦 Experiment Configuration (補齊與 A 裝置相同的變數)
# ==================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_METADATA_PATH="./information/lora_metadata.json"
export SIMULATION="0"
export LOG_PATH="./experiment_deviceB_logs"
mkdir -p "$LOG_PATH"

# ==================================================
# 🌐 Port Configuration
# ==================================================
BASE_PORT=8000
CTRL2_PORT=$((BASE_PORT + 102))  # 8102
COMP_PORT=$((BASE_PORT + 203))   # 8203 
export EFO_URL="http://127.0.0.1:$((BASE_PORT + 900))"

stop() {
  echo ""
  echo "🛑 Stopping Device B services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done
  echo "✅ Stopped."
  exit 0
}
trap stop EXIT INT TERM

echo "🧹 Clearing old logs..."
rm -f "$LOG_PATH"/*.log || true

# --- 1. 啟動 Control Node 2 ---
echo "🚀 Starting Control Node for cluster_2..."
CLUSTER_NAME="cluster_2" \
EFO_URL="$EFO_URL" \
PORT="$CTRL2_PORT" \
CONTROL_NODE_URL="http://127.0.0.1:$CTRL2_PORT" \
uvicorn control_node_server:app --host 0.0.0.0 --port "$CTRL2_PORT" \
>> "$LOG_PATH/control_2.log" 2>&1 &
PIDS+=($!)
sleep 2

# --- 2. 啟動 Compute Node (Cluster 2) ---
echo "🚀 Starting 1 Compute Node for cluster_2..."
CUDA_VISIBLE_DEVICES="0" \
NODE_ID="c2-n1" \
CONTROL_NODE_URL="http://127.0.0.1:$CTRL2_PORT" \
PORT="$COMP_PORT" \
uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" \
>> "$LOG_PATH/compute_c2-n1.log" 2>&1 &
PIDS+=($!)

echo "✅ Device B services started. Connected to EFO at $EFO_URL."
echo "⏳ Waiting for requests... (Press Ctrl+C to stop)"
wait
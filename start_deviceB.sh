#!/usr/bin/env bash
set -euo pipefail

# === 載入 conda 環境 ===
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "❌ 找不到 conda.sh，請確認 conda 安裝路徑" >&2
  exit 1
fi

conda activate myenv
echo "✅ Using python: $(which python)"
echo "✅ Using uvicorn: $(which uvicorn)"
python --version

# === 接收傳入的參數 ===
CTRL_APP=${1:-"control_node_server:app"}
export LORA_METADATA_PATH=${2:-"./information/lora_metadata.json"}
DISK_CAPACITY_GB=${3:-""}
DISPATCH_STRATEGY=${4:-""}

# 動態配置環境變數
if [ -n "$DISK_CAPACITY_GB" ]; then export DISK_CAPACITY_GB="$DISK_CAPACITY_GB"; fi
if [ -n "$DISPATCH_STRATEGY" ]; then export DISPATCH_STRATEGY="$DISPATCH_STRATEGY"; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SIMULATION="0"
export LOG_PATH="./experiment_deviceB_logs"
mkdir -p "$LOG_PATH"

BASE_PORT=8000
CTRL2_PORT=$((BASE_PORT + 102))  # 8102
COMP_PORT=$((BASE_PORT + 203))   # 8203 
export EFO_URL="http://127.0.0.1:$((BASE_PORT + 900))"

PIDS=()
stop() {
  echo "🛑 Stopping Device B services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done
  exit 0
}
trap stop EXIT INT TERM

echo "🧹 Clearing old logs and ports on Device B..."
pkill -f "uvicorn.*--port $CTRL2_PORT" || true
pkill -f "uvicorn.*--port $COMP_PORT" || true
rm -f "$LOG_PATH"/*.log || true

# --- 1. 啟動 Control Node 2 ---
echo "🚀 Starting Control Node 2 ($CTRL_APP)..."
CLUSTER_NAME="cluster_2" \
EFO_URL="$EFO_URL" \
PORT="$CTRL2_PORT" \
CONTROL_NODE_URL="http://127.0.0.1:$CTRL2_PORT" \
uvicorn "$CTRL_APP" --host 0.0.0.0 --port "$CTRL2_PORT" \
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

echo "✅ Device B started. Waiting for requests..."
wait
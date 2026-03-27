#!/usr/bin/env bash
set -euo pipefail

# ================= 實驗 1_1 變數配置 =================
EFO_APP="EFO_server:app"
CTRL_APP="control_node_server:app"
export LORA_METADATA_PATH="./information/lora_metadata.json"
export DISK_CAPACITY_GB=""
export DISPATCH_STRATEGY=""
# =====================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOG_PATH="./experiment_deviceA_logs_1"
export SIMULATION="0"
mkdir -p "$LOG_PATH"

BASE_PORT=8000
EFO_PORT=$((BASE_PORT + 900))
CTRL1_PORT=$((BASE_PORT + 100))
CTRL2_PORT=$((BASE_PORT + 102))
export EFO_URL="http://127.0.0.1:${EFO_PORT}"

SSH_SOCKET="/tmp/multi_lora_tunnel.sock"
REMOTE_USER="glenchen"
REMOTE_HOST="140.112.20.183"

PIDS=()

stop() {
  echo "🛑 Stopping Device A local services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done
  
  echo "🛑 Stopping Device B remote services via SSH..."
  # 透過 SSH 呼叫 B 機器把剛開的服務清掉
  ssh -S "$SSH_SOCKET" "$REMOTE_USER@$REMOTE_HOST" "pkill -f start_deviceB.sh || true; pkill -f \"uvicorn.*--port $CTRL2_PORT\" || true; pkill -f \"uvicorn.*--port $((BASE_PORT + 203))\" || true" 2>/dev/null || true
  sleep 2
}
trap stop EXIT INT TERM

echo "🧹 Clearing old processes..."
pkill -f "uvicorn.*--port $EFO_PORT" || true
pkill -f "uvicorn.*--port $CTRL1_PORT" || true

echo "🚀 Starting EFO (Port $EFO_PORT)..."
PORT="$EFO_PORT" \
CLUSTERS="{\"cluster_1\":\"http://127.0.0.1:${CTRL1_PORT}\", \"cluster_2\":\"http://127.0.0.1:${CTRL2_PORT}\"}" \
uvicorn "$EFO_APP" --host 0.0.0.0 --port "$EFO_PORT" > "$LOG_PATH/efo.log" 2>&1 &
PIDS+=($!)
sleep 5

echo "📡 透過 Tunnel 啟動 Device B 服務..."
ssh -f -n -S "$SSH_SOCKET" -p 6666 "$REMOTE_USER@$REMOTE_HOST" \
  "cd \"$PWD\" && nohup ./start_deviceB.sh \"$CTRL_APP\" \"$LORA_METADATA_PATH\" \"$DISK_CAPACITY_GB\" \"$DISPATCH_STRATEGY\" > deviceB_nohup.log 2>&1 < /dev/null &"
sleep 30

echo "🚀 Starting Control Node cluster_1..."
CLUSTER_NAME="cluster_1" \
EFO_URL="$EFO_URL" \
PORT="$CTRL1_PORT" \
CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
uvicorn "$CTRL_APP" --host 0.0.0.0 --port "$CTRL1_PORT" > "$LOG_PATH/control_1.log" 2>&1 &
PIDS+=($!)
sleep 2

echo "🚀 Starting 2 Compute Nodes for cluster_1..."
for n in 1 2; do
  COMP_PORT=$((BASE_PORT + 200 + n))
  CUDA_VISIBLE_DEVICES="$((n-1))" \
  NODE_ID="c1-n${n}" \
  CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
  PORT="$COMP_PORT" \
  uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" > "$LOG_PATH/compute_c1-n${n}.log" 2>&1 &
  PIDS+=($!)
done

# --- 3. 自動偵測 Device B 是否已連線 ---
echo "⏳ 等待 Device B ($CTRL2_PORT) 啟動完成..."
while ! (echo > /dev/tcp/127.0.0.1/$CTRL2_PORT) >/dev/null 2>&1; do
  sleep 1
done
echo "✅ Device B 啟動完成，開始發送 Request！"
sleep 5 # 多等5秒確保 EFO 註冊完畢

# --- 4. 執行實驗 ---
export TARGET_CLUSTERS='["cluster_1", "cluster_2"]'
export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT}, \"cluster_2\":${CTRL2_PORT}}"

set +e
python test_simulation.py
SIM_EXIT_CODE=$?
set -e

exit "$SIM_EXIT_CODE"
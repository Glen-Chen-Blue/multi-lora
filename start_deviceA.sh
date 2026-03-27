#!/usr/bin/env bash
set -euo pipefail

PIDS=()
# 定義一個專門用來控制 SSH Tunnel 的 Socket 檔案
SSH_SOCKET="/tmp/multi_lora_tunnel.sock"

# === 實驗與路徑設定 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOG_PATH="./experiment_deviceA_logs"
export LORA_METADATA_PATH="./information/lora_metadata.json"
export SIMULATION="0"
mkdir -p "$LOG_PATH"

# === Port 設定 ===
BASE_PORT=8000
EFO_PORT=$((BASE_PORT + 900))        # 8900
CTRL1_PORT=$((BASE_PORT + 100))      # 8100
CTRL2_PORT=$((BASE_PORT + 102))      # 8102
export EFO_URL="http://127.0.0.1:${EFO_PORT}"

# === 清理與停止函式 ===
stop() {
  echo ""
  echo "🛑 Stopping Device A services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done

  # 透過 Control Socket 優雅地關閉 SSH Tunnel
  if [[ -S "$SSH_SOCKET" ]]; then
    echo "🔌 Closing SSH Tunnel..."
    ssh -S "$SSH_SOCKET" -O exit glenchen@140.112.20.183 2>/dev/null || true
  fi
  
  echo "✅ Stopped."
  exit 0  # <--- 重要：強制結束腳本，解決 Ctrl+C 關不掉的問題
}
# 攔截結束訊號
trap stop EXIT INT TERM

echo "🧹 Clearing old logs..."
rm -f "$LOG_PATH"/*.log || true

# ==================================================
# 1. 建立 SSH 雙向 Tunnel
# ==================================================
echo "🔗 Establishing SSH Tunnel to Device B (140.112.20.183)..."
echo "🔑 請在下方輸入密碼 (若有詢問憑證請輸入 yes)..."

# 改用 -f (等密碼輸入完才進背景)，-M 建立 Socket，並加上 accept-new 自動接受未知指紋
ssh -M -S "$SSH_SOCKET" -f -N -p 6666 \
  -o StrictHostKeyChecking=accept-new \
  -L ${CTRL2_PORT}:127.0.0.1:${CTRL2_PORT} \
  -R ${EFO_PORT}:127.0.0.1:${EFO_PORT} \
  glenchen@140.112.20.183

echo "✅ Tunnel established!"
sleep 2

# ==================================================
# 2. 啟動 EFO Server
# ==================================================
echo "🚀 Starting EFO (Port $EFO_PORT)..."
PORT="$EFO_PORT" \
CLUSTERS="{\"cluster_1\":\"http://127.0.0.1:${CTRL1_PORT}\", \"cluster_2\":\"http://127.0.0.1:${CTRL2_PORT}\"}" \
uvicorn EFO_server:app --host 0.0.0.0 --port "$EFO_PORT" \
>> "$LOG_PATH/efo.log" 2>&1 &
PIDS+=($!)
sleep 5

# ==================================================
# 3. 啟動 Device A 本地的 Control 與 Compute Nodes
# ==================================================
echo "🚀 Starting Control Node cluster_1..."
CLUSTER_NAME="cluster_1" \
EFO_URL="$EFO_URL" \
PORT="$CTRL1_PORT" \
CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
uvicorn control_node_server:app --host 0.0.0.0 --port "$CTRL1_PORT" \
>> "$LOG_PATH/control_1.log" 2>&1 &
PIDS+=($!)
sleep 2

echo "🚀 Starting 2 Compute Nodes for cluster_1..."
for n in 1 2; do
  COMP_PORT=$((BASE_PORT + 200 + n))
  NODE_ID="c1-n${n}"
  GPU_ID=$((n-1))

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  NODE_ID="$NODE_ID" \
  CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
  PORT="$COMP_PORT" \
  uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" \
  >> "$LOG_PATH/compute_${NODE_ID}.log" 2>&1 &
  PIDS+=($!)
done

# ==================================================
# 4. 暫停等待 Device B 啟動
# ==================================================
echo ""
echo "==========================================================="
echo "⏸️  【請注意】EFO 和 Tunnel 已經就緒！"
echo "👉 請現在切換到 Device B，執行 ./start_deviceB.sh"
echo "👉 確認 Device B 啟動完成後，回到這裡按下 [Enter] 鍵繼續..."
echo "==========================================================="
# 暫停並等待使用者按下 Enter
read -r

# ==================================================
# 5. 執行實驗
# ==================================================
export TARGET_CLUSTERS='["cluster_1", "cluster_2"]'
export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT}, \"cluster_2\":${CTRL2_PORT}}"

echo "🧪 Running test_simulation.py..."
set +e
python test_simulation.py
SIM_EXIT_CODE=$?
set -e

echo "🧾 Simulation finished with exit code: $SIM_EXIT_CODE"
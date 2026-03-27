#!/usr/bin/env bash
set -euo pipefail

PIDS=()
TUNNEL_PID=""

# === 實驗與路徑設定 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOG_PATH="./experiment_deviceA_logs"
export LORA_METADATA_PATH="./information/lora_metadata.json"
export SIMULATION="0"
mkdir -p "$LOG_PATH"

# === Port 設定 ===
BASE_PORT=8000
EFO_PORT=$((BASE_PORT + 900))        # 8900
CTRL1_PORT=$((BASE_PORT + 100))      # 8100 (Device A 本地)
CTRL2_PORT=$((BASE_PORT + 102))      # 8102 (透過 Tunnel 導向 Device B)
export EFO_URL="http://127.0.0.1:${EFO_PORT}"

# === 清理與停止函式 ===
stop() {
  echo ""
  echo "🛑 Stopping Device A services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done

  # 自動關閉 SSH Tunnel
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "🔌 Closing SSH Tunnel (PID: $TUNNEL_PID)..."
    kill "$TUNNEL_PID" || true
  fi
  echo "✅ Stopped."
}
# 無論是正常結束 (EXIT)、錯誤中斷 (ERR) 或 Ctrl+C (INT)，都會觸發 stop 關閉 Tunnel
trap stop EXIT INT TERM ERR

echo "🧹 Clearing old logs..."
rm -f "$LOG_PATH"/*.log || true

# ==================================================
# 1. 建立 SSH 雙向 Tunnel
# ==================================================
echo "🔗 Establishing SSH Tunnel to Device B (140.112.20.183)..."
# 注意：這裡不加 -f，改用 & 將其放入背景，這樣我們才能抓到它的 PID 來自動關閉
ssh -N -p 6666 \
  -L ${CTRL2_PORT}:127.0.0.1:${CTRL2_PORT} \
  -R ${EFO_PORT}:127.0.0.1:${EFO_PORT} \
  glenchen@140.112.20.183 &

TUNNEL_PID=$!
echo "✅ Tunnel established in background (PID: $TUNNEL_PID)"
sleep 3 # 等待 Tunnel 穩定建立

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
  COMP_PORT=$((BASE_PORT + 200 + n)) # 8201, 8202
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
read -p ""

# ==================================================
# 5. 執行實驗
# ==================================================
export TARGET_CLUSTERS='["cluster_1", "cluster_2"]'
# 因為 Tunnel 的關係，cluster_2 對 test_simulation 來說也是在 127.0.0.1
export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT}, \"cluster_2\":${CTRL2_PORT}}"

echo "🧪 Running test_simulation.py..."
set +e
python test_simulation.py
SIM_EXIT_CODE=$?
set -e

echo "🧾 Simulation finished with exit code: $SIM_EXIT_CODE"
# 腳本結束時，trap 會自動觸發 stop() 關閉 Tunnel 與所有 Node
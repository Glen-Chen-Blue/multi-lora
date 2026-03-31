#!/usr/bin/env bash
set -euo pipefail

export START_TIME=$((86400*2))

SSH_SOCKET="/tmp/multi_lora_tunnel.sock"
REMOTE_USER="glenchen"
REMOTE_HOST="140.112.20.183"

BASE_PORT=8000
EFO_PORT=$((BASE_PORT + 900))
CTRL2_PORT=$((BASE_PORT + 102))

echo "🚀 開始自動連續執行雙 Cluster 實驗 (Experiment 1_1 ~ 1_6)..."

# 1. 建立持久化的 SSH Tunnel
echo "🔗 Establishing SSH Tunnel to Device B ($REMOTE_HOST)..."
echo "🔑 請輸入一次密碼建立連線："
ssh -M -S "$SSH_SOCKET" -f -N -p 6666 \
  -o StrictHostKeyChecking=accept-new \
  -L ${CTRL2_PORT}:127.0.0.1:${CTRL2_PORT} \
  -R ${EFO_PORT}:127.0.0.1:${EFO_PORT} \
  ${REMOTE_USER}@${REMOTE_HOST}

echo "✅ Tunnel established!"

# 確保腳本中斷或結束時，關閉 Tunnel
trap 'echo "🔌 Closing SSH Tunnel..."; ssh -S "$SSH_SOCKET" -O exit ${REMOTE_USER}@${REMOTE_HOST} 2>/dev/null || true' EXIT INT TERM

# 2. 依序執行 6 個實驗
for i in 1 2 3 5 6; do
  script_file="start_deviceA_${i}.sh"
  
  echo ""
  echo "=================================================="
  echo "▶️ 準備執行: $script_file"
  echo "=================================================="
  
  ./"$script_file"
  EXIT_CODE=$?
  
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌ $script_file 執行失敗 (Exit code: $EXIT_CODE)，中斷後續實驗。"
    exit $EXIT_CODE
  fi

  echo "✅ $script_file 執行完成！"
  
  if [ "$i" -ne 6 ]; then
    echo "⏳ 暫停 60 秒鐘，等待 TCP Port (TIME_WAIT) 完全釋放..."
    sleep 60
  fi
done

echo "🎉 所有雙 Cluster 實驗 (1_1 ~ 1_6) 皆已順利執行完畢！"
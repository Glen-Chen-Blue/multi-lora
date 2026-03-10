#!/usr/bin/env bash
set -euo pipefail

# 設定模擬開始前的時間偏移 (此處設定為 2 天，可依需求修改)
export START_OFFSET=$((86400))

echo "🚀 開始自動連續執行實驗 (Experiment 1_1 ~ 1_6)..."
echo "🕒 已經設定 START_OFFSET = $START_OFFSET"

for i in {2..6}; do
  script_file="start_experiment1_${i}.sh"
  
  echo ""
  echo "=================================================="
  echo "▶️ 準備執行: $script_file"
  echo "=================================================="
  
  # 檢查檔案是否存在
  if [[ ! -f "$script_file" ]]; then
    echo "❌ 找不到檔案: $script_file，腳本中斷。"
    exit 1
  fi

  # 確保有執行權限
  if [[ ! -x "$script_file" ]]; then
    chmod +x "$script_file"
  fi
  
  # 執行實驗腳本
  ./"$script_file"
  
  EXIT_CODE=$?
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌ $script_file 執行失敗 (Exit code: $EXIT_CODE)，中斷後續實驗。"
    exit $EXIT_CODE
  fi

  echo "✅ $script_file 執行完成！"
  
  # 暫停 5 秒鐘讓系統資源 (Port, VRAM 等) 有充裕時間完全釋放
  sleep 5
done

echo "🎉 所有實驗 (1_1 ~ 1_6) 皆已順利執行完畢！"
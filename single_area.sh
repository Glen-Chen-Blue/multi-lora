#!/usr/bin/env bash
trap stop INT TERM
set -e

PIDS=()

start() {
  echo "=== Mode 1: 1 EFO -> 1 Control -> 2 Compute (Distributed LoRA Storage) ==="

  # 0. 準備模擬的分散式儲存環境
  echo "📂 Checking storage directories..."
  
  # 建立主目錄
  mkdir -p lora_repo

  # --- EFO (Source of Truth) 處理邏輯 ---
  # 如果 lora_repo/efo 存在且不為空，就不重新複製
  if [ -d "lora_repo/efo" ] && [ "$(ls -A lora_repo/efo)" ]; then
    echo "✅ EFO storage found (lora_repo/efo). Skipping copy from ./testLoRA."
    echo "   (If you want to update EFO data, delete lora_repo/efo and restart)"
  else
    echo "📂 EFO storage empty or missing. Initializing from ./testLoRA..."
    rm -rf lora_repo/efo
    mkdir -p lora_repo/efo
    
    if [ -d "./testLoRA" ]; then
      cp -r ./testLoRA/* lora_repo/efo/
      echo "✅ Seeded EFO storage from ./testLoRA"
    else
      echo "⚠️  Warning: ./testLoRA not found. EFO will be empty."
    fi
  fi

  # --- Control & Compute Nodes (Cache) 處理邏輯 ---
  # 為了確保模擬準確性，每次啟動時重置 Cache 節點的儲存空間
  # 如果你也希望這些節點資料保留，可以註解掉下面這行 rm -rf
  echo "🧹 Resetting cache directories for Control/Compute nodes..."
  rm -rf lora_repo/control lora_repo/cn_1 lora_repo/cn_2
  
  mkdir -p lora_repo/control
  mkdir -p lora_repo/cn_1
  mkdir -p lora_repo/cn_2

  # 1. 啟動 EFO Server (Port 9080)
  echo "Starting EFO Server..."
  LORA_PATH="./lora_repo/efo" \
  uvicorn EFO_server:app --host 0.0.0.0 --port 9080 &
  PIDS+=($!)
  sleep 2

  # 2. 啟動 Compute Node 1 (Port 8001)
  echo "Starting Compute Node 1..."
  CUDA_VISIBLE_DEVICES=0 \
  NODE_ID=cn-1 \
  MAX_BATCH_SIZE=64 \
  LORA_PATH="./lora_repo/cn_1" \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  uvicorn compute_node_server:app --port 8001 &
  PIDS+=($!)

  # 3. 啟動 Compute Node 2 (Port 8002)
  echo "Starting Compute Node 2..."
  CUDA_VISIBLE_DEVICES=1 \
  NODE_ID=cn-2 \
  MAX_BATCH_SIZE=64 \
  LORA_PATH="./lora_repo/cn_2" \
  CONTROL_NODE_URL="http://127.0.0.1:9000" \
  uvicorn compute_node_server:app --port 8002 &
  PIDS+=($!)

  # [保留修復] 增加等待時間，讓 Compute Node 有時間載入 LLM
  echo "Waiting 20 seconds for compute nodes to warm up..."
  sleep 20

  # 4. 啟動 Control Node (Port 9000)
  echo "Starting Control Node..."
  EFO_URL="http://127.0.0.1:9080" \
  MY_NODE_URL="http://127.0.0.1:9000" \
  COMPUTE_NODES="http://127.0.0.1:8001,http://127.0.0.1:8002" \
  LORA_PATH="./lora_repo/control" \
  uvicorn control_node_server:app --host 0.0.0.0 --port 9000 &
  PIDS+=($!)

  echo "✅ All services started. Press Ctrl+C to stop."
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

  # 強制殺死殘留進程
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing PID $pid"
      kill -9 "$pid"
    fi
  done

  echo "All services stopped."
}

# 執行啟動
start
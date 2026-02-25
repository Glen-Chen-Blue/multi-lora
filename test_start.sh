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

  # 清理 EFO
  kill_port 9100
  
  # 清理 Control Nodes (9000~9002)
  for p in {9000..9002}; do kill_port $p; done
  
  # 清理 Compute Nodes (8001~8003, 8011~8013, 8021~8023)
  for p in {8001..8003}; do kill_port $p; done
  for p in {8011..8013}; do kill_port $p; done
  for p in {8021..8023}; do kill_port $p; done

  echo "=== Phase 1: Infrastructure Test (Dynamic Registration Mode) ==="
  echo "📂 Using centralized configurations from config.py"

  echo "🚀 Starting EFO (Port 9100)..."
  PORT=9100 \
  CLUSTERS='{"cluster_1":"http://127.0.0.1:9000","cluster_2":"http://127.0.0.1:9001","cluster_3":"http://127.0.0.1:9002"}' \
  uvicorn EFO_server:app --host 0.0.0.0 --port 9100 &
  PIDS+=($!)

  sleep 5

  echo "🚀 Starting Control Nodes..."
  for c in 1 2 3; do
    CTRL_PORT=$((8999 + c)) # 9000, 9001, 9002
    echo "   -> Control Node cluster_$c (Port $CTRL_PORT)"
    
    CLUSTER_NAME="cluster_$c" \
    EFO_URL="http://127.0.0.1:9100" \
    PORT=$CTRL_PORT \
    LOCAL_URL="http://127.0.0.1:$CTRL_PORT" \
    uvicorn control_node_server:app --host 0.0.0.0 --port $CTRL_PORT &
    PIDS+=($!)
  done
  
  sleep 2

  echo "🚀 Starting Compute Nodes..."
  for c in 1 2 3; do
    CTRL_PORT=$((8999 + c)) # 取得對應的 Control Node Port
    echo "   --- Cluster $c (Connecting to Control Node $CTRL_PORT) ---"
    
    for n in 1 2 3; do
      # 產生 Compute Port: c=1 -> 800x, c=2 -> 801x, c=3 -> 802x
      COMP_PORT=$(( 8000 + (c-1)*10 + n )) 
      NODE_ID="c${c}-n${n}"
      
      echo "      -> Compute Node $NODE_ID (Port $COMP_PORT)"
      
      NODE_ID=$NODE_ID \
      CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
      PORT=$COMP_PORT \
      uvicorn compute_node_server:app --host 0.0.0.0 --port $COMP_PORT > /dev/null 2>&1 &
      PIDS+=($!)
    done
  done

  # ==========================================
  # ⏳ 等待節點註冊並觸發 /start
  # ==========================================
  echo ""
  echo "⏳ Waiting 5 seconds for all 3 clusters (12 nodes) to register with EFO..."
  sleep 5
  
  echo "🚦 Sending /start signal to EFO Server..."
  curl -X POST http://127.0.0.1:9100/start
  echo "" 

  echo "✅ All services started and EFO background tasks triggered. Press Ctrl+C to stop."
  wait
}

start
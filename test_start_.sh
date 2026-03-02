#!/usr/bin/env bash

PIDS=()

# ===============================
# 📂 Create log folder
# ===============================
mkdir -p logs

# ===============================
# 🧨 Kill port helper
# ===============================
kill_port() {
  PORT=$1

  # 檢查 port 是否有被 listen（bash 內建）
  (echo > /dev/tcp/127.0.0.1/$PORT) >/dev/null 2>&1

  if [ $? -eq 0 ]; then
    echo "⚠️  Port $PORT is in use → killing uvicorn on this port"

    # 只殺掉 uvicorn（避免誤殺別的東西）
    pkill -f "uvicorn.*--port $PORT"

    sleep 0.5
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

  echo "🧹 Clearing old logs..."
  rm -f logs/*.log

  echo "🧹 Cleaning target ports..."

  kill_port 9900
  for p in {9000..9002}; do kill_port $p; done
  for p in {8001..8003}; do kill_port $p; done
  for p in {8011..8013}; do kill_port $p; done
  for p in {8021..8023}; do kill_port $p; done

  echo "=== Phase 1: Infrastructure Test (Dynamic Registration Mode) ==="

  echo "🚀 Starting EFO (Port 9900)..."
  PORT=9900 \
  CLUSTERS='{"cluster_1":"http://127.0.0.1:9000","cluster_2":"http://127.0.0.1:9001","cluster_3":"http://127.0.0.1:9002"}' \
  uvicorn EFO_server:app --host 0.0.0.0 --port 9900 \
  >> logs/efo.log 2>&1 &
  PIDS+=($!)

  sleep 5

  echo "🚀 Starting Control Nodes..."
  for c in 1 2 3; do
    CTRL_PORT=$((8999 + c))
    echo "   -> Control Node cluster_$c (Port $CTRL_PORT)"
    
    CLUSTER_NAME="cluster_$c" \
    EFO_URL="http://127.0.0.1:9900" \
    PORT=$CTRL_PORT \
    CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
    uvicorn control_node_server:app --host 0.0.0.0 --port $CTRL_PORT \
    >> logs/control_${c}.log 2>&1 &
    PIDS+=($!)
  done
  
  sleep 2

  echo "🚀 Starting Compute Nodes..."
  for c in 1 2 3; do
    CTRL_PORT=$((8999 + c))
    echo "   --- Cluster $c (Connecting to Control Node $CTRL_PORT) ---"
    
    for n in 1 2 3; do
      COMP_PORT=$(( 8000 + (c-1)*10 + n ))
      NODE_ID="c${c}-n${n}"
      
      echo "      -> Compute Node $NODE_ID (Port $COMP_PORT)"
      
      NODE_ID=$NODE_ID \
      CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
      PORT=$COMP_PORT \
      uvicorn compute_node_server:app --host 0.0.0.0 --port $COMP_PORT \
      >> logs/compute_${NODE_ID}.log 2>&1 &
      PIDS+=($!)
    done
  done

  echo ""
  echo "⏳ Waiting 5 seconds for all clusters to register with EFO..."
  sleep 5
  

  echo "✅ All services started. Logs are in ./logs/"
  echo "Press Ctrl+C to stop."

  wait
}

start
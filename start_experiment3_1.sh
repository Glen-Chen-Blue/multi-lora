#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65535

# ---- safety: must be bash ----
if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: Please run this script with bash (not sh)." >&2
  exit 1
fi

# ---- helpful error trace ----
trap 'echo "❌ Error on line $LINENO. Exit code: $?" >&2' ERR

PIDS=()

# ==================================================
# 📦 Experiment Configuration (export to all services)
# ==================================================

# Logs
export LOG_PATH="./experiment_1_logs"

# Metadata
export LORA_METADATA_PATH="./information/lora_metadata.json"

# ==================================================
# 🌐 Port Configuration
# ==================================================

BASE_PORT=8000

EFO_PORT=$((BASE_PORT + 900))        # 8900
CTRL_BASE_PORT=$((BASE_PORT + 100))  # 8100..8102
COMP_BASE_PORT=$((BASE_PORT + 200))  # 8201..

# ==================================================
# 🌍 Service URLs
# ==================================================

export EFO_URL="http://127.0.0.1:${EFO_PORT}"

# ==================================================
# 🧪 Simulation Configuration
# ==================================================

export TARGET_CLUSTERS='["cluster_1","cluster_2","cluster_3"]'

CTRL1_PORT=$((CTRL_BASE_PORT + 0))
CTRL2_PORT=$((CTRL_BASE_PORT + 1))
CTRL3_PORT=$((CTRL_BASE_PORT + 2))

export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT},\"cluster_2\":${CTRL2_PORT},\"cluster_3\":${CTRL3_PORT}}"

# ==================================================
# 📂 Prepare log folder
# ==================================================

mkdir -p "$LOG_PATH"

# ==================================================
# 🧨 Kill port helper
# ==================================================

kill_port() {
  local PORT="$1"

  if (echo > /dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is in use → killing uvicorn on this port"
    pkill -f "uvicorn.*--port $PORT" || true
    sleep 0.5
  fi
}

# ==================================================
# 🛑 Stop handler
# ==================================================

stop() {
  echo ""
  echo "🛑 Stopping services..."

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done

  sleep 1
  echo "✅ Stopped."
}

trap stop INT TERM

# ==================================================
# 🚀 Start services
# ==================================================

start() {

  echo "🧹 Clearing old logs..."
  rm -f "$LOG_PATH"/*.log || true

  echo "🧹 Cleaning target ports..."

  # Kill EFO
  kill_port "$EFO_PORT"

  # Kill Control Nodes
  for c in 1 2 3; do
    CTRL_PORT=$((CTRL_BASE_PORT + (c-1)))
    kill_port "$CTRL_PORT"
  done

  # Kill Compute Nodes
  for c in 1 2 3; do
    for n in 1 2 3; do
      COMP_PORT=$((COMP_BASE_PORT + (c-1)*10 + n))
      kill_port "$COMP_PORT"
    done
  done

  echo "=== Phase 1: Infrastructure Test (Dynamic Registration Mode) ==="

  # ==================================================
  # Start EFO
  # ==================================================

  echo "🚀 Starting EFO (Port $EFO_PORT)..."

  PORT="$EFO_PORT" \
  CLUSTERS="{\"cluster_1\":\"http://127.0.0.1:${CTRL1_PORT}\",\"cluster_2\":\"http://127.0.0.1:${CTRL2_PORT}\",\"cluster_3\":\"http://127.0.0.1:${CTRL3_PORT}\"}" \
  uvicorn EFO_server:app --host 0.0.0.0 --port "$EFO_PORT" \
  >> "$LOG_PATH/efo.log" 2>&1 &

  PIDS+=($!)

  sleep 5

  # ==================================================
  # Start Control Nodes
  # ==================================================

  echo "🚀 Starting Control Nodes..."

  for c in 1 2 3; do

    CTRL_PORT=$((CTRL_BASE_PORT + (c-1)))

    echo "   -> Control Node cluster_$c (Port $CTRL_PORT)"

    CLUSTER_NAME="cluster_$c" \
    EFO_URL="$EFO_URL" \
    PORT="$CTRL_PORT" \
    CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
    uvicorn control_node_server:app --host 0.0.0.0 --port "$CTRL_PORT" \
    >> "$LOG_PATH/control_${c}.log" 2>&1 &

    PIDS+=($!)

  done

  sleep 2

  # ==================================================
  # Start Compute Nodes
  # ==================================================

  echo "🚀 Starting Compute Nodes..."

  for c in 1 2 3; do

    CTRL_PORT=$((CTRL_BASE_PORT + (c-1)))

    echo "   --- Cluster $c (Connecting to Control Node $CTRL_PORT) ---"

    for n in 1 2 3; do

      COMP_PORT=$((COMP_BASE_PORT + (c-1)*10 + n))
      NODE_ID="c${c}-n${n}"

      echo "      -> Compute Node $NODE_ID (Port $COMP_PORT)"

      NODE_ID="$NODE_ID" \
      CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
      PORT="$COMP_PORT" \
      uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" \
      >> "$LOG_PATH/compute_${NODE_ID}.log" 2>&1 &

      PIDS+=($!)

    done
  done

  echo ""
  echo "⏳ Waiting 20 seconds for all clusters to register with EFO..."
  sleep 20

  echo "✅ All services started."
  echo "   EFO:          $EFO_URL"
  echo "   ControlNodes: $CTRL1_PORT, $CTRL2_PORT, $CTRL3_PORT"
  echo "   Logs:         $LOG_PATH/"
  echo ""

  # ==================================================
  # 🧪 Run Simulation
  # ==================================================

  echo "🧪 Running test_simulation.py..."
  echo "   EFO_URL=$EFO_URL"
  echo "   TARGET_CLUSTERS=$TARGET_CLUSTERS"
  echo "   CLUSTER_PORT_MAP=$CLUSTER_PORT_MAP"
  echo ""

  set +e
  python test_simulation.py
  SIM_EXIT_CODE=$?
  set -e

  echo ""
  echo "🧾 Simulation finished with exit code: $SIM_EXIT_CODE"

  echo "🛑 Shutting down services..."
  stop

  exit "$SIM_EXIT_CODE"
}

start
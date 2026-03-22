#!/usr/bin/env bash
set -euo pipefail

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

# (optional) CUDA allocator config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs
export LOG_PATH="./experiment_single_cluster_2nodes2_logs"

# Metadata (如果你服務端會讀這個 env)
export LORA_METADATA_PATH="./information/lora_metadata_without_substitutes.json"

export SIMULATION="1"

# ==================================================
# 🌐 Port Configuration
# ==================================================

BASE_PORT=7000

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

export TARGET_CLUSTERS='["cluster_1"]'

CTRL1_PORT=$((CTRL_BASE_PORT + 0))

export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT}}"

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

  # Kill Control Node (only 1)
  kill_port "$CTRL1_PORT"

  # Kill Compute Nodes (2 nodes)
  for n in 1 2; do
    COMP_PORT=$((COMP_BASE_PORT + n))  # 8001, 8002
    kill_port "$COMP_PORT"
  done

  echo "=== Phase 1: Infrastructure Test (Single Cluster, 2 Compute Nodes) ==="

  # ==================================================
  # Start EFO
  # ==================================================

  echo "🚀 Starting EFO (Port $EFO_PORT)..."

  PORT="$EFO_PORT" \
  CLUSTERS="{\"cluster_1\":\"http://127.0.0.1:${CTRL1_PORT}\"}" \
  uvicorn EFO_server:app --host 0.0.0.0 --port "$EFO_PORT" \
  >> "$LOG_PATH/efo.log" 2>&1 &

  PIDS+=($!)

  sleep 5

  # ==================================================
  # Start Control Node (cluster_1)
  # ==================================================

  echo "🚀 Starting Control Nodes..."
  echo "   -> Control Node cluster_1 (Port $CTRL1_PORT)"

  CLUSTER_NAME="cluster_1" \
  EFO_URL="$EFO_URL" \
  PORT="$CTRL1_PORT" \
  CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
  uvicorn control_node_server:app --host 0.0.0.0 --port "$CTRL1_PORT" \
  >> "$LOG_PATH/control_1.log" 2>&1 &

  PIDS+=($!)

  sleep 2

  # ==================================================
  # Start Compute Nodes (2 nodes)
  # ==================================================

  echo "🚀 Starting Compute Nodes..."
  echo "   --- Cluster 1 (Connecting to Control Node $CTRL1_PORT) ---"

  for n in 1 2; do
    COMP_PORT=$((COMP_BASE_PORT + n))   # 8001, 8002
    NODE_ID="c1-n${n}"
    GPU_ID=$((n-1))                     # n=1 -> GPU0, n=2 -> GPU1

    echo "      -> Compute Node $NODE_ID (Port $COMP_PORT, GPU $GPU_ID)"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    NODE_ID="$NODE_ID" \
    CONTROL_NODE_URL="http://127.0.0.1:$CTRL1_PORT" \
    PORT="$COMP_PORT" \
    uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" \
    >> "$LOG_PATH/compute_${NODE_ID}.log" 2>&1 &

    PIDS+=($!)
  done

  echo ""
  echo "⏳ Waiting 20 seconds for cluster to register with EFO..."
  sleep 20

  echo "✅ All services started."
  echo "   EFO:          $EFO_URL"
  echo "   ControlNodes: $CTRL1_PORT"
  echo "   Logs:         $LOG_PATH/"
  echo ""

  # ==================================================
  # 🧪 Run Simulation (仿照 experiment1/2)
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
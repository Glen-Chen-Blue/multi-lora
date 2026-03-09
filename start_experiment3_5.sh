#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65535

# ---- safety: must be bash ----
if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: Please run this script with bash (not sh)." >&2
  exit 1
fi

trap 'echo "❌ Error on line $LINENO. Exit code: $?" >&2' ERR

PIDS=()

# ==================================================
# 📦 Experiment Configuration (export to all services)
# ==================================================
export LOG_PATH="./experiment_5_logs"
# Random/LRU 不考慮語意替代，所以套用 without_substitutes
export LORA_METADATA_PATH="./information/lora_metadata_without_substitutes.json"

export DISK_CAPACITY_GB=2.5       # 每個 Cluster 硬碟的 LoRA 儲存容量上限 (GB)


# ==================================================
# 🌐 Port Configuration
# ==================================================
BASE_PORT=4000

EFO_PORT=$((BASE_PORT + 900))        # 5900
CTRL_BASE_PORT=$((BASE_PORT + 100))  # 5100..5102
COMP_BASE_PORT=$((BASE_PORT + 200))  # 5201..

export EFO_URL="http://127.0.0.1:${EFO_PORT}"
export TARGET_CLUSTERS='["cluster_1","cluster_2","cluster_3"]'

CTRL1_PORT=$((CTRL_BASE_PORT + 0))
CTRL2_PORT=$((CTRL_BASE_PORT + 1))
CTRL3_PORT=$((CTRL_BASE_PORT + 2))

export CLUSTER_PORT_MAP="{\"cluster_1\":${CTRL1_PORT},\"cluster_2\":${CTRL2_PORT},\"cluster_3\":${CTRL3_PORT}}"

mkdir -p "$LOG_PATH"

kill_port() {
  local PORT="$1"
  if (echo > /dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    pkill -f "uvicorn.*--port $PORT" || true
    sleep 0.5
  fi
}

stop() {
  echo "🛑 Stopping services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" || true; fi
  done
  sleep 1
  echo "✅ Stopped."
}
trap stop INT TERM

start() {
  echo "🧹 Clearing old logs..."
  rm -f "$LOG_PATH"/*.log || true

  kill_port "$EFO_PORT"
  for c in 1 2 3; do kill_port $((CTRL_BASE_PORT + (c-1))); done
  for c in 1 2 3; do
    for n in 1 2 3; do kill_port $((COMP_BASE_PORT + (c-1)*10 + n)); done
  done

  echo "=== Phase 1: Start EFO (LRU Baseline) ==="
  PORT="$EFO_PORT" \
  CLUSTERS="{\"cluster_1\":\"http://127.0.0.1:${CTRL1_PORT}\",\"cluster_2\":\"http://127.0.0.1:${CTRL2_PORT}\",\"cluster_3\":\"http://127.0.0.1:${CTRL3_PORT}\"}" \
  uvicorn EFO_server_dlora:app --host 0.0.0.0 --port "$EFO_PORT" \
  >> "$LOG_PATH/efo.log" 2>&1 &
  PIDS+=($!)
  sleep 5

  echo "=== Phase 2: Start Control Nodes (Random Policy) ==="
  for c in 1 2 3; do
    CTRL_PORT=$((CTRL_BASE_PORT + (c-1)))
    CLUSTER_NAME="cluster_$c" \
    EFO_URL="$EFO_URL" \
    PORT="$CTRL_PORT" \
    CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
    uvicorn control_node_server_dlora:app --host 0.0.0.0 --port "$CTRL_PORT" \
    >> "$LOG_PATH/control_${c}.log" 2>&1 &
    PIDS+=($!)
  done
  sleep 2

  echo "=== Phase 3: Start Compute Nodes ==="
  for c in 1 2 3; do
    CTRL_PORT=$((CTRL_BASE_PORT + (c-1)))
    for n in 1 2 3; do
      COMP_PORT=$((COMP_BASE_PORT + (c-1)*10 + n))
      NODE_ID="c${c}-n${n}"
      NODE_ID="$NODE_ID" \
      CONTROL_NODE_URL="http://127.0.0.1:$CTRL_PORT" \
      PORT="$COMP_PORT" \
      uvicorn compute_node_server:app --host 0.0.0.0 --port "$COMP_PORT" \
      >> "$LOG_PATH/compute_${NODE_ID}.log" 2>&1 &
      PIDS+=($!)
    done
  done

  echo "⏳ Waiting 20 seconds for all clusters to register with EFO..."
  sleep 20

  echo "🧪 Running test_simulation.py..."
  set +e
  python test_simulation.py
  SIM_EXIT_CODE=$?
  set -e

  echo "🧾 Simulation finished with exit code: $SIM_EXIT_CODE"
  stop
  exit "$SIM_EXIT_CODE"
}

start
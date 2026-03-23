# Discrete-Time Multi-LoRA Simulation

A single-process, class-based, discrete-time simulation replacing the original multi-process HTTP-based system (EFO + Control Node + Compute Node via FastAPI/uvicorn).

## File Structure (12 files, 2,456 lines)

```
discrete_sim/
├── __init__.py              (1 line)   - Package init
├── sim_types.py             (122 lines) - SimRequest, SimulationConfig, ExperimentDef, 6 experiment configs
├── sim_clock.py             (62 lines)  - Discrete clock + heapq event scheduler (1ms granularity)
├── sim_network.py           (51 lines)  - Shifted lognormal network delay simulator
├── sim_engine.py            (416 lines) - Discrete-time MultiLoRAEngine (state machine: IDLE→LOAD→PREFILL→DECODE)
├── sim_compute_node.py      (114 lines) - Wraps engine, manages node status (active/standby/draining)
├── sim_control_node.py      (654 lines) - Base + 4 subclasses: SP2/Random/LRU/DLoRA
├── sim_efo.py               (504 lines) - Base + 3 subclasses: SP1(CSG-Swap)/LRU/DLoRA(LFU+decay)
├── sim_logger.py            (61 lines)  - JSONL logger (cost2.py compatible format)
├── sim_trace_reader.py      (62 lines)  - CSV trace reader
├── simulation.py            (293 lines) - Top-level Simulation class + main loop
└── run_experiments.py       (116 lines) - CLI entry point
```

## 6 Experiment Configurations

| Exp | EFO | Control Node | Metadata | Disk (GB) | Dispatch |
|-----|-----|-------------|----------|-----------|----------|
| 1 | SP1 (CSG-Swap) | SP2 (Lyapunov) | lora_metadata.json (with substitutes) | 5.0 | lyapunov |
| 2 | SP1 | SP2 | lora_metadata_without_substitutes.json | 5.0 | lyapunov |
| 3 | SP1 | Random | without_substitutes | 5.0 | random |
| 4 | LRU | LRU | without_substitutes | 2.0 | random |
| 5 | dLoRA (LFU+decay) | dLoRA | without_substitutes | 2.2 | greedy |
| 6 | LRU | LRU (greedy) | without_substitutes | 2.0 | greedy |

## Architecture

```
┌──────────────────────────────────────────────────┐
│              SimEFO (Global Orchestrator)          │
│  SP1: Hourly provisioning (CSG-Swap / LRU / LFU) │
│  SP2: 3-second routing broadcast                  │
│  Metrics logging (every SP1/10)                   │
└──────────┬───────────────────────────┬────────────┘
           │                           │
  ┌────────▼──────────┐     ┌──────────▼──────────┐
  │ SimControlNode     │     │ SimControlNode       │
  │ (cluster_1)        │     │ (cluster_2)          │
  │ Admission control  │     │ ...                  │
  │ Scheduler (500ms)  │     │                      │
  │ Auto-scaling (1s)  │     │                      │
  └──┬─────────┬───────┘     └──────────────────────┘
     │         │
┌────▼───┐ ┌──▼─────┐
│Compute │ │Compute │
│Node 1  │ │Node 2  │
│(Engine)│ │(Engine)│
└────────┘ └────────┘
```

## Core Design

### Time Model
- All components share a single `SimClock` instance (1ms granularity)
- Original `time.sleep()` calls replaced with countdown timers
- Periodic events (SP1, SP2, scheduler, logging) driven by clock callbacks

### Engine State Machine
```
IDLE → LOADING (66ms/adapter miss) → PREFILL (65ms × N × multiplier) → DECODE (25ms + 1ms × batch × multiplier) → IDLE
```
- Merged mode: capacity=12, multiplier=0.861
- Unmerged mode: capacity=10 (requests + unique LoRAs)
- CPU LRU cache (max 30), GPU slot LRU (10 slots)

### Request Model (matches test_simulation.py)
- Payload: `{prompt: "test", adapter_id: "LoRA_{id}", max_new_tokens: 256}`
- Lifecycle: SEND → admit → schedule → engine process → first token (TTFT) → finish
- Console output: `[HH:MM:SS] [SEND/DONE/DROP/FAIL] Req:N/Total | Target:LoRA_X ...`

### Log Compatibility
- Outputs `efo_global_metrics.log` in JSONL format
- Directly readable by existing `cost2.py` for chart generation

### Configurable Topology
```python
cluster_topology = {"cluster_1": 2, "cluster_2": 3}
# Creates:
#   cluster_1: 1 control node + 2 compute nodes
#   cluster_2: 1 control node + 3 compute nodes
#   1 EFO managing both control nodes
```

## Prerequisites

```bash
pip install pandas
```

## Usage

```bash
# Run single experiment (1 cluster, 2 compute nodes)
python3 -m discrete_sim.run_experiments \
    --experiment 1 \
    --topology '{"cluster_1": 2}' \
    --duration-hours 8 \
    --output-dir ./results/exp1/

# Run all 6 experiments
python3 -m discrete_sim.run_experiments --all \
    --topology '{"cluster_1": 2}' \
    --duration-hours 8 \
    --output-dir ./results/

# Custom topology (2 clusters, different node counts)
python3 -m discrete_sim.run_experiments \
    --experiment 1 \
    --topology '{"cluster_1": 3, "cluster_2": 2}' \
    --duration-hours 4

# Generate cost2.py chart from results
python3 -m discrete_sim.run_experiments --plot --output-dir ./results/
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment` | — | Experiment ID (1-6) |
| `--all` | — | Run all 6 experiments |
| `--plot` | — | Generate cost chart from results |
| `--topology` | `{"cluster_1": 2}` | JSON: `{cluster_name: num_compute_nodes}` |
| `--start-offset` | 172800 | CSV trace start offset (seconds) |
| `--duration-hours` | 8 | Simulation duration (hours) |
| `--target-clusters` | all keys | JSON list of target clusters |
| `--seed` | 42 | Random seed |
| `--output-dir` | `./results/` | Output directory |
| `--trace-csv` | `./information/simulation_data.csv` | Path to trace CSV |
| `--metadata-dir` | `./information/` | Path to metadata directory |

## Verified Behavior

- All 12 files pass Python syntax check
- `SimClock`: periodic scheduling fires correctly at expected intervals
- `SimMultiLoRAEngine`: 5-token request completes in 239ms (66ms load + 65ms prefill + decode)
- `SimComputeNode`: TTFT=132ms, 10-token request completes in 374ms
- Output log format compatible with `cost2.py` chart generation

## Performance Estimate

- 8-hour simulation = 28.8M iterations (1ms each)
- Each idle iteration is O(1)
- Estimated wall-clock: 30-120 seconds (pure Python, no I/O, no sleep)

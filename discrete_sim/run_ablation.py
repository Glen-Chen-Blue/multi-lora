"""
run_ablation.py  —  System Component Ablation Study (Figure 1)
===============================================================
Four Knobs, pure self-ablation (no external baselines).

Variants
--------
  ours         Full System  (K1+K2+K3+K4 all on, dynamic auto-scaling)
  no_semantic  w/o Semantic Substitution         (K1 off)
  no_provision w/o Predictive LoRA Provision     (K2 off)
  no_merge     w/o Dynamic Merge/Unmerge         (K3 off, always unmerge)
  no_autoscale w/o Auto-scaling                  (K4 off, fixed all-open)

Metrics (per variant)
---------------------
  max_throughput       highest total Req/s where P95 TTFT <= 6.0 s
  network_cost_per_req (downloads*3.0 + offloads*0.001) / total_requests
  gpu_compute_per_req  total_inference_time_ms / total_requests  [ms/req]
  gpu_active_node_ms   sum of all-node ACTIVE ms (resource consumption)

Architecture
------------
Each (variant, rps) combination runs in an isolated subprocess via
monkey-patching.  The master collects JSON results and draws bar charts.
"""

import os
import sys
import json
import argparse
import subprocess
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor

# ── Experiment parameters ────────────────────────────────────────────────────

VARIANTS = [
    ("ours",         "Ours\n(Full)"),
    ("no_semantic",  "w/o\nSemantic"),
    ("no_provision", "w/o\nProvision"),
    ("no_merge",     "w/o\nMerge"),
    ("no_autoscale", "w/o\nAutoScale"),
]

# RPS values to sweep for max-throughput search (total req/s across ALL clusters)
RPS_SEARCH = [10, 20, 30, 40, 50]

# Fixed RPS used when collecting cost / GPU metrics
FIXED_METRIC_RPS = 30
assert FIXED_METRIC_RPS in RPS_SEARCH, "FIXED_METRIC_RPS must be in RPS_SEARCH"

SLO_S       = 6.0   # P95 TTFT SLO (seconds)
OPTIMAL_V   = 8.0   # Lyapunov V for dynamic autoscale
CLUSTER_TOPOLOGY = {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}
DURATION_HOURS   = 4
ZIPF_S           = 1.2
MAX_WORKERS      = 12   # parallel subprocesses


# ── Subprocess launcher ──────────────────────────────────────────────────────

def _launch(variant: str, rps: float) -> dict:
    cmd = [sys.executable, __file__, "--worker",
           "--variant", variant, "--rps", str(rps)]
    print(f"[Master] start  variant={variant:<14s}  rps={rps:4.0f} Req/s")
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            data.update(variant=variant, rps=rps)
            ok = "✅" if data.get("p95_ttft", 99) <= SLO_S else "❌"
            print(f"[Master]   {ok}  variant={variant:<14s}  rps={rps:4.0f}"
                  f"  P95={data['p95_ttft']:.3f}s"
                  f"  net={data['network_cost_per_req']:.5f}"
                  f"  gpu={data['gpu_compute_per_req']:.1f}ms/req")
            return data
    print(f"[Master] ⚠️  variant={variant} rps={rps} FAILED\n"
          f"{res.stderr[-600:]}")
    return dict(variant=variant, rps=rps, p95_ttft=99.0,
                network_cost_per_req=0.0, gpu_compute_per_req=0.0,
                gpu_active_node_ms=0)


# ── Worker (runs inside the subprocess) ─────────────────────────────────────

def _worker(variant: str, rps: float):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # ── global accumulators (shared via list so closures can mutate) ──
    import config
    config.GLOBAL_TTFT_RECORDS   = []
    config.GLOBAL_GPU_ACTIVE_MS  = [0]

    # ── import modules that simulation.py actually uses ───────────────
    # IMPORTANT: simulation.py uses sim_control_node (no underscore)
    import discrete_sim.sim_control_node  as scn
    import discrete_sim.sim_compute_node  as scn_compute
    import discrete_sim.sim_efo           as efo_mod

    # ─────────────────────────────────────────────────────────────────
    # Patch 0: TTFT interceptor  (all variants)
    # ─────────────────────────────────────────────────────────────────
    _orig_first_token = scn.SimControlNodeBase._on_first_token
    def _new_first_token(self, req):
        _orig_first_token(self, req)
        if req.ttft_ms is not None:
            config.GLOBAL_TTFT_RECORDS.append(req.ttft_ms)
    scn.SimControlNodeBase._on_first_token = _new_first_token

    # ─────────────────────────────────────────────────────────────────
    # Patch 1: GPU Active Node tracker  (all variants)
    # ─────────────────────────────────────────────────────────────────
    _orig_node_step = scn_compute.SimComputeNode.step
    def _tracked_step(self):
        if self.status != scn_compute.NodeStatus.STANDBY:
            config.GLOBAL_GPU_ACTIVE_MS[0] += 1
        _orig_node_step(self)
    scn_compute.SimComputeNode.step = _tracked_step

    # ─────────────────────────────────────────────────────────────────
    # Patch 2: Unified improved scheduler  (all variants)
    # Improves queue handling (waits up to 60 s before drop)
    # Uses _ablation_allow_merge flag (set per variant below).
    # ─────────────────────────────────────────────────────────────────
    def _unified_scheduler(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        allow_merge = getattr(self, "_ablation_allow_merge", True)

        # -- mode-switch block --
        if allow_merge:
            MERGE_T   = max(1, scn.UNMERGED_CAPACITY - 1)
            UNMERGE_T = max(1, scn.UNMERGED_CAPACITY - 2)
            n_unmerged = sum(1 for v in v_nodes if v.mode == "unmerge")
            for v in v_nodes:
                if v.node.node_id in self.switching_nodes:
                    continue
                if (v.mode == "unmerge" and n_unmerged > 1
                        and v.running_batch >= MERGE_T
                        and len(v.active_loras) == 1):
                    aid = next(iter(v.active_loras))
                    v.node.merge_adapter(aid)
                    v.mode, v.merged_adapter = "merge", aid
                    n_unmerged -= 1
                elif v.mode == "merge" and v.running_batch < UNMERGE_T:
                    v.node.unmerge_all()
                    v.mode, v.merged_adapter = "unmerge", None
                    n_unmerged += 1
        else:
            # no_merge: force all nodes to stay unmerged
            for v in v_nodes:
                if v.node.node_id in self.switching_nodes:
                    continue
                if v.mode == "merge":
                    v.node.unmerge_all()
                    v.mode, v.merged_adapter = "unmerge", None

        # -- dispatch loop --
        dispatched = True
        while dispatched and self.pending_queue:
            dispatched = False
            for req in list(self.pending_queue):
                target_aid = req.original_adapter_id
                meta = self.lora_metadata.get(target_aid, {})
                subs = [s for s in meta.get("substitutes", [])
                        if s in self.local_available_loras]
                valid = [a for a in ([target_aid] + subs)
                         if a in self.local_available_loras]
                if not valid:
                    valid = [target_aid]

                best = None
                for aid in valid:
                    for v in v_nodes:
                        if v.node.node_id in self.switching_nodes:
                            continue
                        free = v.get_free_slots(aid)
                        if free <= 0:
                            continue
                        score = (
                            int(v.mode == "merge" and v.merged_adapter == aid),
                            int(v.mode == "unmerge" and aid in v.active_loras),
                            int(v.mode == "unmerge" and aid in v.loaded_adapters),
                            int(v.mode == "unmerge" and len(v.active_loras) == 0),
                            free,
                        )
                        if best is None or score > best[2]:
                            best = (v, aid, score)

                if best:
                    v, aid, _ = best
                    req.adapter_id = aid
                    v.commit_request(aid)
                    v.node.submit_request(req)
                    self.pending_queue.remove(req)
                    if not req.is_delegated:
                        self.Z_debt = max(0.0, self.Z_debt - scn.EPSILON)
                    dispatched = True
                    break
                else:
                    offloaded = False
                    if not req.is_delegated and self.offload_callback:
                        tgt = self._select_best_offload_target(target_aid)
                        if tgt:
                            offloaded = self.offload_callback(req, tgt=tgt)
                            if offloaded:
                                self.offload_out += 1
                                self.pending_queue.remove(req)
                                dispatched = True
                                break
                    if not offloaded:
                        waited_s = (self._clock.now() - req.arrival_time_ms) / 1000.0
                        if waited_s > 60.0:
                            self._handle_drop(
                                req, f"Congestion (waited {waited_s:.1f}s)")
                            if not req.is_delegated:
                                self.Z_debt += scn.PSI_DROP
                            self.recent_drops.append(self._clock.now())
                            self.pending_queue.remove(req)
                            dispatched = True
                            break
                        # else: keep waiting in queue

    scn.SimControlNodeSP2._scheduler_tick = _unified_scheduler

    # ─────────────────────────────────────────────────────────────────
    # Patch 3: Dynamic auto-scaling  (all variants EXCEPT no_autoscale)
    # ─────────────────────────────────────────────────────────────────
    def _dynamic_autoscale(self):
        if self.system_paused:
            return
        now = self._clock.now()
        thresh = max(1, int(OPTIMAL_V * 2))
        if (len(self.pending_queue) >= thresh or self.Z_debt >= thresh):
            if now - self._last_scale_time_ms > 4000:
                for node in self.compute_nodes:
                    if node.status == scn.NodeStatus.STANDBY:
                        node.activate()
                        self._last_scale_time_ms = now
                        self._surplus_duration_ms = 0
                        self.Z_debt = max(0.0, self.Z_debt - thresh)
                        break
                return

        active = [n for n in self.compute_nodes
                  if n.status == scn.NodeStatus.ACTIVE]
        if len(active) > 1:
            v_nodes  = self._get_virtual_node_states()
            pending  = len(self.pending_queue)
            free     = sum(v.get_free_slots("") for v in v_nodes)
            patience = max(2000, int(10000 / (OPTIMAL_V + 1)))
            surplus  = free - pending
            SCALE_DOWN_SURPLUS = getattr(scn, "SCALE_DOWN_SURPLUS_THRESHOLD", 10)
            if surplus >= SCALE_DOWN_SURPLUS:
                self._surplus_duration_ms += 1000
            else:
                self._surplus_duration_ms = 0
            if (self._surplus_duration_ms >= patience
                    and now - self._last_scale_time_ms > 6000):
                least = min(active, key=lambda n: n.engine.get_running_count())
                least.drain()
                self._last_scale_time_ms = now
                self._surplus_duration_ms = 0

    if variant != "no_autoscale":
        scn.SimControlNodeSP2._autoscale_tick = _dynamic_autoscale
    # no_autoscale: keep sim_control_node.py's original fixed-all-open behaviour

    # ─────────────────────────────────────────────────────────────────
    # Patch 4: inject _ablation_allow_merge flag via __init__ wrapper
    # ─────────────────────────────────────────────────────────────────
    _orig_sp2_init = scn.SimControlNodeSP2.__init__
    allow_merge_flag = (variant != "no_merge")
    def _patched_sp2_init(self_, *args, **kwargs):
        _orig_sp2_init(self_, *args, **kwargs)
        self_._ablation_allow_merge = allow_merge_flag
        # Also ensure _last_scale_time_ms and _surplus_duration_ms exist
        if not hasattr(self_, "_last_scale_time_ms"):
            self_._last_scale_time_ms = 0
        if not hasattr(self_, "_surplus_duration_ms"):
            self_._surplus_duration_ms = 0
    scn.SimControlNodeSP2.__init__ = _patched_sp2_init

    # ─────────────────────────────────────────────────────────────────
    # Patch 5: K2 — disable SP1 provisioning
    # ─────────────────────────────────────────────────────────────────
    if variant == "no_provision":
        efo_mod.SimEFOSP1._sp1_tick = lambda self_: None

    # ─────────────────────────────────────────────────────────────────
    # Build simulation config
    # K1 (no_semantic): use experiment_id=2 (no-substitutes metadata)
    # All others: experiment_id=1
    # ─────────────────────────────────────────────────────────────────
    experiment_id = 2 if variant == "no_semantic" else 1

    from discrete_sim.sim_types      import SimulationConfig
    from discrete_sim.simulation     import Simulation
    from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

    out_dir = os.path.join(project_root, "discrete_sim", "results",
                           f"ablation_{variant}_rps{int(rps)}")
    os.makedirs(out_dir, exist_ok=True)

    info_dir    = os.path.join(project_root, "information")
    lora_map    = os.path.join(info_dir, "lora_mapping.json")
    clusters    = list(CLUSTER_TOPOLOGY.keys())
    rps_per_cl  = rps / len(clusters)

    # Build synthetic generator (drop-in replacement for SimTraceReader)
    gen = SimSyntheticGenerator(
        lora_mapping_path=lora_map,
        duration_s=DURATION_HOURS * 3600,
        target_clusters=clusters,
        rps_per_cluster=rps_per_cl,
        zipf_s=ZIPF_S,
        seed=42,
    )

    # Monkey-patch SimTraceReader so Simulation uses synthetic events
    import discrete_sim.sim_trace_reader as trace_mod
    _orig_reader = trace_mod.SimTraceReader
    trace_mod.SimTraceReader = lambda *a, **kw: gen

    sim_config = SimulationConfig(
        experiment_id=experiment_id,
        cluster_topology=CLUSTER_TOPOLOGY,
        start_offset=0,
        duration_hours=DURATION_HOURS,
        output_dir=out_dir,
        trace_csv=os.path.join(info_dir, "simulation_data.csv"),
        metadata_dir=info_dir,
    )

    sim = Simulation(sim_config)
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        sim.run()

    trace_mod.SimTraceReader = _orig_reader   # restore (good practice)

    # ── Collect metrics ───────────────────────────────────────────────
    p95_s = 99.0
    if config.GLOBAL_TTFT_RECORDS:
        p95_s = float(np.percentile(config.GLOBAL_TTFT_RECORDS, 95)) / 1000.0

    net_cost = 0.0
    gpu_ms   = 0.0
    log_path = os.path.join(out_dir, "efo_global_metrics.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            tot      = json.loads(lines[-1]).get("efo_totals", {})
            n_req    = max(1, tot.get("total_requests", 1))
            inf_ms   = tot.get("total_inference_time", 0.0)
            offloads = tot.get("total_offloads", 0)
            dls      = tot.get("artifact_downloads", 0)
            net_cost = (offloads * 0.001 + dls * 3.0) / n_req
            gpu_ms   = inf_ms / n_req   # ms / req

    result = dict(
        p95_ttft            = p95_s,
        network_cost_per_req= net_cost,
        gpu_compute_per_req = gpu_ms,
        gpu_active_node_ms  = config.GLOBAL_GPU_ACTIVE_MS[0],
    )
    print(f"RESULT_JSON:{json.dumps(result)}")


# ── Master logic ─────────────────────────────────────────────────────────────

def _draw(summary: dict, out_dir: str):
    keys   = [v for v, _ in VARIANTS]
    labels = [summary[v]["label"] for v in keys]

    tputs  = [summary[v]["max_throughput"]        for v in keys]
    nets   = [summary[v]["network_cost_per_req"]   for v in keys]
    gpus   = [summary[v]["gpu_compute_per_req"]    for v in keys]

    # Normalise network cost to Ours = 1.0
    base   = nets[0] if nets[0] > 0 else 1e-9
    nets_n = [c / base for c in nets]

    x  = np.arange(len(keys))
    bw = 0.55
    colors = ["#1a6ea8", "#5b9ec9", "#88bcd8", "#b4d3e8", "#d9e8f4"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=150)
    fig.suptitle("System Component Ablation Study", fontsize=14,
                 fontweight="bold", y=1.01)

    def _plot(ax, vals, ylabel, title, higher_better, fmt="%.1f"):
        bars = ax.bar(x, vals, width=bw, color=colors,
                      edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        mv = max(vals) if vals else 1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mv * 0.015,
                    fmt % val, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        note = "↑ Higher is better" if higher_better else "↓ Lower is better"
        ax.text(0.97, 0.96, note, transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top", color="gray")

    _plot(axes[0], tputs,  "Max Stable Throughput (Req/s)",
          "(a) Max Throughput",  higher_better=True)
    _plot(axes[1], nets_n, "Network Cost / Request\n(Ours = 1.0, lower is better)",
          "(b) Network Cost per Request", higher_better=False)
    _plot(axes[2], gpus,   "GPU Compute Time / Request (ms/req)",
          "(c) GPU Compute Time per Request", higher_better=False)

    # Append GPU Active Node-Hours as footnote (K4 resource visibility)
    act_hrs = [summary[v]["gpu_active_node_ms"] / 3_600_000 for v in keys]
    if any(h > 0 for h in act_hrs):
        note = "GPU Node-Active Hours: " + "  |  ".join(
            f"{summary[v]['label'].replace(chr(10),' ')}={h:.1f}h"
            for v, h in zip(keys, act_hrs)
        )
        fig.text(0.5, -0.03, note, ha="center", fontsize=7.5,
                 color="#555", style="italic")

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_bar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Master] 📊 Bar chart saved → {path}")


def run_all():
    print("=" * 70)
    print("🚀 System Component Ablation Study")
    print(f"   Variants : {[v for v,_ in VARIANTS]}")
    print(f"   RPS scan : {RPS_SEARCH}")
    print(f"   Topology : {CLUSTER_TOPOLOGY}")
    print(f"   Duration : {DURATION_HOURS}h per run")
    print("=" * 70)

    tasks = [(v, r) for v, _ in VARIANTS for r in RPS_SEARCH]
    raw: dict = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_launch, v, r): (v, r) for v, r in tasks}
        for fut in futs:
            res = fut.result()
            v, r = futs[fut]
            raw.setdefault(v, {})[r] = res

    # ── Summarise ──────────────────────────────────────────────────────
    summary = {}
    for variant, label in VARIANTS:
        rmap = raw.get(variant, {})
        # Max throughput: highest RPS with P95 TTFT <= SLO
        max_tput = 0.0
        for r in sorted(RPS_SEARCH):
            if rmap.get(r, {}).get("p95_ttft", 99) <= SLO_S:
                max_tput = float(r)
        if max_tput == 0.0:
            max_tput = float(RPS_SEARCH[0])

        metric = rmap.get(FIXED_METRIC_RPS, {})
        summary[variant] = dict(
            label               = label,
            max_throughput      = max_tput,
            network_cost_per_req= metric.get("network_cost_per_req", 0.0),
            gpu_compute_per_req = metric.get("gpu_compute_per_req",  0.0),
            gpu_active_node_ms  = metric.get("gpu_active_node_ms",   0),
        )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Master] 💾 Results saved → {json_path}")

    _draw(summary, out_dir)
    print("[Master] ✅ Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker",  action="store_true")
    p.add_argument("--variant", type=str,   default="ours")
    p.add_argument("--rps",     type=float, default=30.0)
    args = p.parse_args()

    if args.worker:
        _worker(args.variant, args.rps)
    else:
        run_all()


if __name__ == "__main__":
    main()

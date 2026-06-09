"""
run_tau_sweep.py  —  Semantic Similarity Threshold (tau_sim) Sensitivity
=========================================================================
Figure 2: how tau_sim (semantic similarity threshold) affects substitution
rate and network download cost.

Design
------
tau_sim controls which adapters are considered "semantically equivalent"
for substitution at DISPATCH TIME only.  SP1 provisioning uses the FIXED
original metadata (DISTANCE_THRESHOLD = 0.10) throughout, so the set of
pre-loaded LoRAs is held constant across all sweep points.  This isolates
tau_sim as the sole independent variable.

To create meaningful cache pressure:
  * disk_capacity_gb = 4.0  (40 LoRA slots per cluster)
  * 10 mandatory local LoRAs per cluster → only 30 slots for global LoRAs
  * 70 global LoRAs total → 40 never provisioned in a given cluster
  * Requests for unprovisioned LoRAs must use semantic substitution or pay
    a download cost → substitution rate and network cost differ across tau.

tau_sim values  (evenly spaced 1.0 → 0.50, step 0.05)
---------------------------------------------------------------
  tau_sim | Euclidean d  | sub pairs (approx)
  --------|--------------|--------------------
  1.00    | 0.000        |   0   (no substitution)
  0.95    | 0.316        | ~300
  0.90    | 0.447        | ~500
  0.85    | 0.548        | ~700
  0.80    | 0.632        | ~900
  0.75    | 0.707        | ~1100
  0.70    | 0.775        | ~1300
  0.65    | 0.837        | ~1500
  0.60    | 0.894        | ~1700
  0.55    | 0.949        | ~1900
  0.50    | 1.000        | ~2100

System Default: tau_sim ≈ 0.995  (DISTANCE_THRESHOLD = 0.10, d=0.10)
  → included as an extra point between 1.00 and 0.95.

Metrics
-------
  substitution_rate    fraction of served requests that used a substitute
  network_cost_per_req (downloads*3.0 + offloads*0.001) / total_requests
  p95_ttft_s           P95 First-Token Time (seconds)
"""

import os
import sys
import json
import math
import argparse
import subprocess
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# ── Parameters ────────────────────────────────────────────────────────────────

# tau_sim values: evenly spaced 1.0 → 0.50 + system default at 0.995
TAU_SIM_MAIN   = [round(1.0 - i * 0.05, 2) for i in range(11)]  # [1.0, 0.95, ..., 0.50]
TAU_SIM_DEFAULT = 0.995   # corresponds to DISTANCE_THRESHOLD = 0.10
TAU_SIM_ALL    = sorted(set(TAU_SIM_MAIN + [TAU_SIM_DEFAULT]), reverse=True)

ORIGINAL_DIST_THRESHOLD = 0.10  # fixed threshold for SP1 provisioning

CLUSTER_TOPOLOGY = {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}
DURATION_HOURS   = 4
RPS_TOTAL        = 40             # total req/s across all clusters (higher = more cache pressure)
ZIPF_S           = 1.2
OPTIMAL_V        = 8.0
DISK_CAPACITY_GB = 4.0            # 40 slots per cluster → meaningful cache misses
MAX_WORKERS      = 8


def _tau_to_dist(tau_sim: float) -> float:
    """Approximate Euclidean distance from cosine similarity (unit-vector approx)."""
    return math.sqrt(max(0.0, 2.0 * (1.0 - tau_sim)))


# ── Metadata generation ───────────────────────────────────────────────────────

def _gen_metadata_for_dist(base_meta: dict, dist_threshold: float) -> dict:
    """Return metadata with substitutes recomputed for given Euclidean threshold."""
    new_meta = {}
    for lid, info in base_meta.items():
        new_meta[lid] = {k: v for k, v in info.items() if k != "substitutes"}
        new_meta[lid]["substitutes"] = []

    ids = list(new_meta.keys())
    for i, id1 in enumerate(ids):
        for id2 in ids[i + 1:]:
            i1, i2 = new_meta[id1], new_meta[id2]
            if i1.get("type") != i2.get("type"):
                continue
            if i1.get("type") == "local" and i1.get("cluster") != i2.get("cluster"):
                continue
            p1 = i1.get("pos") or base_meta[id1].get("pos")
            p2 = i2.get("pos") or base_meta[id2].get("pos")
            if p1 and p2:
                d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if d <= dist_threshold:
                    new_meta[id1]["substitutes"].append(id2)
                    new_meta[id2]["substitutes"].append(id1)
    return new_meta


# ── Subprocess launcher ───────────────────────────────────────────────────────

def _launch(tau_sim: float) -> dict:
    cmd = [sys.executable, __file__, "--worker", "--tau", str(tau_sim)]
    print(f"[Master] start  tau_sim={tau_sim:.3f}  (d={_tau_to_dist(tau_sim):.3f})")
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            data["tau_sim"] = tau_sim
            print(f"[Master]   tau={tau_sim:.3f}"
                  f"  subst={data.get('substitution_rate', 0)*100:.1f}%"
                  f"  net={data.get('network_cost_per_req', 0):.5f}"
                  f"  p95={data.get('p95_ttft_s', 0):.3f}s")
            return data
    print(f"[Master] ⚠️  tau={tau_sim:.3f} FAILED\n{res.stderr[-500:]}")
    return dict(tau_sim=tau_sim, substitution_rate=0.0,
                network_cost_per_req=0.0, p95_ttft_s=99.0)


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(tau_sim: float):
    dist_threshold = _tau_to_dist(tau_sim)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    import config
    config.GLOBAL_TTFT_RECORDS  = []
    config.GLOBAL_SUBST_COUNT   = [0]
    config.GLOBAL_TOTAL_SERVED  = [0]

    import discrete_sim.sim_control_node as scn
    import discrete_sim.sim_efo          as efo_mod

    # ── TTFT + substitution-rate interceptor ──────────────────────────
    _orig_ftk = scn.SimControlNodeBase._on_first_token
    def _new_ftk(self, req):
        _orig_ftk(self, req)
        if req.ttft_ms is not None:
            config.GLOBAL_TTFT_RECORDS.append(req.ttft_ms)
        config.GLOBAL_TOTAL_SERVED[0] += 1
        # A substitution occurred when the served adapter != the originally requested one
        if getattr(req, "original_adapter_id", None) and \
           req.adapter_id != req.original_adapter_id:
            config.GLOBAL_SUBST_COUNT[0] += 1
    scn.SimControlNodeBase._on_first_token = _new_ftk

    # ── Dynamic auto-scale ────────────────────────────────────────────
    def _dyn_autoscale(self):
        if self.system_paused:
            return
        now    = self._clock.now()
        thresh = max(1, int(OPTIMAL_V * 2))
        if len(self.pending_queue) >= thresh or self.Z_debt >= thresh:
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
            free     = sum(v.get_free_slots("") for v in v_nodes)
            pending  = len(self.pending_queue)
            patience = max(2000, int(10000 / (OPTIMAL_V + 1)))
            SURPLUS  = getattr(scn, "SCALE_DOWN_SURPLUS_THRESHOLD", 10)
            if (free - pending) >= SURPLUS:
                self._surplus_duration_ms += 1000
            else:
                self._surplus_duration_ms = 0
            if (self._surplus_duration_ms >= patience
                    and now - self._last_scale_time_ms > 6000):
                least = min(active, key=lambda n: n.engine.get_running_count())
                least.drain()
                self._last_scale_time_ms = now
                self._surplus_duration_ms = 0

    scn.SimControlNodeSP2._autoscale_tick = _dyn_autoscale

    # ── Improved scheduler with proper queueing + tau-aware substitution ──
    def _tau_scheduler(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        # Merge/unmerge switching (same as full-system)
        MERGE_T   = max(1, scn.UNMERGED_CAPACITY - 1)
        UNMERGE_T = max(1, scn.UNMERGED_CAPACITY - 2)
        n_un = sum(1 for v in v_nodes if v.mode == "unmerge")
        for v in v_nodes:
            if v.node.node_id in self.switching_nodes:
                continue
            if (v.mode == "unmerge" and n_un > 1
                    and v.running_batch >= MERGE_T
                    and len(v.active_loras) == 1):
                aid = next(iter(v.active_loras))
                v.node.merge_adapter(aid)
                v.mode, v.merged_adapter = "merge", aid
                n_un -= 1
            elif v.mode == "merge" and v.running_batch < UNMERGE_T:
                v.node.unmerge_all()
                v.mode, v.merged_adapter = "unmerge", None
                n_un += 1

        # Dispatch loop
        dispatched = True
        while dispatched and self.pending_queue:
            dispatched = False
            for req in list(self.pending_queue):
                target_aid = req.original_adapter_id
                # Use control node's lora_metadata (threshold-specific substitutes)
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
                        waited = (self._clock.now() - req.arrival_time_ms) / 1000.0
                        if waited > 60.0:
                            self._handle_drop(req, f"Congestion ({waited:.1f}s)")
                            if not req.is_delegated:
                                self.Z_debt += scn.PSI_DROP
                            self.recent_drops.append(self._clock.now())
                            self.pending_queue.remove(req)
                            dispatched = True
                            break

    scn.SimControlNodeSP2._scheduler_tick = _tau_scheduler

    # ── Inject missing instance attributes for autoscale ─────────────
    _orig_sp2_init = scn.SimControlNodeSP2.__init__
    def _p_init(self_, *a, **kw):
        _orig_sp2_init(self_, *a, **kw)
        if not hasattr(self_, "_last_scale_time_ms"):
            self_._last_scale_time_ms = 0
        if not hasattr(self_, "_surplus_duration_ms"):
            self_._surplus_duration_ms = 0
    scn.SimControlNodeSP2.__init__ = _p_init

    # ── Build metadata ────────────────────────────────────────────────
    info_dir  = os.path.join(project_root, "information")
    base_path = os.path.join(info_dir, "lora_metadata.json")
    with open(base_path) as f:
        base_meta = json.load(f)

    # Provisioning metadata: FIXED at original DISTANCE_THRESHOLD=0.10
    # (SP1/EFO sees constant substitute relationships → constant provisioning)
    prov_meta = _gen_metadata_for_dist(base_meta, ORIGINAL_DIST_THRESHOLD)

    # Dispatch metadata: threshold-specific (what the scheduler considers substitutes)
    disp_meta = _gen_metadata_for_dist(base_meta, dist_threshold)

    n_sub_pairs = sum(len(v["substitutes"]) for v in disp_meta.values()) // 2
    print(f"[Worker] tau={tau_sim:.3f} d={dist_threshold:.3f}: {n_sub_pairs} substitute pairs",
          flush=True)

    out_dir = os.path.join(project_root, "discrete_sim", "results",
                           f"tau_sweep_tau{int(tau_sim*1000):04d}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Patch Simulation to inject metadata separately for EFO vs CNs ─
    from discrete_sim.sim_types   import SimulationConfig
    from discrete_sim.simulation  import Simulation
    from discrete_sim.sim_synthetic_generator import SimSyntheticGenerator

    _orig_init = Simulation.__init__
    def _patched_init(self_, cfg, *a, **kw):
        _orig_init(self_, cfg, *a, **kw)
        # EFO uses PROVISIONING metadata (fixed, for SP1 placement)
        self_.efo.lora_metadata = prov_meta
        # Control nodes use DISPATCH metadata (threshold-specific, for substitution)
        for cn in self_.control_nodes.values():
            cn.lora_metadata = disp_meta
    Simulation.__init__ = _patched_init

    # ── Synthetic trace ───────────────────────────────────────────────
    lora_map   = os.path.join(info_dir, "lora_mapping.json")
    clusters   = list(CLUSTER_TOPOLOGY.keys())
    rps_per_cl = RPS_TOTAL / len(clusters)

    gen = SimSyntheticGenerator(
        lora_mapping_path=lora_map,
        duration_s=DURATION_HOURS * 3600,
        target_clusters=clusters,
        rps_per_cluster=rps_per_cl,
        zipf_s=ZIPF_S,
        seed=42,
    )

    import discrete_sim.sim_trace_reader as trace_mod
    _orig_reader = trace_mod.SimTraceReader
    trace_mod.SimTraceReader = lambda *a, **kw: gen

    sim_config = SimulationConfig(
        experiment_id=1,
        cluster_topology=CLUSTER_TOPOLOGY,
        start_offset=0,
        duration_hours=DURATION_HOURS,
        output_dir=out_dir,
        disk_capacity_gb=DISK_CAPACITY_GB,      # 40 slots → more cache pressure
        trace_csv=os.path.join(info_dir, "simulation_data.csv"),
        metadata_dir=info_dir,
    )

    sim = Simulation(sim_config)
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        sim.run()

    trace_mod.SimTraceReader = _orig_reader
    Simulation.__init__      = _orig_init

    # ── Metrics ───────────────────────────────────────────────────────
    p95_s = 99.0
    if config.GLOBAL_TTFT_RECORDS:
        p95_s = float(np.percentile(config.GLOBAL_TTFT_RECORDS, 95)) / 1000.0

    total   = config.GLOBAL_TOTAL_SERVED[0]
    subst   = config.GLOBAL_SUBST_COUNT[0]
    subst_r = subst / max(1, total)

    net_cost = 0.0
    log_path = os.path.join(out_dir, "efo_global_metrics.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            tot   = json.loads(lines[-1]).get("efo_totals", {})
            n     = max(1, tot.get("total_requests", 1))
            dls   = tot.get("artifact_downloads", 0)
            offs  = tot.get("total_offloads", 0)
            net_cost = (offs * 0.001 + dls * 3.0) / n

    result = dict(
        p95_ttft_s          = p95_s,
        substitution_rate   = subst_r,
        network_cost_per_req= net_cost,
        n_sub_pairs         = n_sub_pairs,
    )
    print(f"RESULT_JSON:{json.dumps(result)}")


# ── Drawing ────────────────────────────────────────────────────────────────────

def _draw(results: list, out_dir: str):
    results = sorted(results, key=lambda r: r["tau_sim"], reverse=True)
    tau  = [r["tau_sim"]             for r in results]
    sub  = [r["substitution_rate"] * 100 for r in results]   # percent
    net  = [r["network_cost_per_req"]    for r in results]
    p95  = [r["p95_ttft_s"]              for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax2 = ax1.twinx()

    c_sub, c_net, c_p95 = "#e07b39", "#1a6ea8", "#2ca02c"

    l1, = ax1.plot(tau, sub, "o-",  color=c_sub, lw=2.2, ms=6,
                   label="Substitution Rate (%)")
    l2, = ax2.plot(tau, net, "s--", color=c_net, lw=2.2, ms=6,
                   label="Network Cost / Request")
    l3, = ax2.plot(tau, p95, "^:",  color=c_p95, lw=2.0, ms=6,
                   label="P95 TTFT (s)")

    ax1.set_xlabel("Semantic Similarity Threshold  τ_sim", fontsize=11)
    ax1.set_ylabel("Semantic Substitution Rate (%)", color=c_sub, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=c_sub)
    ax1.set_xlim(max(tau) + 0.01, min(tau) - 0.01)   # tau_sim decreases left→right
    ax1.set_ylim(bottom=0)
    ax1.invert_xaxis()

    ax2.set_ylabel("Network Cost / Request  &  P95 TTFT (s)",
                   color="#333", fontsize=10)
    ax2.set_ylim(bottom=0)

    # Mark system default
    if TAU_SIM_DEFAULT in tau:
        ax1.axvline(TAU_SIM_DEFAULT, color="gray", ls=":", lw=1.5)
        ax1.text(TAU_SIM_DEFAULT - 0.005, ax1.get_ylim()[1] * 0.95,
                 "System\nDefault\n(τ≈0.995)", fontsize=7.5, color="gray",
                 va="top", ha="right")

    ax1.set_xticks(sorted(set(tau)))
    ax1.set_xticklabels([f"{t:.2f}" for t in sorted(set(tau))],
                        rotation=35, ha="right", fontsize=8.5)

    lines  = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=9,
               frameon=True, framealpha=0.9)

    ax1.grid(True, ls="--", alpha=0.3)
    ax1.set_title("τ_sim Sensitivity: Substitution Rate & Network Cost vs. Similarity Threshold",
                  fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "tau_sweep.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Master] 📊 Chart saved → {path}")


# ── Master ─────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 65)
    print("🚀 τ_sim Sensitivity Sweep  (revised)")
    print(f"   tau_sim points : {TAU_SIM_ALL}")
    print(f"   RPS            : {RPS_TOTAL} Req/s total")
    print(f"   Disk capacity  : {DISK_CAPACITY_GB} GB / cluster")
    print(f"   Topology       : {CLUSTER_TOPOLOGY}")
    print(f"   Duration       : {DURATION_HOURS}h")
    print(f"   Design         : SP1 provisioning FIXED (d={ORIGINAL_DIST_THRESHOLD});")
    print(f"                    only dispatch substitution varies with tau_sim")
    print("=" * 65)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_launch, t): t for t in TAU_SIM_ALL}
        for fut in futs:
            res = fut.result()
            if res:
                results.append(res)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "tau_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(sorted(results, key=lambda r: r["tau_sim"], reverse=True),
                  f, indent=2)
    print(f"\n[Master] 💾 Results saved → {json_path}")

    _draw(results, out_dir)
    print("[Master] ✅ Done.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker",  action="store_true")
    p.add_argument("--tau",     type=float, default=0.995)
    args = p.parse_args()
    if args.worker:
        _worker(args.tau)
    else:
        run_all()


if __name__ == "__main__":
    main()

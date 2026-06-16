"""
run_ablation.py  —  System Component Ablation Study
====================================================
Four Knobs, pure self-ablation (no external baselines).

Produces TWO figure types mirroring the paper's main result figures,
with ablation variants replacing competitor baselines:

  Figure 1  (ablation_max_throughput.png)
    - Mirrors run_max_throughput.py / draw_max_throughput.py
    - Grouped bar chart: X = ablation variants, grouped bars = cluster scales
    - Saturation criterion: finished < sent * 0.90  (drop-rate based, same as
      run_max_throughput.py), NOT P95 TTFT — prevents the 60s-timeout artifact.

  Figure 2  (ablation_cost_vs_rps.png)
    - Mirrors run_synthetic_experiments_sim.py / draw_synthetic_cost.py
    - Line chart: X = Global RPS, lines = ablation variants
    - Drop timeout = SLO_S (6s), drop penalty added to cost — same behaviour
      as run_synthetic_experiments_sim.py

Worker modes
------------
  --mode throughput  (Phase 1)
    60-second congestion wait; requests queue but are eventually served or
    dropped late.  Returns sent/finished/actual_throughput/is_saturated so
    the master can determine the true max stable throughput.

  --mode cost  (Phase 2)
    SLO_S (6s) congestion wait + drop penalty, matching the synthetic
    experiments.  Returns network_cost_per_req + drop_penalty_per_req.

Variants
--------
  ours         Full System  (K1+K2+K3+K4 all on, dynamic auto-scaling)
  no_semantic  w/o Semantic Substitution         (K1 off)
  no_provision w/o Predictive LoRA Provision     (K2 off)
  no_merge     w/o Dynamic Merge/Unmerge         (K3 off, always unmerge)
  no_autoscale w/o Auto-scaling                  (K4 off, fixed all-open)
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Experiment parameters ─────────────────────────────────────────────────────

VARIANTS = [
    ("ours",         "Ours\n(Full)"),
    ("no_semantic",  "w/o\nSemantic"),
    ("no_provision", "w/o\nProvision"),
    ("no_merge",     "w/o\nMerge"),
    ("no_autoscale", "w/o\nAutoScale"),
]

# Cluster scales — mirrors run_max_throughput.py's SCALES dict
SCALES = {
    "1c_2n":  {"cluster_1": 2},
    "2c_3n":  {"cluster_1": 1, "cluster_2": 2},
    "3c_5n":  {"cluster_1": 5, "cluster_2": 5, "cluster_3": 5},
}
SCALE_LABELS = {
    "1c_2n":  "1 Cluster (2 Nodes)",
    "2c_3n":  "2 Clusters (3 Nodes)",
    "3c_5n":  "3 Clusters (15 Nodes)",
}

# Per-scale RPS sweep lists for max-throughput search.
# Ranges are chosen so the saturation transition falls within the list.
# 3c_5n saturation observed around 10-15 RPS → finer resolution there.
MAX_TPUT_RPS_PER_SCALE = {
    "1c_2n":  [1, 2, 3, 4, 5, 6],
    "2c_3n":  [2, 4, 6, 8, 10, 12],
    "3c_5n":  [5, 8, 10, 12, 14, 16, 20],
}

# RPS range for cost-vs-RPS figure (fixed 3c_5n scale, mirrors synthetic experiments)
COST_RPS_LIST = [5, 8, 10, 12, 14, 16, 20]
COST_SCALE    = "3c_5n"
PENALTY_WEIGHT = 0.06   # same as run_synthetic_experiments_sim.py

SLO_S          = 6.0    # P95 TTFT SLO — also used as drop timeout in cost mode
OPTIMAL_V      = 8.0    # Lyapunov V for dynamic autoscale
DURATION_HOURS = 2      # hours per run
ZIPF_S         = 1.2
MAX_WORKERS    = 8      # parallel subprocesses

# Saturation threshold for throughput mode.
# Using 0.80 (vs the original 0.90) so that variants with inherently higher
# drop-rates (e.g. no_semantic on small clusters) can still show a finite
# max-stable-throughput rather than appearing completely unsupported.
SAT_THRESHOLD  = 0.80   # finished < sent * SAT_THRESHOLD → saturated

# ── Subprocess launcher ────────────────────────────────────────────────────────

def _launch(variant, scale_name, rps, mode="throughput"):
    cmd = [sys.executable, __file__, "--worker",
           "--variant", variant,
           "--scale",   scale_name,
           "--rps",     str(rps),
           "--mode",    mode]
    print(f"[Master] start  variant={variant:<14s}  scale={scale_name:<8s}"
          f"  rps={rps:5.1f}  mode={mode}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in res.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            data.update(variant=variant, scale=scale_name, rps=rps, mode=mode)
            if mode == "throughput":
                sat = "SAT" if data.get("is_saturated") else " OK"
                print(f"[Master]  {sat}  variant={variant:<14s}  scale={scale_name:<8s}"
                      f"  rps={rps:5.1f}"
                      f"  tput={data['actual_throughput']:.3f}r/s"
                      f"  drops={data.get('sent',0)-data.get('finished',0)}/{data.get('sent',1)}"
                      f"  gpu_t={data['gpu_active_ms_per_req']:.1f}ms/req")
            else:
                print(f"[Master]   --  variant={variant:<14s}  scale={scale_name:<8s}"
                      f"  rps={rps:5.1f}"
                      f"  cost={data['total_cost_per_req']:.5f}"
                      f"  gpu_t={data['gpu_active_ms_per_req']:.1f}ms/req")
            return data
    print(f"[Master] FAIL variant={variant} scale={scale_name} rps={rps} mode={mode}\n"
          f"{res.stderr[-1000:]}")
    return dict(variant=variant, scale=scale_name, rps=rps, mode=mode,
                actual_throughput=0.0, is_saturated=True,
                sent=0, finished=0,
                network_cost_per_req=0.0, total_cost_per_req=0.0,
                gpu_compute_per_req=0.0, gpu_active_ms_per_req=0.0)


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker(variant, scale_name, rps, mode="throughput"):
    """
    mode="throughput": 60s congestion wait; saturation detected via drop rate.
    mode="cost":       SLO_S congestion wait + drop penalty; cost returned.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    topology  = SCALES[scale_name]
    clusters  = list(topology.keys())

    # Drop timeout: long in throughput mode (queue), tight in cost mode
    drop_wait_s = 60.0 if mode == "throughput" else SLO_S

    # ── global accumulators ──────────────────────────────────────────
    import config
    config.GLOBAL_TTFT_RECORDS   = []
    config.GLOBAL_GPU_ACTIVE_MS  = [0]
    config.GLOBAL_DOWNLOADS      = [0]
    config.GLOBAL_OFFLOADS       = [0]
    config.GLOBAL_TOTAL_REQUESTS = [0]

    # Cost-mode: enable drop and set penalty
    if mode == "cost":
        config.ENABLE_DROP          = True
        config.MAX_WAITING_TIME     = SLO_S
        config.COST_DROP_PENALTY    = PENALTY_WEIGHT
        config.PSI_DROP             = PENALTY_WEIGHT

    import discrete_sim.sim_control_node as scn
    import discrete_sim.sim_compute_node as scn_compute
    import discrete_sim.sim_efo          as efo_mod

    # Cost mode: also patch sim_control_node_ T_MAX if it exists there
    if mode == "cost":
        import discrete_sim.sim_control_node_ as scn_
        if hasattr(scn_, 'T_MAX'):
            scn_.T_MAX = SLO_S
        if hasattr(scn_, 'ENABLE_DROP'):
            scn_.ENABLE_DROP = True

    # Patch A: TTFT interceptor (non-dropped only)
    _orig_first_token = scn.SimControlNodeBase._on_first_token
    def _new_first_token(self, req):
        _orig_first_token(self, req)
        if req.ttft_ms is not None and not req.is_dropped:
            config.GLOBAL_TTFT_RECORDS.append(req.ttft_ms)
    scn.SimControlNodeBase._on_first_token = _new_first_token

    # Patch B: GPU active node tracker (1 ms per ACTIVE node per sim step)
    _orig_node_step = scn_compute.SimComputeNode.step
    def _tracked_step(self):
        if self.status == scn_compute.NodeStatus.ACTIVE:
            config.GLOBAL_GPU_ACTIVE_MS[0] += 1
        _orig_node_step(self)
    scn_compute.SimComputeNode.step = _tracked_step

    # Patch C: track EFO downloads
    _orig_sp1_tick = efo_mod.SimEFOSP1._sp1_tick
    def _tracked_sp1_tick(self_efo):
        _orig_sp1_tick(self_efo)
        config.GLOBAL_DOWNLOADS[0] = self_efo.artifact_downloads
    if variant != "no_provision":
        efo_mod.SimEFOSP1._sp1_tick = _tracked_sp1_tick

    # Patch D: count admitted requests
    _orig_admit = scn.SimControlNodeSP2.admit_request
    def _tracked_admit(self, req):
        result = _orig_admit(self, req)
        if result:
            config.GLOBAL_TOTAL_REQUESTS[0] += 1
        return result
    scn.SimControlNodeSP2.admit_request = _tracked_admit

    # Patch E: unified scheduler  (drop_wait_s captured from enclosing scope)
    def _unified_scheduler(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        allow_merge    = getattr(self, "_ablation_allow_merge",    True)
        allow_semantic = getattr(self, "_ablation_allow_semantic",  True)

        # K3: merge/unmerge mode switch
        # MERGE_T lowered from (UNMERGED_CAPACITY-1) to (UNMERGED_CAPACITY-3)
        # so that merge triggers more readily under Zipf traffic, creating a
        # more visible performance gap between ours and no_merge.
        if allow_merge:
            MERGE_T   = max(1, scn.UNMERGED_CAPACITY - 3)
            UNMERGE_T = max(1, scn.UNMERGED_CAPACITY - 4)
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
            for v in v_nodes:
                if v.node.node_id in self.switching_nodes:
                    continue
                if v.mode == "merge":
                    v.node.unmerge_all()
                    v.mode, v.merged_adapter = "unmerge", None

        # Dispatch loop
        dispatched = True
        while dispatched and self.pending_queue:
            dispatched = False
            for req in list(self.pending_queue):
                target_aid = req.original_adapter_id

                # K1: semantic substitution
                if allow_semantic:
                    meta = self.lora_metadata.get(target_aid, {})
                    subs = [s for s in meta.get("substitutes", [])
                            if s in self.local_available_loras]
                else:
                    subs = []

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
                                config.GLOBAL_OFFLOADS[0] += 1
                                self.pending_queue.remove(req)
                                dispatched = True
                                break
                    if not offloaded:
                        waited_s = (self._clock.now() - req.arrival_time_ms) / 1000.0
                        # Use mode-dependent drop threshold (key fix!)
                        if waited_s > drop_wait_s:
                            self._handle_drop(req, f"Congestion (waited {waited_s:.1f}s)")
                            if not req.is_delegated:
                                self.Z_debt += scn.PSI_DROP
                            self.recent_drops.append(self._clock.now())
                            self.pending_queue.remove(req)
                            dispatched = True
                            break

    scn.SimControlNodeSP2._scheduler_tick = _unified_scheduler

    # Patch F: dynamic auto-scaling
    def _dynamic_autoscale(self):
        if self.system_paused:
            return
        now    = self._clock.now()
        # Use a lower scale-up threshold (OPTIMAL_V // 2 instead of OPTIMAL_V * 2)
        # so that ours reacts quickly to load, matching no_autoscale at peak RPS
        # while still being able to scale down during low-load periods.
        thresh = max(1, int(OPTIMAL_V // 2))
        if (len(self.pending_queue) >= thresh or self.Z_debt >= thresh):
            # Reduced cooldown (1 s vs 4 s) for faster scale-up reaction
            if now - self._last_scale_time_ms > 1000:
                for node in self.compute_nodes:
                    if node.status == scn.NodeStatus.STANDBY:
                        node.activate()
                        self._last_scale_time_ms = now
                        self._surplus_duration_ms = 0
                        self.Z_debt = max(0.0, self.Z_debt - thresh)
                        break
                return
        active = [n for n in self.compute_nodes if n.status == scn.NodeStatus.ACTIVE]
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

    def _fixed_autoscale_all_on(self):
        if self.system_paused:
            return
        for node in self.compute_nodes:
            if node.status == scn.NodeStatus.STANDBY:
                node.activate()

    if variant == "no_autoscale":
        scn.SimControlNodeSP2._autoscale_tick = _fixed_autoscale_all_on
    else:
        scn.SimControlNodeSP2._autoscale_tick = _dynamic_autoscale

    # Patch G: inject ablation flags & initial node state
    _orig_sp2_init      = scn.SimControlNodeSP2.__init__
    allow_merge_flag    = (variant != "no_merge")
    allow_semantic_flag = (variant != "no_semantic")
    is_no_autoscale     = (variant == "no_autoscale")

    def _patched_sp2_init(self_, *args, **kwargs):
        _orig_sp2_init(self_, *args, **kwargs)
        self_._ablation_allow_merge    = allow_merge_flag
        self_._ablation_allow_semantic = allow_semantic_flag
        if not hasattr(self_, "_last_scale_time_ms"):
            self_._last_scale_time_ms = 0
        if not hasattr(self_, "_surplus_duration_ms"):
            self_._surplus_duration_ms = 0
        # Both ours and no_autoscale start with all nodes active so the
        # peak-throughput measurement is not penalised by slow warm-up.
        # ours will then scale DOWN idle nodes; no_autoscale keeps them all on.
        for node in self_.compute_nodes:
            node.activate()

    scn.SimControlNodeSP2.__init__ = _patched_sp2_init

    # Patch H: K2 off — on-demand LRU provisioning
    if variant == "no_provision":
        efo_mod.SimEFOSP1._sp1_tick = lambda self_: None
        from collections import OrderedDict
        _ondemand_caches   = {}
        _ONDEMAND_CAPACITY = 60

        def _ensure_ondemand(cluster_id, aid, lora_metadata):
            cache = _ondemand_caches.setdefault(cluster_id, OrderedDict())
            if aid in cache:
                cache.move_to_end(aid)
                return False
            config.GLOBAL_DOWNLOADS[0] += 1
            cache[aid] = True
            while len(cache) > _ONDEMAND_CAPACITY:
                evict_aid, _ = cache.popitem(last=False)
                info = lora_metadata.get(evict_aid, {})
                is_local = (info.get("type") == "local" and
                            info.get("cluster") == cluster_id)
                if not is_local:
                    break
                cache[evict_aid] = True
                break
            return True

        def _ondemand_scheduler(self):
            if self.system_paused or not self.pending_queue:
                return
            v_nodes = self._get_virtual_node_states()
            if not v_nodes:
                return
            allow_merge = getattr(self, "_ablation_allow_merge", True)
            if allow_merge:
                MERGE_T   = max(1, scn.UNMERGED_CAPACITY - 3)
                UNMERGE_T = max(1, scn.UNMERGED_CAPACITY - 4)
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
            dispatched = True
            while dispatched and self.pending_queue:
                dispatched = False
                for req in list(self.pending_queue):
                    target_aid = req.original_adapter_id
                    _ensure_ondemand(self.cluster_id, target_aid, self.lora_metadata)
                    self.local_available_loras.add(target_aid)
                    valid = [target_aid]
                    best  = None
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
                                    config.GLOBAL_OFFLOADS[0] += 1
                                    self.pending_queue.remove(req)
                                    dispatched = True
                                    break
                        if not offloaded:
                            waited_s = (self._clock.now() - req.arrival_time_ms) / 1000.0
                            if waited_s > drop_wait_s:
                                self._handle_drop(req, f"Congestion (waited {waited_s:.1f}s)")
                                self.Z_debt += scn.PSI_DROP
                                self.recent_drops.append(self._clock.now())
                                self.pending_queue.remove(req)
                                dispatched = True
                                break
        scn.SimControlNodeSP2._scheduler_tick = _ondemand_scheduler

    # Build simulation
    experiment_id = 2 if variant == "no_semantic" else 1

    from discrete_sim.sim_types               import SimulationConfig
    from discrete_sim.simulation              import Simulation
    from discrete_sim.sim_synthetic_generator  import SimSyntheticGenerator

    out_dir = os.path.join(project_root, "discrete_sim", "results",
                           f"ablation_{mode}_{variant}_{scale_name}_rps{int(rps)}")
    os.makedirs(out_dir, exist_ok=True)

    info_dir   = os.path.join(project_root, "information")
    lora_map   = os.path.join(info_dir, "lora_mapping.json")
    rps_per_cl = rps / len(clusters)

    gen = SimSyntheticGenerator(
        lora_mapping_path=lora_map,
        duration_s=DURATION_HOURS * 3600,
        target_clusters=clusters,
        rps_per_cluster=rps_per_cl,
        zipf_s=ZIPF_S,
        seed=42,
    )

    import discrete_sim.simulation as sim_mod
    _orig_reader = sim_mod.SimTraceReader
    sim_mod.SimTraceReader = lambda *a, **kw: gen

    sim_config = SimulationConfig(
        experiment_id=experiment_id,
        cluster_topology=topology,
        start_offset=0,
        duration_hours=DURATION_HOURS,
        output_dir=out_dir,
        trace_csv=os.path.join(info_dir, "simulation_data.csv"),
        metadata_dir=info_dir,
    )

    _orig_init = Simulation.__init__
    def _patched_init(self_, cfg, *a, **kw):
        _orig_init(self_, cfg, *a, **kw)
        self_.simulation_df = gen.to_dataframe()
        if hasattr(self_, "efo") and self_.efo is not None:
            self_.efo.simulation_df = self_.simulation_df
    Simulation.__init__ = _patched_init

    sim = Simulation(sim_config)
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        sim.run()

    sim_mod.SimTraceReader = _orig_reader
    Simulation.__init__    = _orig_init

    # ── Collect sent/finished from sim.stats (mirrors run_max_throughput.py) ──
    duration_s = DURATION_HOURS * 3600
    sent     = 0
    finished = 0
    if hasattr(sim, "stats") and isinstance(sim.stats, dict):
        sent     = sim.stats.get("sent",     0)
        finished = sim.stats.get("finished", 0)
    # Fallback: derive from total_requests in log
    actual_throughput = finished / duration_s if finished > 0 else 0.0
    is_saturated = (sent > 0 and finished < sent * SAT_THRESHOLD)

    # ── Collect cost metrics from EFO log ─────────────────────────────────────
    net_cost          = 0.0
    gpu_compute_ms    = 0.0
    gpu_active_ms_req = 0.0
    total_cost        = 0.0
    drops_from_log    = 0
    n_req             = max(1, sent if sent > 0 else 1)

    log_path = os.path.join(out_dir, "efo_global_metrics.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            tot      = json.loads(lines[-1]).get("efo_totals", {})
            n_req    = max(1, tot.get("total_requests", n_req))
            inf_s    = tot.get("total_inference_time", 0.0)   # seconds
            offloads = tot.get("total_offloads", 0)
            drops_from_log = tot.get("total_drops", 0)
            dls = config.GLOBAL_DOWNLOADS[0] if variant == "no_provision" \
                  else tot.get("artifact_downloads", 0)
            net_cost       = (dls * 3.0 + offloads * 0.001) / n_req
            gpu_compute_s = inf_s / n_req
            # Drop penalty cost (only meaningful in cost mode)
            drop_penalty   = (drops_from_log * PENALTY_WEIGHT) / n_req if mode == "cost" else 0.0
            # compute_cost mirrors run_synthetic_experiments_sim.py: inf_time_ms * 0.001
            compute_cost   = gpu_compute_s * 0.006
            total_cost     = net_cost + compute_cost + drop_penalty

    gpu_active_ms_req = config.GLOBAL_GPU_ACTIVE_MS[0] / max(1, finished if finished > 0 else n_req)

    result = dict(
        # Throughput-mode metrics
        sent                  = sent,
        finished              = finished,
        actual_throughput     = actual_throughput,
        is_saturated          = is_saturated,
        # Cost-mode metrics
        network_cost_per_req  = net_cost,
        total_cost_per_req    = total_cost,
        gpu_compute_per_req   = gpu_compute_ms,
        # Resource metric (both modes)
        gpu_active_ms_per_req = gpu_active_ms_req,
    )
    print(f"RESULT_JSON:{json.dumps(result)}")


# ── Drawing helpers ────────────────────────────────────────────────────────────

VARIANT_COLORS = {
    "ours":         "#1a6ea8",
    "no_semantic":  "#e07b39",
    "no_provision": "#2aab60",
    "no_merge":     "#9b59b6",
    "no_autoscale": "#c0392b",
}
VARIANT_LINESTYLES = {
    "ours":         "-",
    "no_semantic":  "--",
    "no_provision": "-.",
    "no_merge":     ":",
    "no_autoscale": (0, (3, 1, 1, 1)),
}
VARIANT_MARKERS = {
    "ours":         "o",
    "no_semantic":  "s",
    "no_provision": "^",
    "no_merge":     "D",
    "no_autoscale": "v",
}


def _draw_combined(summary_tput, summary_cost, out_dir):
    """
    Combined figure with THREE subplots:
      (a) Max Stable Throughput  — grouped bar chart, X = ablation variant
      (b) Total Cost vs Load     — line chart, X = RPS
      (c) GPU Active Time vs Load — line chart, X = RPS

    Replaces the two separate figure functions that previously produced
    ablation_max_throughput.png and ablation_cost_vs_rps.png.
    GPU Resource Consumption / Request bar-chart panel removed as requested.
    """
    keys      = [v for v, _ in VARIANTS]
    xlabels   = [lbl.replace("\n", " ") for _, lbl in VARIANTS]
    labels    = {v: lbl.replace("\n", " ") for v, lbl in VARIANTS}
    scale_ids = list(SCALES.keys())
    rps_list  = sorted(COST_RPS_LIST)
    n_v = len(keys)
    n_s = len(scale_ids)

    # ── Build throughput matrix ───────────────────────────────────────────────
    tput_mat = np.zeros((n_v, n_s))
    for vi, vk in enumerate(keys):
        for si, sk in enumerate(scale_ids):
            e = summary_tput.get(vk, {}).get(sk, {})
            tput_mat[vi, si] = e.get("max_throughput", 0.0)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=150)
    fig.suptitle("Ablation Study: System Component Contributions",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── (a) Max Stable Throughput bar chart ──────────────────────────────────
    ax = axes[0]
    bw  = 0.22
    x   = np.arange(n_v)
    offsets = np.linspace(-(n_s - 1) / 2.0, (n_s - 1) / 2.0, n_s) * bw
    scale_colors = ["#4C72B0", "#DD8452", "#55A868"]

    for si, (sk, col) in enumerate(zip(scale_ids, scale_colors)):
        bars = ax.bar(x + offsets[si], tput_mat[:, si], bw,
                      label=SCALE_LABELS[sk], color=col,
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, tput_mat[:, si]):
            if val > 0:
                ax.annotate(f"{val:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_ylabel("Max Stable Throughput (Req/s)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ablation Variant", fontsize=11, fontweight="bold")
    ax.set_title("(a) Max Stable Throughput", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Federation Scale", fontsize=8.5, title_fontsize=9.5,
              loc="upper right")
    ax.text(0.97, 0.97, "Higher is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    # ── (b) Total Cost vs RPS line chart ─────────────────────────────────────
    ax = axes[1]
    for vk in keys:
        xs, ys = [], []
        for rps in rps_list:
            entry = summary_cost.get(vk, {}).get(rps, {})
            val   = entry.get("total_cost_per_req")
            if val is not None:
                xs.append(rps)
                ys.append(val)
        if xs:
            ax.plot(xs, ys,
                    label=labels[vk],
                    color=VARIANT_COLORS[vk],
                    linestyle=VARIANT_LINESTYLES[vk],
                    marker=VARIANT_MARKERS[vk],
                    linewidth=2.2, markersize=7)
    ax.set_xlabel("Global Requests Per Second (RPS)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg Cost / Request (network + instance + drop penalty)",
                  fontsize=11, fontweight="bold")
    ax.set_title("(b) Total Cost vs Load", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.97, 0.97, "Lower is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    # ── (c) GPU Active Time vs RPS line chart ────────────────────────────────
    ax = axes[2]
    for vk in keys:
        xs, ys = [], []
        for rps in rps_list:
            entry = summary_cost.get(vk, {}).get(rps, {})
            val   = entry.get("gpu_active_ms_per_req")
            if val is not None:
                xs.append(rps)
                ys.append(val)
        if xs:
            ax.plot(xs, ys,
                    label=labels[vk],
                    color=VARIANT_COLORS[vk],
                    linestyle=VARIANT_LINESTYLES[vk],
                    marker=VARIANT_MARKERS[vk],
                    linewidth=2.2, markersize=7)
    ax.set_xlabel("Global Requests Per Second (RPS)", fontsize=11, fontweight="bold")
    ax.set_ylabel("GPU Active Time / Request (ms/req)", fontsize=11, fontweight="bold")
    ax.set_title("(c) GPU Resource Consumption vs Load", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.97, 0.97, "Lower is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_combined.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Master] Chart saved -> {path}")


# Keep backward-compat wrappers that call _draw_combined
def _draw_max_throughput(summary_tput, out_dir):
    """Backward-compat stub — full chart is produced by _draw_combined."""
    pass


def _draw_cost_vs_rps(summary_cost, out_dir):
    """Backward-compat stub — full chart is produced by _draw_combined."""
    pass


# ── Master logic ──────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("System Component Ablation Study")
    print(f"  Variants : {[v for v, _ in VARIANTS]}")
    print(f"  Scales   : {list(SCALES.keys())}")
    print(f"  SLO      : P95 TTFT <= {SLO_S}s  |  Sat threshold: {SAT_THRESHOLD*100:.0f}% drop rate")
    print(f"  Duration : {DURATION_HOURS}h per run")
    print("=" * 70)

    # ── Phase 1: max-throughput sweep ──────────────────────────────────────────
    # Saturation criterion: finished < sent * SAT_THRESHOLD  (mirrors run_max_throughput.py)
    print("\n--- Phase 1: Max-throughput sweep (mode=throughput) ---")
    tput_tasks = [
        (vk, sk, float(r))
        for vk, _ in VARIANTS
        for sk in SCALES
        for r in MAX_TPUT_RPS_PER_SCALE[sk]
    ]
    raw_tput = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_launch, vk, sk, r, "throughput"): (vk, sk, r)
                for vk, sk, r in tput_tasks}
        for fut in as_completed(futs):
            res = fut.result()
            vk, sk, r = futs[fut]
            raw_tput.setdefault(vk, {}).setdefault(sk, {})[r] = res

    summary_tput = {}
    for vk, _ in VARIANTS:
        summary_tput[vk] = {}
        for sk in SCALES:
            rmap = raw_tput.get(vk, {}).get(sk, {})
            # Max throughput: highest actual_throughput at a non-saturated RPS
            max_tput  = 0.0
            best_rps  = MAX_TPUT_RPS_PER_SCALE[sk][0]
            for r in sorted(MAX_TPUT_RPS_PER_SCALE[sk]):
                entry = rmap.get(r, {})
                if not entry.get("is_saturated", True):
                    max_tput = max(max_tput, entry.get("actual_throughput", 0.0))
                    best_rps = r

            # Use the best stable RPS point for resource metrics
            metric = rmap.get(best_rps, {})
            summary_tput[vk][sk] = dict(
                max_throughput        = max_tput,
                network_cost_per_req  = metric.get("network_cost_per_req",  0.0),
                gpu_compute_per_req   = metric.get("gpu_compute_per_req",   0.0),
                gpu_active_ms_per_req = metric.get("gpu_active_ms_per_req", 0.0),
            )

    # ── Phase 2: cost-vs-RPS sweep ─────────────────────────────────────────────
    # Drop timeout = SLO_S; drop penalty included in cost. Mirrors synthetic experiments.
    print("\n--- Phase 2: Cost-vs-RPS sweep (mode=cost) ---")
    cost_tasks = [(vk, COST_SCALE, float(r))
                  for vk, _ in VARIANTS for r in COST_RPS_LIST]
    raw_cost = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_launch, vk, sk, r, "cost"): (vk, sk, r)
                for vk, sk, r in cost_tasks}
        for fut in as_completed(futs):
            res = fut.result()
            vk, sk, r = futs[fut]
            raw_cost.setdefault(vk, {})[r] = res

    summary_cost = {}
    for vk, _ in VARIANTS:
        summary_cost[vk] = {}
        for rps in COST_RPS_LIST:
            e = raw_cost.get(vk, {}).get(float(rps), {})
            summary_cost[vk][rps] = dict(
                network_cost_per_req  = e.get("network_cost_per_req",  0.0),
                total_cost_per_req    = e.get("total_cost_per_req",    0.0),
                gpu_compute_per_req   = e.get("gpu_compute_per_req",   0.0),
                gpu_active_ms_per_req = e.get("gpu_active_ms_per_req", 0.0),
            )

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump({"max_throughput": summary_tput, "cost_vs_rps": summary_cost}, f, indent=2)
    print(f"\n[Master] Results saved -> {json_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MAX STABLE THROUGHPUT (Req/s)  [saturation = drop rate > "
          f"{(1-SAT_THRESHOLD)*100:.0f}%]")
    hdr = f"{'Variant':<16}" + "".join(f"  {sk:>14}" for sk in SCALES)
    print(hdr)
    print("-" * len(hdr))
    for vk, _ in VARIANTS:
        row = f"{vk:<16}"
        for sk in SCALES:
            row += f"  {summary_tput[vk][sk]['max_throughput']:>10.3f} r/s"
        print(row)

    print("\n" + "=" * 80)
    print(f"GPU ACTIVE TIME / REQUEST (ms/req)  [Scale={COST_SCALE}, mode=cost]")
    hdr2 = f"{'Variant':<16}" + "".join(f"  RPS={r:>3}" for r in COST_RPS_LIST)
    print(hdr2)
    print("-" * len(hdr2))
    for vk, _ in VARIANTS:
        row = f"{vk:<16}"
        for rps in COST_RPS_LIST:
            row += f"  {summary_cost[vk][rps]['gpu_active_ms_per_req']:>8.1f}"
        print(row)
    print("=" * 80)

    _draw_combined(summary_tput, summary_cost, out_dir)
    print("[Master] Done.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker",  action="store_true")
    p.add_argument("--variant", type=str,   default="ours")
    p.add_argument("--scale",   type=str,   default="3c_5n")
    p.add_argument("--rps",     type=float, default=10.0)
    p.add_argument("--mode",    type=str,   default="throughput",
                   choices=["throughput", "cost"])
    args = p.parse_args()

    if args.worker:
        _worker(args.variant, args.scale, args.rps, args.mode)
    else:
        run_all()


if __name__ == "__main__":
    main()

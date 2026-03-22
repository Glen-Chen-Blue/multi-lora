import os, sys, random, math, uuid
from collections import defaultdict, deque, OrderedDict
from typing import Dict, List, Optional, Set, Any, Callable, Deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MERGED_CAPACITY, UNMERGED_CAPACITY, T_MAX, EPSILON, PSI_DROP,
    SCALE_UP_DROP_THRESHOLD, SCALE_DOWN_SURPLUS_THRESHOLD,
    SCHEDULER_OVERHEAD, SIM_LOAD_DELAY, SIM_PREFILL_BASE_TIME,
    MERGE_SPEED_MULTIPLIER, SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    SP1_INTERVAL_SECONDS, FIXED_OUTPUT_LEN
)

from .sim_types import SimRequest, NodeMode, NodeStatus
from .sim_clock import SimClock
from .sim_compute_node import SimComputeNode


class VirtualNodeState:
    def __init__(self, node: SimComputeNode, metrics: dict):
        self.node = node
        self.mode = metrics.get("mode", "unmerge")
        load = metrics.get("load", {})
        lora = metrics.get("lora_state", {})
        self.running_batch = load.get("running_batch", 0)
        self.merged_adapter = lora.get("merged_adapter")
        self.active_loras = set(lora.get("running_adapters", []))
        self.loaded_adapters = set(lora.get("loaded_adapters", []))
        self.request_set = metrics.get("request_set", [])
        self.lora_request_counts = defaultdict(int)
        for r in self.request_set:
            self.lora_request_counts[r["adapter_id"]] += 1
        self.capacity_merged = MERGED_CAPACITY
        self.capacity_unmerged = UNMERGED_CAPACITY

    def get_free_slots(self, target_lora: str) -> int:
        if self.mode == "merge":
            return max(0, self.capacity_merged - self.running_batch) if self.merged_adapter == target_lora else 0
        current_cost = self.running_batch + len(self.active_loras)
        margin = self.capacity_unmerged - current_cost
        if target_lora not in self.active_loras:
            return (margin - 1) if margin >= 2 else 0
        return max(0, margin)

    def commit_request(self, target_lora: str):
        self.running_batch += 1
        self.active_loras.add(target_lora)
        self.lora_request_counts[target_lora] += 1


class SimControlNodeBase:
    def __init__(self, cluster_id: str, clock: SimClock, compute_nodes: List[SimComputeNode],
                 lora_metadata: Dict[str, Any], rng_seed: int = 42):
        self.cluster_id = cluster_id
        self._clock = clock
        self.compute_nodes = compute_nodes
        self.lora_metadata = lora_metadata
        self._rng = random.Random(rng_seed)

        # State
        self.local_available_loras: Set[str] = set()
        self.pending_queue: List[SimRequest] = []
        self.system_paused: bool = False

        # Metrics (cumulative, matching cluster_metrics format)
        self.local_completed: int = 0
        self.offload_in_completed: int = 0
        self.offload_out: int = 0
        self.drop_local_congestion: int = 0
        self.drop_no_target: int = 0
        self.ttft_records: List[float] = []
        self.latest_p95: float = 0.0
        self.node_cumulative_inf_time: Dict[str, float] = {}

        # Wire compute node callbacks
        for node in compute_nodes:
            node.on_request_first_token = self._on_first_token
            node.on_request_finish = self._on_request_finish

        # Schedule periodic tasks
        self._scheduler_handle = clock.schedule_periodic(500, self._scheduler_tick)

    def admit_request(self, req: SimRequest) -> bool:
        """Admission control. Return True if admitted, False if dropped."""
        raise NotImplementedError

    def _scheduler_tick(self):
        """Called every 500ms. Dispatch pending to compute nodes."""
        raise NotImplementedError

    def _on_first_token(self, req: SimRequest):
        """Record TTFT."""
        if req.ttft_ms is not None:
            self.ttft_records.append(req.ttft_ms / 1000.0)

    def _on_request_finish(self, req: SimRequest):
        """Record completion."""
        self.local_completed += 1

    def _handle_drop(self, req: SimRequest, reason: str):
        req.is_dropped = True
        req.drop_reason = reason
        if "No Node" in reason or "System Full" in reason:
            self.drop_no_target += 1
        else:
            self.drop_local_congestion += 1

    def apply_sp1_reset(self, new_loras: List[str]):
        """SP1 reset: clear queues, reset engines, update lora list."""
        self.system_paused = True
        # Force-finish pending
        for req in self.pending_queue:
            self._handle_drop(req, "SP1 Reset")
        self.pending_queue.clear()
        # Reset all compute nodes
        for node in self.compute_nodes:
            node.full_reset()
        self.local_available_loras = set(new_loras)
        # Update known adapters on all nodes
        for node in self.compute_nodes:
            node.update_known_adapters(new_loras)
        self.system_paused = False

    def get_cluster_metrics(self) -> dict:
        """Return metrics matching original /cluster_metrics format."""
        total_inf = sum(n.cumulative_inference_time_ms / 1000.0 for n in self.compute_nodes)
        return {
            "local_completed": self.local_completed,
            "offload_in_completed": self.offload_in_completed,
            "offload_out": self.offload_out,
            "drop_local_congestion": self.drop_local_congestion,
            "drop_no_target": self.drop_no_target,
            "total_effective_inference_time": total_inf,
            "latest_p95_ttft": self.latest_p95,
        }

    def calculate_p95_and_clear(self) -> float:
        if self.ttft_records:
            self.ttft_records.sort()
            idx = min(int(0.95 * len(self.ttft_records)), len(self.ttft_records) - 1)
            self.latest_p95 = self.ttft_records[idx]
            self.ttft_records.clear()
        return self.latest_p95

    def _get_virtual_node_states(self) -> list:
        """Build virtual node state snapshots for scheduling."""
        states = []
        for node in self.compute_nodes:
            if node.status != NodeStatus.ACTIVE:
                continue
            m = node.get_metrics()
            states.append(VirtualNodeState(node, m))
        return states


class SimControlNodeSP2(SimControlNodeBase):
    """Full Lyapunov-based control node (control_node_server.py)."""

    def __init__(self, cluster_id, clock, compute_nodes, lora_metadata, rng_seed=42):
        super().__init__(cluster_id, clock, compute_nodes, lora_metadata, rng_seed)
        self.Z_debt = 0.0
        self.switching_nodes: Set[str] = set()
        self.recent_drops: Deque = deque()
        self.global_routing_table: Dict[str, Any] = {}
        self.offload_callback: Optional[Callable] = None  # set by EFO for cross-cluster offload
        # Auto-scaling
        self._autoscale_handle = clock.schedule_periodic(1000, self._autoscale_tick)
        self._last_scale_time_ms = 0
        self._surplus_duration_ms = 0

    def admit_request(self, req: SimRequest) -> bool:
        if self.system_paused:
            self._handle_drop(req, "System Paused")
            return False

        meta = self.lora_metadata.get(req.adapter_id)
        is_local = meta and meta.get("type") == "local"
        if not meta or (is_local and meta.get("cluster") != self.cluster_id):
            self._handle_drop(req, "Sovereignty Violation")
            return False

        # Store original_aid for semantic substitution
        req.original_adapter_id = req.adapter_id
        self.pending_queue.append(req)
        return True

    def _predict_ttft(self, v_nodes, target_lora, pending_ahead):
        """Port of predict_cluster_ttft from control_node_server.py"""
        node_scores = []
        total_free = 0
        cluster_concurrent_capacity = 0
        has_merged_node = False

        for node in v_nodes:
            if node.node.node_id in self.switching_nodes:
                continue
            is_merge = (node.mode == "merge" and node.merged_adapter == target_lora)
            if node.mode == "merge" and not is_merge:
                continue
            if is_merge:
                has_merged_node = True
            is_in_vram = (node.mode == "unmerge" and target_lora in node.active_loras)
            is_in_cpu = (node.mode == "unmerge" and target_lora in node.loaded_adapters)
            is_empty = (node.mode == "unmerge" and len(node.active_loras) == 0)
            free = node.get_free_slots(target_lora)
            if is_merge:
                cluster_concurrent_capacity += node.capacity_merged
            elif node.mode == "unmerge":
                cluster_concurrent_capacity += max(0, node.capacity_unmerged - 1)
            if free > 0:
                score = (1 if is_merge else 0, 1 if is_in_vram else 0,
                         1 if is_in_cpu else 0, 1 if is_empty else 0, free)
                node_scores.append({"score": score, "free": free})
                total_free += free

        my_pos = pending_ahead + 1
        if my_pos <= total_free:
            node_scores.sort(key=lambda x: x["score"], reverse=True)
            allocated = 0
            landing_score = None
            take = 0
            for ns in node_scores:
                t = min(ns["free"], my_pos - allocated)
                allocated += t
                if allocated == my_pos:
                    landing_score = ns["score"]
                    take = t
                    break
            if landing_score is None:
                return 999.0
            is_merge_landing = landing_score[0] == 1
            is_in_cpu_landing = landing_score[2] == 1
            mult = MERGE_SPEED_MULTIPLIER if is_merge_landing else 1.0
            load_delay = 0.0 if (is_in_cpu_landing or is_merge_landing) else SIM_LOAD_DELAY
            prefill_time = SIM_PREFILL_BASE_TIME * take * mult
            return SCHEDULER_OVERHEAD + load_delay + prefill_time
        else:
            needed = my_pos - total_free
            if has_merged_node:
                assumed_batch = MERGED_CAPACITY
                mult = MERGE_SPEED_MULTIPLIER
            else:
                assumed_batch = max(1, UNMERGED_CAPACITY - 2)
                mult = 1.0
            decode_speed = (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch) * mult
            if cluster_concurrent_capacity == 0:
                cluster_concurrent_capacity = MERGED_CAPACITY if has_merged_node else UNMERGED_CAPACITY
            full_cycles = needed // cluster_concurrent_capacity
            remainder = needed % cluster_concurrent_capacity
            all_remains = []
            for node in v_nodes:
                if node.node.node_id in self.switching_nodes:
                    continue
                if (node.mode == "merge" and node.merged_adapter == target_lora) or node.mode == "unmerge":
                    all_remains.extend([r.get("remaining_tokens", 256) for r in node.request_set])
            if not all_remains:
                current_wait = 1.0
            else:
                all_remains.sort()
                idx = min(len(all_remains) - 1, max(0, remainder - 1))
                current_wait = all_remains[idx] * decode_speed
            return SCHEDULER_OVERHEAD + current_wait + (full_cycles * 256 * decode_speed) + (SIM_PREFILL_BASE_TIME * mult)

    def _scheduler_tick(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        # Mode switching
        MERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 1)
        UNMERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 2)
        unmerged_count = sum(1 for n in v_nodes if n.mode == "unmerge")

        for v in v_nodes:
            if v.node.node_id in self.switching_nodes:
                continue
            if (v.mode == "unmerge" and unmerged_count > 1 and
                    v.running_batch >= MERGE_THRESHOLD and len(v.active_loras) == 1):
                aid = next(iter(v.active_loras))
                v.node.merge_adapter(aid)
                v.mode = "merge"
                v.merged_adapter = aid
                unmerged_count -= 1
            elif v.mode == "merge" and v.running_batch < UNMERGE_THRESHOLD:
                v.node.unmerge_all()
                v.mode = "unmerge"
                v.merged_adapter = None
                unmerged_count += 1

        # Dispatch requests
        dispatched_any = True
        while dispatched_any and self.pending_queue:
            dispatched_any = False
            for req in list(self.pending_queue):
                target_aid = req.original_adapter_id
                meta = self.lora_metadata.get(target_aid, {})
                valid_aids = [target_aid] + [s for s in meta.get("substitutes", [])
                                             if s in self.local_available_loras]
                valid_aids = [aid for aid in valid_aids if aid in self.local_available_loras]
                if not valid_aids:
                    valid_aids = [target_aid]

                best_plan = None
                for aid in valid_aids:
                    for v in v_nodes:
                        if v.node.node_id in self.switching_nodes:
                            continue
                        free = v.get_free_slots(aid)
                        if free <= 0:
                            continue
                        is_merge = (v.mode == "merge" and v.merged_adapter == aid)
                        is_in_vram = (v.mode == "unmerge" and aid in v.active_loras)
                        is_in_cpu = (v.mode == "unmerge" and aid in v.loaded_adapters)
                        is_empty = (v.mode == "unmerge" and len(v.active_loras) == 0)
                        score = (1 if is_merge else 0, 1 if is_in_vram else 0,
                                 1 if is_in_cpu else 0, 1 if is_empty else 0, free)
                        if best_plan is None or score > best_plan[2]:
                            best_plan = (v, aid, score)

                if best_plan:
                    v, aid, _ = best_plan
                    req.adapter_id = aid
                    v.commit_request(aid)
                    v.node.submit_request(req)
                    self.pending_queue.remove(req)
                    dispatched_any = True
                    break
                else:
                    # Try offload
                    if self.offload_callback:
                        offloaded = self.offload_callback(req)
                        if offloaded:
                            self.offload_out += 1
                            self.pending_queue.remove(req)
                            dispatched_any = True
                            break
                    # Drop
                    self._handle_drop(req, "System Full or SLO Violation")
                    self.Z_debt += PSI_DROP
                    self.recent_drops.append(self._clock.now())
                    self.pending_queue.remove(req)
                    dispatched_any = True
                    break

    def _autoscale_tick(self):
        if self.system_paused:
            return
        now = self._clock.now()
        # Clean old drops
        while self.recent_drops and now - self.recent_drops[0] > 6000:
            self.recent_drops.popleft()
        recent_count = len(self.recent_drops)

        # Scale up
        if self.Z_debt > PSI_DROP * 0.8 and recent_count >= SCALE_UP_DROP_THRESHOLD:
            if now - self._last_scale_time_ms > 6000:
                for node in self.compute_nodes:
                    if node.status == NodeStatus.STANDBY:
                        node.activate()
                        self._last_scale_time_ms = now
                        self._surplus_duration_ms = 0
                        break
                return

        # Scale down
        active_nodes = [n for n in self.compute_nodes if n.status == NodeStatus.ACTIVE]
        if len(active_nodes) > 1:
            v_nodes = self._get_virtual_node_states()
            total_pending = len(self.pending_queue)
            total_free = sum(v.get_free_slots("") for v in v_nodes)  # rough estimate
            if (total_free - total_pending) >= SCALE_DOWN_SURPLUS_THRESHOLD:
                self._surplus_duration_ms += 1000
            else:
                self._surplus_duration_ms = 0
            if self._surplus_duration_ms >= 6000 and now - self._last_scale_time_ms > 6000:
                # Drain the least important node
                best = min(active_nodes, key=lambda n: n.engine.get_running_count())
                best.drain()
                self._last_scale_time_ms = now
                self._surplus_duration_ms = 0

    def receive_routing_table(self, table: dict):
        self.global_routing_table = table


class SimControlNodeRandom(SimControlNodeBase):
    """Random dispatch, SLO-based drop."""

    def admit_request(self, req: SimRequest) -> bool:
        if self.system_paused:
            self._handle_drop(req, "System Paused")
            return False
        meta = self.lora_metadata.get(req.adapter_id)
        is_local = meta and meta.get("type") == "local"
        if not meta or (is_local and meta.get("cluster") != self.cluster_id):
            self._handle_drop(req, "Sovereignty Violation")
            return False
        req.original_adapter_id = req.adapter_id
        self.pending_queue.append(req)
        return True

    def _scheduler_tick(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        for req in list(self.pending_queue):
            aid = req.adapter_id
            valid = [v for v in v_nodes if v.get_free_slots(aid) > 0]
            # Check TTFT
            valid_ttft = []
            for v in valid:
                wait_ms = self._clock.now() - req.arrival_time_ms
                exec_time = self._predict_ttft_simple(v, aid)
                total = wait_ms / 1000.0 + exec_time
                if total <= T_MAX:
                    valid_ttft.append(v)

            if valid_ttft:
                target = self._rng.choice(valid_ttft)
                target.commit_request(aid)
                target.node.submit_request(req)
            else:
                self._handle_drop(req, f"System Full or SLO Violation (No valid nodes for T_MAX={T_MAX}s)")
            self.pending_queue.remove(req)

    def _predict_ttft_simple(self, node: VirtualNodeState, target_lora: str) -> float:
        is_in_vram = target_lora in node.active_loras
        is_in_cpu = target_lora in node.loaded_adapters
        load_delay = 0.0 if (is_in_cpu or is_in_vram) else SIM_LOAD_DELAY
        assumed_batch = UNMERGED_CAPACITY - 1
        prefill = SIM_PREFILL_BASE_TIME
        decode_speed = SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch
        all_remains = [r.get("remaining_tokens", 128) for r in node.request_set]
        decode_wait = (sum(all_remains) / len(all_remains) * decode_speed) if all_remains else 0.0
        return SCHEDULER_OVERHEAD + load_delay + decode_wait + prefill


class SimControlNodeLRU(SimControlNodeBase):
    """LRU cache, random or greedy dispatch. No mode switching (always unmerge)."""

    SIM_DOWNLOAD_DELAY = 2.0

    def __init__(self, cluster_id, clock, compute_nodes, lora_metadata,
                 dispatch_strategy="random", efo_ref=None, rng_seed=42):
        super().__init__(cluster_id, clock, compute_nodes, lora_metadata, rng_seed)
        self.dispatch_strategy = dispatch_strategy
        self.efo_ref = efo_ref  # reference to EFO for LRU cache operations

    def admit_request(self, req: SimRequest) -> bool:
        if self.system_paused:
            self._handle_drop(req, "System Paused")
            return False
        meta = self.lora_metadata.get(req.adapter_id)
        is_local = meta and meta.get("type") == "local"
        if not meta or (is_local and meta.get("cluster") != self.cluster_id):
            self._handle_drop(req, "Sovereignty Violation")
            return False
        req.original_adapter_id = req.adapter_id
        self.pending_queue.append(req)
        return True

    def _scheduler_tick(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        for req in list(self.pending_queue):
            aid = req.adapter_id
            download_penalty = 0.0

            # LRU cache check
            if aid not in self.local_available_loras:
                if self.efo_ref:
                    result = self.efo_ref.fetch_and_evict_lora(self.cluster_id, aid)
                    if result.get("downloaded"):
                        self.local_available_loras = set(result.get("current_cache", []))
                        download_penalty = self.SIM_DOWNLOAD_DELAY
                    else:
                        self.local_available_loras = set(result.get("current_cache",
                                                                     list(self.local_available_loras)))
                else:
                    self.local_available_loras.add(aid)
            else:
                if self.efo_ref:
                    self.efo_ref.access_lora(self.cluster_id, aid)

            # Find valid nodes
            valid = []
            for v in v_nodes:
                if v.get_free_slots(aid) > 0:
                    wait_ms = self._clock.now() - req.arrival_time_ms
                    exec_time = self._predict_ttft_simple(v, aid)
                    total = wait_ms / 1000.0 + download_penalty + exec_time
                    if total <= T_MAX:
                        valid.append(v)

            target = None
            if not valid:
                self._handle_drop(req, f"System Full or SLO Violation (No valid nodes for T_MAX={T_MAX}s)")
            elif self.dispatch_strategy == "greedy":
                # Greedy: prefer cache hit, then lowest node index
                cache_hits = [v for v in valid if aid in v.active_loras or aid in v.loaded_adapters]
                if cache_hits:
                    target = cache_hits[0]
                else:
                    target = valid[0]
            else:
                target = self._rng.choice(valid)

            if target:
                target.commit_request(aid)
                target.node.submit_request(req)

            self.pending_queue.remove(req)

    def _predict_ttft_simple(self, node, target_lora):
        is_in = target_lora in node.active_loras or target_lora in node.loaded_adapters
        load_delay = 0.0 if is_in else SIM_LOAD_DELAY
        assumed_batch = UNMERGED_CAPACITY - 1
        prefill = SIM_PREFILL_BASE_TIME
        decode_speed = SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch
        all_remains = [r.get("remaining_tokens", 128) for r in node.request_set]
        decode_wait = (sum(all_remains) / len(all_remains) * decode_speed) if all_remains else 0.0
        return SCHEDULER_OVERHEAD + load_delay + decode_wait + prefill

    def apply_sp1_reset(self, new_loras):
        # LRU mode ignores SP1 reset (per original: "Ignored in LRU mode")
        pass


class SimControlNodeDLoRA(SimControlNodeBase):
    """dLoRA: greedy dispatch + dynamic merge threshold."""

    SIM_DOWNLOAD_DELAY = 2.0
    DLORA_MERGE_RIGHT_THRESHOLD = 1.0
    DLORA_MERGE_LEFT_THRESHOLD = 0.555
    MIN_MERGE_REQUESTS = 4

    def __init__(self, cluster_id, clock, compute_nodes, lora_metadata, efo_ref=None, rng_seed=42):
        super().__init__(cluster_id, clock, compute_nodes, lora_metadata, rng_seed)
        self.efo_ref = efo_ref
        # dLoRA batching controller runs every 500ms alongside scheduler
        self._dlora_handle = clock.schedule_periodic(500, self._dlora_batching_tick)

    def admit_request(self, req: SimRequest) -> bool:
        if self.system_paused:
            self._handle_drop(req, "System Paused")
            return False
        meta = self.lora_metadata.get(req.adapter_id)
        is_local = meta and meta.get("type") == "local"
        if not meta or (is_local and meta.get("cluster") != self.cluster_id):
            self._handle_drop(req, "Sovereignty Violation")
            return False
        req.original_adapter_id = req.adapter_id
        self.pending_queue.append(req)
        return True

    def _dlora_batching_tick(self):
        """dLoRA dynamic merge/unmerge based on queue composition."""
        for node in self.compute_nodes:
            if node.status != NodeStatus.ACTIVE:
                continue
            m = node.get_metrics()
            mode = m.get("mode", "unmerge")
            merged_adapter = m.get("lora_state", {}).get("merged_adapter")
            req_set = m.get("request_set", [])
            total = len(req_set)

            if total < self.MIN_MERGE_REQUESTS:
                if mode == "merge":
                    node.unmerge_all()
                continue

            counts = defaultdict(int)
            for r in req_set:
                counts[r["adapter_id"]] += 1
            l_max = max(counts, key=counts.get)
            ratio = counts[l_max] / total

            if mode == "merge":
                merged_count = counts.get(merged_adapter, 0)
                merged_ratio = merged_count / total
                if merged_ratio < self.DLORA_MERGE_LEFT_THRESHOLD:
                    node.unmerge_all()
                    continue

            if ratio >= self.DLORA_MERGE_RIGHT_THRESHOLD:
                if mode != "merge" or merged_adapter != l_max:
                    node.merge_adapter(l_max)

    def _scheduler_tick(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        for req in list(self.pending_queue):
            aid = req.adapter_id
            download_penalty = 0.0

            if aid not in self.local_available_loras:
                if self.efo_ref:
                    result = self.efo_ref.fetch_and_evict_lora(self.cluster_id, aid)
                    if result.get("downloaded"):
                        self.local_available_loras = set(result.get("current_cache", []))
                        download_penalty = self.SIM_DOWNLOAD_DELAY
                    else:
                        self.local_available_loras = set(result.get("current_cache",
                                                                     list(self.local_available_loras)))
                else:
                    self.local_available_loras.add(aid)
            else:
                if self.efo_ref:
                    self.efo_ref.access_lora(self.cluster_id, aid)

            # Greedy: pick node with minimum expected pending time
            best_node = None
            min_time = float('inf')
            for v in v_nodes:
                pt = self._calc_pending_time(v, aid)
                if pt < min_time:
                    min_time = pt
                    best_node = v

            if not best_node:
                self._handle_drop(req, "System Full (No Nodes Available)")
            else:
                wait_s = (self._clock.now() - req.arrival_time_ms) / 1000.0
                total = wait_s + download_penalty + min_time
                if total > T_MAX * 5.0:
                    self._handle_drop(req, f"Extreme Congestion (Pred: {total:.2f}s)")
                else:
                    best_node.node.submit_request(req)

            self.pending_queue.remove(req)

    def _calc_pending_time(self, node: VirtualNodeState, target_lora: str) -> float:
        is_in_cpu = target_lora in node.loaded_adapters
        load_delay = 0.0 if is_in_cpu else SIM_LOAD_DELAY
        avg_per_req = SIM_DECODE_BASE_TIME * FIXED_OUTPUT_LEN
        queue_len = len(node.request_set)
        return queue_len * avg_per_req + load_delay

    def apply_sp1_reset(self, new_loras):
        # dLoRA ignores SP1 reset
        pass

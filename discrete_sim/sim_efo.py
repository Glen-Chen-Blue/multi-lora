import os, sys, json, math
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Set, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LORA_METADATA_PATH, LORA_SIZE_GB, DISK_CAPACITY_GB,
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB, COST_INST_LOCAL,
    COST_NET_TRAFFIC, COST_DROP_PENALTY, T_MAX_SLO, SWAP_EPSILON,
    SP1_INTERVAL_SECONDS, SP2_INTERVAL_SECONDS, START_OFFSET
)

from .sim_types import SimRequest
from .sim_clock import SimClock
from .sim_control_node import SimControlNodeBase
from .sim_network import SimNetwork
from .sim_logger import SimLogger


class SimEFOBase:
    def __init__(self, clock: SimClock, control_nodes: Dict[str, SimControlNodeBase],
                 lora_metadata: Dict[str, Any], network: SimNetwork,
                 logger: SimLogger, disk_capacity_gb: float = 5.0,
                 simulation_df=None):
        self._clock = clock
        self.control_nodes = control_nodes  # {cluster_name: control_node}
        self.lora_metadata = lora_metadata
        self.network = network
        self.logger = logger
        self.disk_capacity_gb = disk_capacity_gb
        self.simulation_df = simulation_df  # preprocessed pandas DataFrame

        # State
        self.current_time_step = 0
        self.artifact_downloads = 0
        self.cumulative_stored_loras = 0
        self.global_lora_disk_inventory: Dict[str, List[str]] = {}
        self.predicted_demand: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Schedule SP2 routing broadcast
        self._sp2_handle = clock.schedule_periodic(
            SP2_INTERVAL_SECONDS * 1000, self._sp2_tick
        )
        # Schedule metrics logging (SP1/10 = 360s = 360000ms)
        self._metrics_handle = clock.schedule_periodic(
            SP1_INTERVAL_SECONDS * 1000 // 10, self._log_global_metrics
        )
        self._metrics_sub_step = 0

    def trigger_time_edge(self):
        """Called at SP1 interval boundaries. Runs SP1 provisioning."""
        step = self.current_time_step
        self._metrics_sub_step = 0
        self._sp1_tick()
        self._sp2_tick()  # Force sync after SP1
        self.current_time_step += 1
        return {"status": "success", "completed_step": step, "next_step": self.current_time_step}

    def _sp1_tick(self):
        """SP1 provisioning. Subclass implements."""
        raise NotImplementedError

    def _sp2_tick(self):
        """SP2 routing broadcast - gather REAL status from all control nodes."""
        routing_table = {}
        # 取得網路 P95 延遲矩陣
        p95_delays = self.network.get_p95_info(list(self.control_nodes.keys()))

        # 1. 蒐集每個叢集的真實狀態
        for name, cn in self.control_nodes.items():
            if hasattr(cn, 'get_offload_status'):
                status = cn.get_offload_status()
                budget = status.get("budget", 0)
                lora_status = status.get("lora_status", {"merged": [], "loaded": [], "unloaded": []})
            else:
                budget = 0
                lora_status = {"merged": [], "loaded": [], "unloaded": []}

            routing_table[name] = {
                "ip": name,
                "budget": budget,
                "lora_status": lora_status,
                "delay": p95_delays.get(name, {})
            }

        # 2. 廣播更新後的路由表給所有叢集
        for name, cn in self.control_nodes.items():
            if hasattr(cn, 'receive_routing_table'):
                cn.receive_routing_table(routing_table)

    def _log_global_metrics(self):
        """Log global metrics snapshot - called every SP1/10."""
        self._metrics_sub_step += 1

        efo_totals = {
            "total_inference_time": 0.0,
            "total_drops": 0,
            "total_drop_local_congestion": 0,
            "total_drop_no_target": 0,
            "total_offloads": 0,
            "total_local_completed": 0,
            "total_offload_completed": 0,
            "artifact_downloads": self.artifact_downloads,
            "total_stored_loras": self.cumulative_stored_loras,
        }

        cluster_data = {}
        for name, cn in self.control_nodes.items():
            m = cn.get_cluster_metrics()
            cluster_data[name] = m
            d_local = m.get("drop_local_congestion", 0)
            d_no_tgt = m.get("drop_no_target", 0)
            efo_totals["total_inference_time"] += m.get("total_effective_inference_time", 0.0)
            efo_totals["total_drop_local_congestion"] += d_local
            efo_totals["total_drop_no_target"] += d_no_tgt
            efo_totals["total_drops"] += d_local + d_no_tgt
            efo_totals["total_offloads"] += m.get("offload_out", 0)
            efo_totals["total_local_completed"] += m.get("local_completed", 0)
            efo_totals["total_offload_completed"] += m.get("offload_in_completed", 0)

        # total_requests = completed + dropped
        completed = efo_totals["total_local_completed"] + efo_totals["total_offload_completed"]
        efo_totals["total_requests"] = completed + efo_totals["total_drops"]

        self.logger.log_global_metrics(
            timestamp_s=self._clock.now_s(),
            step_id=self.current_time_step,
            sub_step=self._metrics_sub_step,
            cluster_data=cluster_data,
            efo_totals=efo_totals
        )


class SimEFOSP1(SimEFOBase):
    """CSG-Swap provisioning algorithm (from EFO_server.py)."""

    def _sp1_tick(self):
        """CSG-Swap provisioning algorithm."""
        if not self.lora_metadata or not self.control_nodes:
            return

        # 1. CSV Forecasting
        self._exact_csv_forecasting(self.current_time_step)

        # 2. Run CSG-Swap placement
        p95_delays = self.network.get_p95_info(list(self.control_nodes.keys()))

        C_STORE = COST_STORE_PER_GB
        C_DL = COST_DOWNLOAD_PER_GB
        C_INST = COST_INST_LOCAL
        C_NET = COST_NET_TRAFFIC
        C_DROP = COST_DROP_PENALTY
        S_LORA = LORA_SIZE_GB
        CAPACITY = int(self.disk_capacity_gb / S_LORA)
        T_MAX = T_MAX_SLO
        EPS = SWAP_EPSILON

        # Build serves_map (reverse substitution)
        serves_map = defaultdict(set)
        for lid in self.lora_metadata:
            serves_map[lid].add(lid)
        for lid, info in self.lora_metadata.items():
            for parent in info.get("substitutes", []):
                serves_map[parent].add(lid)

        def is_covered(target_id, stored_set):
            if target_id in stored_set:
                return True
            subs = self.lora_metadata.get(target_id, {}).get("substitutes", [])
            return any(s in stored_set for s in subs)

        def calc_marginal(cluster, cand_id, current_set):
            total = 0.0
            for tid in serves_map.get(cand_id, set()):
                if not is_covered(tid, current_set):
                    total += self.predicted_demand[cluster].get(tid, 0.0)
            return total

        cluster_targets = {}

        # Phase 1: Local provisioning per cluster
        for cluster_name in self.control_nodes:
            target_disk = set()
            mandatory = set()

            cluster_valid = [l for l, info in self.lora_metadata.items()
                            if info.get("type") == "global" or
                            (info.get("type") == "local" and info.get("cluster") == cluster_name)]

            for l in cluster_valid:
                if self.lora_metadata[l].get("type") == "local":
                    mandatory.add(l)
                    target_disk.add(l)

            # Retain existing valuable LoRAs
            current_disk = set(self.global_lora_disk_inventory.get(cluster_name, []))
            for l in current_disk:
                if l in mandatory or l not in [x for x in cluster_valid]:
                    continue
                temp = target_disk.union(current_disk) - {l}
                if is_covered(l, temp):
                    continue
                best_offload_cost = C_DROP
                offload_costs = []
                for k in self.control_nodes:
                    if k == cluster_name:
                        continue
                    delay_sec = p95_delays.get(cluster_name, {}).get(k, 1000.0) / 1000.0
                    gamma = T_MAX / (T_MAX - delay_sec) if delay_sec < T_MAX else float('inf')
                    offload_costs.append(gamma * C_INST + C_NET)
                if offload_costs:
                    best_offload_cost = min(min(offload_costs), C_DROP)
                gain = max(0.0, best_offload_cost - C_INST)
                lambd = self.predicted_demand[cluster_name].get(l, 0.0)
                if (lambd * gain) - (S_LORA * C_STORE) >= 0:
                    target_disk.add(l)

            # Greedy expansion with swap
            candidates = [l for l in cluster_valid if l not in target_disk]
            while True:
                best_cand = None
                max_u = -float('inf')
                for cand in candidates:
                    new_demand = calc_marginal(cluster_name, cand, target_disk)
                    benefit = new_demand * C_DROP
                    cost = S_LORA * (C_STORE + C_DL)
                    net_u = benefit - cost
                    if net_u > max_u:
                        max_u = net_u
                        best_cand = cand

                if best_cand is None or max_u <= 0:
                    break

                if len(target_disk) < CAPACITY:
                    target_disk.add(best_cand)
                    candidates.remove(best_cand)
                else:
                    swappable = sorted([t for t in target_disk if t not in mandatory])
                    if not swappable:
                        break
                    victim = None
                    min_loss = float('inf')
                    for t in swappable:
                        temp = target_disk - {t}
                        loss = calc_marginal(cluster_name, t, temp)
                        loss_val = (loss * C_DROP) - (S_LORA * C_STORE)
                        if loss_val < min_loss:
                            min_loss = loss_val
                            victim = t
                    if max_u > min_loss + EPS:
                        target_disk.remove(victim)
                        target_disk.add(best_cand)
                        candidates.remove(best_cand)
                        candidates.append(victim)
                    else:
                        break

            # Count downloads
            current_static = set(self.global_lora_disk_inventory.get(cluster_name, []))
            new_items = target_disk - current_static
            real_downloads = sum(1 for l in new_items if self.lora_metadata.get(l, {}).get("type") != "local")
            self.artifact_downloads += real_downloads

            cluster_targets[cluster_name] = list(target_disk)

        # Phase 2: Global semantic rescue (simplified)
        global_cands = [l for l, info in self.lora_metadata.items() if info.get("type") == "global"]
        global_cands.sort(key=lambda l: sum(self.predicted_demand[c].get(l, 0) for c in self.control_nodes), reverse=True)

        for cand_id in global_cands:
            best_cluster = None
            best_u = -float('inf')
            for c in self.control_nodes:
                current_set = set(cluster_targets[c])
                if is_covered(cand_id, current_set):
                    continue
                marginal = calc_marginal(c, cand_id, current_set)
                if marginal <= 0:
                    continue
                net_u = marginal * C_DROP - S_LORA * (C_STORE + C_DL)
                if net_u > 0 and net_u > best_u:
                    best_u = net_u
                    best_cluster = c
            if best_cluster:
                ts = set(cluster_targets[best_cluster])
                if len(ts) < CAPACITY:
                    ts.add(cand_id)
                    cluster_targets[best_cluster] = list(ts)
                    self.artifact_downloads += 1

        # Update stored loras count
        total_stored = sum(len(l) for l in cluster_targets.values())
        self.cumulative_stored_loras += total_stored

        # Save inventory
        self.global_lora_disk_inventory.clear()
        for c, loras in cluster_targets.items():
            self.global_lora_disk_inventory[c] = list(loras)

        # Apply SP1 to control nodes
        for c, loras in cluster_targets.items():
            self.control_nodes[c].apply_sp1_reset(loras)

    def _exact_csv_forecasting(self, time_step):
        """Port of exact_csv_forecasting from EFO_server.py"""
        self.predicted_demand.clear()
        start_sec = time_step * SP1_INTERVAL_SECONDS
        end_sec = (time_step + 1) * SP1_INTERVAL_SECONDS

        for c in self.control_nodes:
            self.predicted_demand[c] = {l: 0.0 for l in self.lora_metadata}

        if self.simulation_df is None:
            return

        df = self.simulation_df
        mask = (df["arrival_sec"] >= start_sec) & (df["arrival_sec"] < end_sec)
        target_clusters = list(self.control_nodes.keys())
        filtered = df[mask & df["cluster"].isin(target_clusters)]

        for _, row in filtered.iterrows():
            cluster = str(row["cluster"]).strip()
            try:
                lora_id = f"LoRA_{int(float(row['lora_id']))}"
            except:
                lora_id = str(row["lora_id"])
            if cluster in self.predicted_demand and lora_id in self.predicted_demand[cluster]:
                self.predicted_demand[cluster][lora_id] += 1.0


class SimEFOLRU(SimEFOBase):
    """LRU cache provisioning (from EFO_server_lru.py)."""

    def __init__(self, clock, control_nodes, lora_metadata, network, logger, disk_capacity_gb=5.0, **kw):
        super().__init__(clock, control_nodes, lora_metadata, network, logger, disk_capacity_gb, **kw)
        self.cluster_lru_caches: Dict[str, OrderedDict] = {}
        self._efo_downloads: Dict[str, int] = defaultdict(int)
        # Initialize caches with local LoRAs
        for cluster_name in control_nodes:
            self._init_cluster_lru(cluster_name)
            # Set initial available loras on control node
            cn = control_nodes[cluster_name]
            cn.local_available_loras = set(self.cluster_lru_caches[cluster_name].keys())

    def _init_cluster_lru(self, cluster_name):
        self.cluster_lru_caches[cluster_name] = OrderedDict()
        for lora_id, info in self.lora_metadata.items():
            if info.get("type") == "local" and info.get("cluster") == cluster_name:
                self.cluster_lru_caches[cluster_name][lora_id] = True

    def _get_capacity(self):
        return max(1, int(self.disk_capacity_gb / LORA_SIZE_GB))

    def access_lora(self, cluster_name, lora_id):
        cache = self.cluster_lru_caches.get(cluster_name)
        if cache and lora_id in cache:
            cache.move_to_end(lora_id)

    def fetch_and_evict_lora(self, cluster_name, lora_id) -> dict:
        cache = self.cluster_lru_caches.get(cluster_name)
        if cache is None:
            return {"status": "error"}
        if lora_id in cache:
            cache.move_to_end(lora_id)
            return {"status": "ok", "downloaded": False, "current_cache": list(cache.keys())}

        cache[lora_id] = True
        self._efo_downloads[cluster_name] += 1
        self.artifact_downloads += 1

        cap = self._get_capacity()
        if len(cache) > cap:
            for k in list(cache.keys()):
                info = self.lora_metadata.get(k, {})
                is_local = info.get("type") == "local" and info.get("cluster") == cluster_name
                if not is_local:
                    del cache[k]
                    break

        # Update stored count
        self.cumulative_stored_loras = sum(len(c) for c in self.cluster_lru_caches.values())
        return {"status": "ok", "downloaded": True, "current_cache": list(cache.keys())}

    def _sp1_tick(self):
        # LRU EFO does no SP1 provisioning - just logging
        self.cumulative_stored_loras = sum(len(c) for c in self.cluster_lru_caches.values())

    def _log_global_metrics(self):
        # Override to include LRU-specific download tracking
        self._metrics_sub_step += 1
        efo_totals = {
            "total_inference_time": 0.0,
            "total_drops": 0,
            "total_drop_local_congestion": 0,
            "total_drop_no_target": 0,
            "total_offloads": 0,
            "total_local_completed": 0,
            "total_offload_completed": 0,
            "artifact_downloads": sum(self._efo_downloads.values()),
            "total_stored_loras": sum(len(c) for c in self.cluster_lru_caches.values()),
        }
        cluster_data = {}
        for name, cn in self.control_nodes.items():
            m = cn.get_cluster_metrics()
            cluster_data[name] = m
            d_local = m.get("drop_local_congestion", 0)
            d_no_tgt = m.get("drop_no_target", 0)
            efo_totals["total_inference_time"] += m.get("total_effective_inference_time", 0.0)
            efo_totals["total_drop_local_congestion"] += d_local
            efo_totals["total_drop_no_target"] += d_no_tgt
            efo_totals["total_drops"] += d_local + d_no_tgt
            efo_totals["total_offloads"] += m.get("offload_out", 0)
            efo_totals["total_local_completed"] += m.get("local_completed", 0)
            efo_totals["total_offload_completed"] += m.get("offload_in_completed", 0)
        completed = efo_totals["total_local_completed"] + efo_totals["total_offload_completed"]
        efo_totals["total_requests"] = completed + efo_totals["total_drops"]
        self.logger.log_global_metrics(self._clock.now_s(), self.current_time_step, self._metrics_sub_step, cluster_data, efo_totals)


class SimEFODLoRA(SimEFOBase):
    """LFU with decay provisioning (from EFO_server_dlora.py)."""

    def __init__(self, clock, control_nodes, lora_metadata, network, logger, disk_capacity_gb=5.0, **kw):
        super().__init__(clock, control_nodes, lora_metadata, network, logger, disk_capacity_gb, **kw)
        self.cluster_disk_state: Dict[str, Set[str]] = {}
        self.cluster_lora_freq: Dict[str, Dict[str, float]] = {}
        self._efo_downloads: Dict[str, int] = defaultdict(int)
        for cluster_name in control_nodes:
            self._init_cluster(cluster_name)
            cn = control_nodes[cluster_name]
            cn.local_available_loras = set(self.cluster_disk_state[cluster_name])

    def _init_cluster(self, cluster_name):
        self.cluster_disk_state[cluster_name] = set()
        self.cluster_lora_freq[cluster_name] = defaultdict(float)
        for lora_id, info in self.lora_metadata.items():
            if info.get("type") == "local" and info.get("cluster") == cluster_name:
                self.cluster_disk_state[cluster_name].add(lora_id)
                self.cluster_lora_freq[cluster_name][lora_id] = 999999.0

    def _get_capacity(self):
        return max(1, int(self.disk_capacity_gb / LORA_SIZE_GB))

    def access_lora(self, cluster_name, lora_id):
        self.cluster_lora_freq[cluster_name][lora_id] += 1.0

    def fetch_and_evict_lora(self, cluster_name, lora_id) -> dict:
        disk = self.cluster_disk_state.get(cluster_name)
        if disk is None:
            return {"status": "error"}

        self.cluster_lora_freq[cluster_name][lora_id] += 1.0

        if lora_id in disk:
            return {"status": "ok", "downloaded": False, "current_cache": list(disk)}

        disk.add(lora_id)
        self._efo_downloads[cluster_name] += 1
        self.artifact_downloads += 1

        cap = self._get_capacity()
        if len(disk) > cap:
            # Evict lowest frequency non-local
            min_freq = float('inf')
            victim = None
            for k in disk:
                info = self.lora_metadata.get(k, {})
                is_local = info.get("type") == "local" and info.get("cluster") == cluster_name
                if not is_local:
                    f = self.cluster_lora_freq[cluster_name].get(k, 0)
                    if f < min_freq:
                        min_freq = f
                        victim = k
            if victim:
                disk.discard(victim)

        self.cumulative_stored_loras = sum(len(s) for s in self.cluster_disk_state.values())
        return {"status": "ok", "downloaded": True, "current_cache": list(disk)}

    def _sp1_tick(self):
        # Time decay: multiply non-local frequencies by 0.5
        for cluster_name in self.control_nodes:
            freq = self.cluster_lora_freq.get(cluster_name, {})
            for lora_id in list(freq.keys()):
                info = self.lora_metadata.get(lora_id, {})
                is_local = info.get("type") == "local" and info.get("cluster") == cluster_name
                if not is_local:
                    freq[lora_id] *= 0.5
        self.cumulative_stored_loras = sum(len(s) for s in self.cluster_disk_state.values())

    def _log_global_metrics(self):
        self._metrics_sub_step += 1
        efo_totals = {
            "total_inference_time": 0.0,
            "total_drops": 0, "total_drop_local_congestion": 0, "total_drop_no_target": 0,
            "total_offloads": 0, "total_local_completed": 0, "total_offload_completed": 0,
            "artifact_downloads": sum(self._efo_downloads.values()),
            "total_stored_loras": sum(len(s) for s in self.cluster_disk_state.values()),
        }
        cluster_data = {}
        for name, cn in self.control_nodes.items():
            m = cn.get_cluster_metrics()
            cluster_data[name] = m
            d_local = m.get("drop_local_congestion", 0)
            d_no_tgt = m.get("drop_no_target", 0)
            efo_totals["total_inference_time"] += m.get("total_effective_inference_time", 0.0)
            efo_totals["total_drop_local_congestion"] += d_local
            efo_totals["total_drop_no_target"] += d_no_tgt
            efo_totals["total_drops"] += d_local + d_no_tgt
            efo_totals["total_offloads"] += m.get("offload_out", 0)
            efo_totals["total_local_completed"] += m.get("local_completed", 0)
            efo_totals["total_offload_completed"] += m.get("offload_in_completed", 0)
        completed = efo_totals["total_local_completed"] + efo_totals["total_offload_completed"]
        efo_totals["total_requests"] = completed + efo_totals["total_drops"]
        self.logger.log_global_metrics(self._clock.now_s(), self.current_time_step, self._metrics_sub_step, cluster_data, efo_totals)
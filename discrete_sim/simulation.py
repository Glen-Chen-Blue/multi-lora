import os, sys, json, uuid, math, gc, time as wall_time
from datetime import timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SP1_INTERVAL_SECONDS, START_OFFSET, LORA_SIZE_GB

from .sim_types import (
    SimRequest, SimulationConfig, EXPERIMENT_CONFIGS,
    NodeMode, NodeStatus
)
from .sim_clock import SimClock
from .sim_network import SimNetwork
from .sim_logger import SimLogger
from .sim_trace_reader import SimTraceReader
from .sim_compute_node import SimComputeNode
from .sim_control_node import (
    SimControlNodeBase, SimControlNodeSP2, SimControlNodeRandom,
    SimControlNodeLRU, SimControlNodeDLoRA
)
from .sim_efo import SimEFOBase, SimEFOSP1, SimEFOLRU, SimEFODLoRA


def _format_sim_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS."""
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.exp_def = config.get_experiment_def()
        self.clock = SimClock()
        self.network = SimNetwork(seed=config.seed)

        # Stats (matching test_simulation.py)
        self.stats = {"sent": 0, "finished": 0, "dropped": 0, "errors": 0}
        self.ttft_records: List[float] = []
        self.all_requests: List[SimRequest] = []
        self.finished_requests: List[SimRequest] = []
        self.dropped_requests: List[SimRequest] = []
        self._in_transit = []  # [新增] Track offloaded requests in flight

        # Load metadata
        metadata_path = os.path.join(config.metadata_dir, self.exp_def.metadata_file)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.lora_metadata = json.load(f)

        # Load simulation CSV for EFO forecasting
        self.simulation_df = None
        if os.path.exists(config.trace_csv):
            df = pd.read_csv(config.trace_csv)
            df["arrival_sec"] = df["arrive_timestamp"].astype(float)
            df = df[df["arrival_sec"] >= config.start_offset].copy()
            df["arrival_sec"] -= config.start_offset
            self.simulation_df = df

        # Build topology
        self._build_topology()

        # Load trace
        duration_s = config.duration_hours * 3600
        self.trace = SimTraceReader(
            config.trace_csv, config.start_offset, duration_s,
            config.get_target_clusters()
        )
        self.TOTAL_REQUESTS = self.trace.total_requests
        self.PAD_LEN = len(str(self.TOTAL_REQUESTS))
        self._request_idx = 0

    def _build_topology(self):
        """Create EFO, control nodes, compute nodes based on experiment config."""
        disk_gb = self.config.get_disk_capacity_gb()
        strategy = self.config.get_dispatch_strategy()

        # Setup output dir
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.sim_logger = SimLogger(self.config.output_dir)

        # Create compute nodes per cluster
        self.all_compute_nodes: Dict[str, List[SimComputeNode]] = {}
        for cluster_name, num_nodes in self.config.cluster_topology.items():
            nodes = []
            for i in range(num_nodes):
                node_id = f"c{cluster_name.split('_')[-1]}-n{i+1}"
                node = SimComputeNode(node_id, cluster_name, self.clock)
                nodes.append(node)
            self.all_compute_nodes[cluster_name] = nodes

        # Create control nodes
        self.control_nodes: Dict[str, SimControlNodeBase] = {}
        ctrl_type = self.exp_def.control_type

        # We'll create EFO first for LRU/dLoRA variants that need efo_ref
        for cluster_name, nodes in self.all_compute_nodes.items():
            seed = hash(cluster_name) & 0xFFFFFFFF
            if ctrl_type == "sp2":
                cn = SimControlNodeSP2(cluster_name, self.clock, nodes, self.lora_metadata, rng_seed=seed)
            elif ctrl_type == "random":
                cn = SimControlNodeRandom(cluster_name, self.clock, nodes, self.lora_metadata, rng_seed=seed)
            elif ctrl_type == "lru":
                cn = SimControlNodeLRU(cluster_name, self.clock, nodes, self.lora_metadata,
                                       dispatch_strategy=strategy, efo_ref=None, rng_seed=seed)
            elif ctrl_type == "dlora":
                cn = SimControlNodeDLoRA(cluster_name, self.clock, nodes, self.lora_metadata,
                                         efo_ref=None, rng_seed=seed)
            else:
                raise ValueError(f"Unknown control type: {ctrl_type}")
            self.control_nodes[cluster_name] = cn

        # Create EFO
        efo_type = self.exp_def.efo_type
        if efo_type == "sp1":
            self.efo = SimEFOSP1(self.clock, self.control_nodes, self.lora_metadata,
                                  self.network, self.sim_logger, disk_gb,
                                  simulation_df=self.simulation_df)
        elif efo_type == "lru":
            self.efo = SimEFOLRU(self.clock, self.control_nodes, self.lora_metadata,
                                  self.network, self.sim_logger, disk_gb)
        elif efo_type == "dlora":
            self.efo = SimEFODLoRA(self.clock, self.control_nodes, self.lora_metadata,
                                    self.network, self.sim_logger, disk_gb)
        else:
            raise ValueError(f"Unknown EFO type: {efo_type}")

        # Link efo_ref for LRU/dLoRA control nodes and setup Offload Callback
        for cn in self.control_nodes.values():
            if hasattr(cn, 'efo_ref'):
                cn.efo_ref = self.efo
            
            # [新增] 綁定 Offload 回呼函數，允許跨叢集拋轉
            # (預設 target_cluster=None 是為了相容尚未修改的 sim_control_node)
            cn.offload_callback = lambda req, tgt=None, src=cn.cluster_id: self._handle_offload(req, src, tgt)

    def _handle_offload(self, req: SimRequest, source_cluster: str, target_cluster: str = None) -> bool:
        """[新增] Global offload handler. Simulates network transmission to target cluster."""
        if target_cluster is None or target_cluster not in self.control_nodes:
            return False
            
        delay_ms = self.network.get_delay_ms(source_cluster, target_cluster)
        delivery_time = self.clock.now() + int(delay_ms)
        
        req.is_delegated = True
        req.assigned_cluster = target_cluster
        
        # 標記進入網路傳輸佇列
        self._in_transit.append((delivery_time, req, target_cluster))
        return True

    def _on_request_finish(self, req: SimRequest):
        """Called when a request finishes (via compute node callback chain)."""
        self.stats["finished"] += 1
        self.finished_requests.append(req)
        if req.ttft_s is not None:
            self.ttft_records.append(req.ttft_s)
            
        # [修改] 關閉單筆完成的輸出，避免洗頻
        # ts = _format_sim_time(self.clock.now())
        # idx = getattr(req, '_sim_idx', '?')
        # req_str = f"{idx:>{self.PAD_LEN}}/{self.TOTAL_REQUESTS}"
        # adapter_str = f"{req.original_adapter_id:^8}"
        # elapsed = req.total_time_s or 0.0
        # ttft = req.ttft_s or elapsed
        # del_str = "[O]" if req.is_delegated else "[L]"
        # print(f"[{ts}] [DONE] Req:{req_str} {del_str} | Target:{adapter_str} | Time: {elapsed:>6.2f}s | TTFT: {ttft:>5.2f}s | Tokens: {req.tokens_generated}")

    def _on_request_first_token(self, req: SimRequest):
        """Called on first token."""
        pass  # TTFT recorded in req object

    def run(self):
        """Main simulation loop."""
        duration_ms = self.config.duration_hours * 3600 * 1000
        sp1_interval_ms = SP1_INTERVAL_SECONDS * 1000

        # [修復] Wire callbacks (Chain them so Control Node also gets notified)
        for cn in self.control_nodes.values():
            for node in cn.compute_nodes:
                orig_finish_cb = node.on_request_finish
                orig_first_token_cb = node.on_request_first_token

                def make_finish_cb(orig_cb):
                    def chained_cb(req):
                        if orig_cb:
                            orig_cb(req)
                        self._on_request_finish(req)
                    return chained_cb

                def make_first_token_cb(orig_cb):
                    def chained_cb(req):
                        if orig_cb:
                            orig_cb(req)
                        self._on_request_first_token(req)
                    return chained_cb

                node.on_request_finish = make_finish_cb(orig_finish_cb)
                node.on_request_first_token = make_first_token_cb(orig_first_token_cb)

        target_clusters = self.config.get_target_clusters()

        # Print header
        print("=" * 65)
        print("=== Trace Replay Pressure Simulator (Discrete) ===")
        print("=" * 65)
        print(f"[INFO] Experiment        : {self.config.experiment_id} ({self.exp_def.efo_type}+{self.exp_def.control_type})")
        print(f"[INFO] Topology          : {self.config.cluster_topology}")
        print(f"[INFO] Target Clusters   : {target_clusters}")
        print(f"[INFO] Duration          : {self.config.duration_hours}h ({duration_ms}ms)")
        print(f"[INFO] Requests Count    : {self.TOTAL_REQUESTS}")
        print("-" * 65)

        # Trigger initial SP1
        ts = _format_sim_time(0)
        print(f"\n[{ts}] [SYS ] Triggering initial SP1 /time_edge (Step 0)...")
        result = self.efo.trigger_time_edge()
        print(f"[{ts}] [SYS ] Initial SP1 complete: {result}\n")

        current_interval = 0
        wall_start = wall_time.time()
        last_progress = 0

        # Main loop
        for t in range(1, duration_ms + 1):
            # 1. Advance clock
            self.clock.advance(1)
            
            # [新增] 2. 處理在途的跨叢集 Offload 請求 (模擬網路延遲抵達)
            if self._in_transit:
                still_in_transit = []
                for item in self._in_transit:
                    delivery_time, req, target_cluster = item
                    
                    if t >= delivery_time:
                        # [時間已到] 請求已抵達，交給目標 Control Node 處理
                        cn = self.control_nodes.get(target_cluster)
                        if cn:
                            admit = cn.admit_request(req)
                            if not admit or req.is_dropped:
                                self.stats["dropped"] += 1
                                self.dropped_requests.append(req)
                    else:
                        # [時間未到] 請求還沒抵達，保留在在途陣列中
                        still_in_transit.append(item)
                
                # 直接用保留下來的項目覆寫原本的陣列
                # 這是一個完美的 O(N) 單次遍歷，完全避免了 unhashable 問題，且效能比原本用 not in 過濾快非常多！
                self._in_transit = still_in_transit

            # 3. Inject requests from trace
            events = self.trace.get_requests_at(t)
            for cluster, lora_id in events:
                if cluster not in self.control_nodes:
                    continue

                self._request_idx += 1
                adapter_id = f"LoRA_{lora_id}"
                req = SimRequest(
                    request_id=str(uuid.uuid4()),
                    adapter_id=adapter_id,
                    original_adapter_id=adapter_id,
                    cluster=cluster,
                    arrival_time_ms=t,
                    max_new_tokens=256,
                )
                req._sim_idx = self._request_idx
                self.all_requests.append(req)
                self.stats["sent"] += 1

                # Check interval boundary
                arrival_sec = t / 1000.0
                req_interval = int(arrival_sec // SP1_INTERVAL_SECONDS)
                if req_interval > current_interval:
                    # 這是系統重要訊息，保留不砍
                    ts_str = _format_sim_time(t)
                    self.efo.trigger_time_edge()
                    current_interval = req_interval

                # Admit to control node
                cn = self.control_nodes[cluster]
                admitted = cn.admit_request(req)
                if not admitted or req.is_dropped:
                    self.stats["dropped"] += 1
                    self.dropped_requests.append(req)
                    # [修改] 關閉單筆丟棄的輸出
                    # ts_str = _format_sim_time(t)
                    # reason = req.drop_reason or "Unknown"
                    # print(f"[{ts_str}] [DROP] Req:{req_str} | Target:{adapter_str} | Reason: {reason}")

            # 4. Step all compute nodes
            for cluster_nodes in self.all_compute_nodes.values():
                for node in cluster_nodes:
                    node.step()

            # 5. Check for newly dropped requests from scheduler
            for cn in self.control_nodes.values():
                pass  # drops are handled inline in scheduler_tick

            if t % 10000 == 0:
                wall_time.sleep(0.01)
            if t % 100000 == 0:
                gc.collect()

        # Wait for remaining in-flight requests (give extra time)
        extra_ms = 30000  # 30s extra
        for t in range(extra_ms):
            self.clock.advance(1)
            for cluster_nodes in self.all_compute_nodes.values():
                for node in cluster_nodes:
                    node.step()

        # Close logger
        self.sim_logger.close()

        # Print summary
        print("\n\n" + "=" * 65)
        print(f"=== SUMMARY: Sent: {self.stats['sent']} | Finished: {self.stats['finished']} | Dropped: {self.stats['dropped']} | Errors: {self.stats['errors']} ===")
        print("=" * 65)

        if self.ttft_records:
            avg = sum(self.ttft_records) / len(self.ttft_records)
            sorted_ttft = sorted(self.ttft_records)
            p95_idx = int(len(sorted_ttft) * 0.95)
            p95 = sorted_ttft[min(p95_idx, len(sorted_ttft) - 1)]
            print(f"[STAT] Average TTFT : {avg:.4f} s")
            print(f"[STAT] P95 TTFT     : {p95:.4f} s")
        print("-" * 65 + "\n")

        total_wall = wall_time.time() - wall_start
        print(f"[INFO] Total wall-clock time: {total_wall:.1f}s")
        print(f"[INFO] Logs saved to: {self.config.output_dir}")
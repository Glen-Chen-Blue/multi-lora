import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .sim_types import SimRequest, NodeMode, NodeStatus
from .sim_clock import SimClock
from .sim_engine import SimMultiLoRAEngine
from typing import Optional, List, Dict, Set, Callable


class SimComputeNode:
    """Wraps SimMultiLoRAEngine, tracks node-level state."""

    def __init__(self, node_id: str, cluster_id: str, clock: SimClock):
        self.node_id = node_id
        self.cluster_id = cluster_id
        
        # [任務 A] 設定正確的初始狀態：只有叢集的第一個節點 (n1) 預設為 ACTIVE
        if self.node_id.endswith("-n1"):
            self.status: NodeStatus = NodeStatus.ACTIVE
        else:
            self.status: NodeStatus = NodeStatus.STANDBY
            
        self.engine = SimMultiLoRAEngine(node_id, clock)
        self._clock = clock
        self.cumulative_inference_time_ms: int = 0
        self._last_step_did_work = False

        # Callbacks - set by control node
        self.on_request_first_token: Optional[Callable] = None  # (req) -> None
        self.on_request_finish: Optional[Callable] = None       # (req) -> None
        self.on_request_drop: Optional[Callable] = None         # (req) -> None

        # Wire engine callbacks
        self.engine.on_token = self._on_engine_token
        self.engine.on_finish = self._on_engine_finish

    def _on_engine_token(self, req: SimRequest, token_count: int = 0):
        """Called by engine when a token is generated."""
        if req.tokens_generated == 1:
            if self.on_request_first_token:
                self.on_request_first_token(req)

    def _on_engine_finish(self, req: SimRequest):
        """Called by engine when request completes."""
        req.finish_time_ms = self._clock.now()
        req.is_finished = True
        if self.on_request_finish:
            self.on_request_finish(req)

    def submit_request(self, req: SimRequest) -> bool:
        """Submit a request to this node. Returns False if not accepting."""
        if self.status != NodeStatus.ACTIVE:
            return False
        req.assigned_node = self.node_id
        req.assigned_time_ms = self._clock.now()
        self.engine.add_request(req)
        return True

    def step(self) -> None:
        """Advance by 1ms."""
        if self.status == NodeStatus.STANDBY:
            return
        did_work = self.engine.step()
        if did_work:
            self.cumulative_inference_time_ms += 1
        self._last_step_did_work = did_work
        # Auto-transition from draining to standby
        if self.status == NodeStatus.DRAINING and self.engine.is_idle():
            self.status = NodeStatus.STANDBY

    def get_mode(self) -> NodeMode:
        return self.engine.mode

    def merge_adapter(self, adapter_id: str):
        self.engine.merge_adapter(adapter_id)

    def unmerge_all(self):
        self.engine.unmerge_all()

    def drain(self):
        """Start draining - stop accepting new requests."""
        self.status = NodeStatus.DRAINING

    def activate(self):
        """Wake up from standby."""
        self.status = NodeStatus.ACTIVE

    def full_reset(self):
        """Full reset of engine state."""
        self.engine.full_reset()
        # [任務 A 修復] SP1 重置時也要恢復正確的初始狀態
        if self.node_id.endswith("-n1"):
            self.status = NodeStatus.ACTIVE
        else:
            self.status = NodeStatus.STANDBY

    def update_known_adapters(self, adapters: List[str]):
        self.engine.update_known_adapters(adapters)

    def get_metrics(self) -> dict:
        """Return metrics snapshot matching original compute_node_server format."""
        engine = self.engine
        running_adapters = list(engine.get_active_loras())
        loaded_adapters = engine.get_loaded_adapters()

        return {
            "status": self.status.value,
            "mode": engine.mode.value,
            "load": {
                "running_batch": engine.get_running_count(),
                "waiting_queue": engine.get_queue_depth(),
            },
            "lora_state": {
                "merged_adapter": engine.current_merged_adapter,
                "running_adapters": running_adapters,
                "loaded_adapters": loaded_adapters,
            },
            "request_set": engine.get_request_set(),
            "metrics": {
                "effective_inference_time": self.cumulative_inference_time_ms / 1000.0,
            },
        }
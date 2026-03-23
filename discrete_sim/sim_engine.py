"""Discrete-time MultiLoRA inference engine.

Replaces _multilora_system.py (which used time.sleep) with a state machine
that uses countdown timers, advancing 1ms per step() call.
"""

import math
import os
import sys
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FIXED_OUTPUT_LEN,
    MAX_CPU_LORAS,
    MERGE_SPEED_MULTIPLIER,
    MERGED_CAPACITY,
    SIM_DECODE_BASE_TIME,
    SIM_DECODE_SLOPE,
    SIM_LOAD_DELAY,
    SIM_PREFILL_BASE_TIME,
    UNMERGED_CAPACITY,
)

from .sim_clock import SimClock
from .sim_types import EnginePhase, NodeMode, SimRequest

# Pre-compute millisecond equivalents of config constants
_LOAD_DELAY_MS = int(round(SIM_LOAD_DELAY * 1000))        # 66 ms
_PREFILL_BASE_MS = int(round(SIM_PREFILL_BASE_TIME * 1000))  # 65 ms
_DECODE_BASE_MS = int(round(SIM_DECODE_BASE_TIME * 1000))    # 25 ms
_DECODE_SLOPE_MS = int(round(SIM_DECODE_SLOPE * 1000))       # 1 ms


class SimMultiLoRAEngine:
    """Discrete-time simulation engine for a single compute node."""

    def __init__(self, node_id: str, clock: SimClock):
        self.node_id = node_id
        self.clock = clock

        # Operating mode
        self.mode: NodeMode = NodeMode.UNMERGED
        self.current_merged_adapter: Optional[str] = None

        # GPU slot management (slot_id -> adapter_id)
        self.gpu_slots: Dict[int, str] = {}
        self.adapter_to_slot: Dict[str, int] = {}

        # CPU LRU cache (adapter_id -> True, ordered by last access)
        self.cpu_cache: OrderedDict[str, bool] = OrderedDict()

        # Known adapters (those available on disk / downloadable)
        self.known_adapters: Set[str] = set()

        # Request queues
        self.request_queue: List[SimRequest] = []
        self.running_requests: List[SimRequest] = []

        # Phase state machine
        self._phase: EnginePhase = EnginePhase.IDLE
        self._phase_remaining_ms: int = 0
        self._pending_loads: List[str] = []
        self._current_batch: List[SimRequest] = []

        # Callbacks
        self.on_token: Optional[Callable] = None
        self.on_finish: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, req: SimRequest) -> None:
        """Enqueue a new inference request."""
        self.request_queue.append(req)

    def step(self) -> bool:
        """Advance the engine by 1 ms. Returns True if work was performed."""
        if self._phase == EnginePhase.IDLE:
            return self._step_idle()
        elif self._phase == EnginePhase.LOADING:
            return self._step_loading()
        elif self._phase == EnginePhase.PREFILL:
            return self._step_prefill()
        elif self._phase == EnginePhase.DECODE:
            return self._step_decode()
        return False

    def merge_adapter(self, adapter_id: str) -> None:
        """Switch to merged (dedicated) mode for a specific adapter."""
        self.mode = NodeMode.MERGED
        self.current_merged_adapter = adapter_id

    def unmerge_all(self) -> None:
        """Switch back to unmerged (shared) mode."""
        self.mode = NodeMode.UNMERGED
        self.current_merged_adapter = None

    def full_reset(self) -> None:
        """Clear all engine state."""
        self.mode = NodeMode.UNMERGED
        self.current_merged_adapter = None
        self.gpu_slots.clear()
        self.adapter_to_slot.clear()
        self.cpu_cache.clear()
        self.known_adapters.clear()
        self.request_queue.clear()
        self.running_requests.clear()
        self._phase = EnginePhase.IDLE
        self._phase_remaining_ms = 0
        self._pending_loads.clear()
        self._current_batch.clear()

    def update_known_adapters(self, adapters: List[str]) -> None:
        """Update the set of adapters this node knows about."""
        self.known_adapters = set(adapters)

    def is_idle(self) -> bool:
        return (
            self._phase == EnginePhase.IDLE
            and len(self.running_requests) == 0
            and len(self.request_queue) == 0
        )

    def get_running_count(self) -> int:
        return len(self.running_requests)

    def get_queue_depth(self) -> int:
        return len(self.request_queue)

    def get_active_loras(self) -> Set[str]:
        """Return set of adapter IDs currently being served by running requests."""
        return {r.adapter_id for r in self.running_requests}

    def get_loaded_adapters(self) -> List[str]:
        """Return all adapters currently on GPU + CPU."""
        gpu_adapters = list(self.gpu_slots.values())
        cpu_adapters = list(self.cpu_cache.keys())
        return gpu_adapters + [a for a in cpu_adapters if a not in self.adapter_to_slot]

    def get_request_set(self) -> List[dict]:
        """Return a list of dicts for metrics: adapter_id and remaining_tokens."""
        result = []
        for r in self.running_requests:
            remaining = r.max_new_tokens - r.tokens_generated
            result.append({
                "adapter_id": r.adapter_id,
                "remaining_tokens": max(0, remaining),
            })
        return result

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    def _capacity(self) -> int:
        """Max batch size for the current mode."""
        if self.mode == NodeMode.MERGED:
            return MERGED_CAPACITY
        return UNMERGED_CAPACITY

    def _can_admit(self, req: SimRequest) -> bool:
        """Check whether a request can be admitted to the running batch."""
        if self.mode == NodeMode.MERGED:
            # Merged mode: only accept requests for the merged adapter
            if req.adapter_id != self.current_merged_adapter:
                return False
            return len(self.running_requests) < MERGED_CAPACITY

        # Unmerged mode: cost = running_requests + unique_active_loras <= capacity
        current_loras = self.get_active_loras()
        new_lora_count = len(current_loras | {req.adapter_id})
        return (len(self.running_requests) + new_lora_count) <= UNMERGED_CAPACITY

    def _multiplier(self) -> float:
        """Speed multiplier based on current mode."""
        if self.mode == NodeMode.MERGED:
            return MERGE_SPEED_MULTIPLIER
        return 1.0

    # ------------------------------------------------------------------
    # GPU slot management
    # ------------------------------------------------------------------

    def _adapter_on_gpu(self, adapter_id: str) -> bool:
        return adapter_id in self.adapter_to_slot

    def _ensure_cpu_loaded(self, adapter_id: str) -> None:
        """Ensure the adapter is in the CPU LRU cache."""
        if adapter_id in self.cpu_cache:
            # Move to end (most recently used)
            self.cpu_cache.move_to_end(adapter_id)
            return
        # Evict from CPU cache if full
        while len(self.cpu_cache) >= MAX_CPU_LORAS:
            evicted, _ = self.cpu_cache.popitem(last=False)
            # If the evicted adapter is also on GPU, leave GPU slot alone
            # (GPU slots are managed separately)
        self.cpu_cache[adapter_id] = True

    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int) -> None:
        """Move an adapter from CPU cache to a GPU slot."""
        self._ensure_cpu_loaded(adapter_id)
        # Clear any previous occupant of this slot
        for aid, sid in list(self.adapter_to_slot.items()):
            if sid == slot_id:
                del self.adapter_to_slot[aid]
                break
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id

    def _find_free_slot(self) -> Optional[int]:
        """Find a free GPU slot, or None if all occupied."""
        for slot_id in range(UNMERGED_CAPACITY):
            if slot_id not in self.gpu_slots:
                return slot_id
        return None

    def _evict_lru_slot(self) -> int:
        """Evict the least-recently-used GPU slot and return its slot_id."""
        # Active adapters should not be evicted
        active = self.get_active_loras()
        for slot_id in range(UNMERGED_CAPACITY):
            adapter = self.gpu_slots.get(slot_id)
            if adapter is not None and adapter not in active:
                del self.adapter_to_slot[adapter]
                del self.gpu_slots[slot_id]
                return slot_id
        # Fallback: evict slot 0 if everything is active (shouldn't happen)
        slot_id = 0
        adapter = self.gpu_slots.get(slot_id)
        if adapter is not None:
            del self.adapter_to_slot[adapter]
            del self.gpu_slots[slot_id]
        return slot_id

    def _allocate_gpu_slot(self, adapter_id: str) -> int:
        """Get or allocate a GPU slot for the adapter."""
        if adapter_id in self.adapter_to_slot:
            return self.adapter_to_slot[adapter_id]
        slot_id = self._find_free_slot()
        if slot_id is None:
            slot_id = self._evict_lru_slot()
        self._load_adapter_to_slot(adapter_id, slot_id)
        return slot_id

    # ------------------------------------------------------------------
    # Phase: IDLE
    # ------------------------------------------------------------------

    def _step_idle(self) -> bool:
        # Clean up finished requests from running batch
        self.running_requests = [
            r for r in self.running_requests if not r.is_finished
        ]

        # Nothing to do?
        if not self.running_requests and not self.request_queue:
            return False

        # Admit eligible requests from queue into running batch
        newly_admitted: List[SimRequest] = []
        remaining_queue: List[SimRequest] = []
        for req in self.request_queue:
            if self._can_admit(req):
                self.running_requests.append(req)
                newly_admitted.append(req)
                if req.assigned_time_ms is None:
                    req.assigned_time_ms = self.clock.now_ms
            else:
                remaining_queue.append(req)
        self.request_queue = remaining_queue

        # Identify adapters that need loading to GPU
        self._pending_loads = []
        for req in self.running_requests:
            if not self._adapter_on_gpu(req.adapter_id):
                if req.adapter_id not in self._pending_loads:
                    self._pending_loads.append(req.adapter_id)

        # Determine the set of new requests that need prefill
        new_reqs = [r for r in self.running_requests if r.needs_prefill]
        self._current_batch = list(self.running_requests)

        # Transition to next phase
        if self._pending_loads:
            total_load_ms = len(self._pending_loads) * _LOAD_DELAY_MS
            self._phase = EnginePhase.LOADING
            self._phase_remaining_ms = total_load_ms
        elif new_reqs:
            prefill_ms = math.ceil(
                _PREFILL_BASE_MS * len(new_reqs) * self._multiplier()
            )
            self._phase = EnginePhase.PREFILL
            self._phase_remaining_ms = prefill_ms
        elif self.running_requests:
            batch_size = len(self.running_requests)
            decode_ms = math.ceil(
                (_DECODE_BASE_MS + _DECODE_SLOPE_MS * batch_size)
                * self._multiplier()
            )
            self._phase = EnginePhase.DECODE
            self._phase_remaining_ms = decode_ms
        else:
            # Only queued requests that couldn't be admitted; stay idle
            return False

        return True

    # ------------------------------------------------------------------
    # Phase: LOADING
    # ------------------------------------------------------------------

    def _step_loading(self) -> bool:
        self._phase_remaining_ms -= 1
        if self._phase_remaining_ms > 0:
            return True

        # Loading complete: place adapters on GPU
        for adapter_id in self._pending_loads:
            self._ensure_cpu_loaded(adapter_id)
            self._allocate_gpu_slot(adapter_id)
        self._pending_loads.clear()

        # Determine next phase
        new_reqs = [r for r in self.running_requests if r.needs_prefill]
        if new_reqs:
            prefill_ms = math.ceil(
                _PREFILL_BASE_MS * len(new_reqs) * self._multiplier()
            )
            self._phase = EnginePhase.PREFILL
            self._phase_remaining_ms = prefill_ms
        elif self.running_requests:
            batch_size = len(self.running_requests)
            decode_ms = math.ceil(
                (_DECODE_BASE_MS + _DECODE_SLOPE_MS * batch_size)
                * self._multiplier()
            )
            self._phase = EnginePhase.DECODE
            self._phase_remaining_ms = decode_ms
        else:
            self._phase = EnginePhase.IDLE
            self._phase_remaining_ms = 0

        return True

    # ------------------------------------------------------------------
    # Phase: PREFILL
    # ------------------------------------------------------------------

    def _step_prefill(self) -> bool:
        self._phase_remaining_ms -= 1
        if self._phase_remaining_ms > 0:
            return True

        # Prefill complete: mark requests and generate first token
        for req in self.running_requests:
            if req.needs_prefill:
                req.needs_prefill = False
                req.past_key_values = True
                req.tokens_generated = 1
                if req.first_token_time_ms is None:
                    req.first_token_time_ms = self.clock.now_ms
                if self.on_token is not None:
                    self.on_token(req, req.tokens_generated)
                # Check if done (unlikely after 1 token, but handle it)
                if req.tokens_generated >= req.max_new_tokens:
                    req.is_finished = True
                    req.finish_time_ms = self.clock.now_ms
                    if self.on_finish is not None:
                        self.on_finish(req)

        # Transition to DECODE for remaining running requests
        active = [r for r in self.running_requests if not r.is_finished]
        if active:
            batch_size = len(active)
            decode_ms = math.ceil(
                (_DECODE_BASE_MS + _DECODE_SLOPE_MS * batch_size)
                * self._multiplier()
            )
            self._phase = EnginePhase.DECODE
            self._phase_remaining_ms = decode_ms
        else:
            self._phase = EnginePhase.IDLE
            self._phase_remaining_ms = 0

        return True

    # ------------------------------------------------------------------
    # Phase: DECODE
    # ------------------------------------------------------------------

    def _step_decode(self) -> bool:
        self._phase_remaining_ms -= 1
        if self._phase_remaining_ms > 0:
            return True

        # Decode complete: each running request generates 1 token
        for req in self.running_requests:
            if req.is_finished:
                continue
            req.tokens_generated += 1
            if self.on_token is not None:
                self.on_token(req, req.tokens_generated)
            if req.tokens_generated >= req.max_new_tokens:
                req.is_finished = True
                req.finish_time_ms = self.clock.now_ms
                if self.on_finish is not None:
                    self.on_finish(req)

        # Back to IDLE to re-evaluate queue and start next round
        self._phase = EnginePhase.IDLE
        self._phase_remaining_ms = 0
        return True

"""Shared data types for discrete-time simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable, Any


class NodeMode(Enum):
    MERGED = "merge"
    UNMERGED = "unmerge"


class NodeStatus(Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    DRAINING = "draining"


class EnginePhase(Enum):
    IDLE = "idle"
    DISPATCH = "dispatch"
    LOADING = "loading"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class SimRequest:
    request_id: str
    adapter_id: str
    original_adapter_id: str  # before semantic substitution
    cluster: str
    prompt: str = "test"
    max_new_tokens: int = 256
    arrival_time_ms: int = 0
    # Tracking
    assigned_time_ms: Optional[int] = None
    first_token_time_ms: Optional[int] = None
    finish_time_ms: Optional[int] = None
    tokens_generated: int = 0
    assigned_node: Optional[str] = None
    assigned_cluster: Optional[str] = None
    drop_reason: Optional[str] = None
    is_dropped: bool = False
    is_finished: bool = False
    is_delegated: bool = False
    # Engine internal
    needs_prefill: bool = True
    prefill_remaining_ms: int = 0
    load_remaining_ms: int = 0
    past_key_values: bool = False  # True after prefill

    @property
    def ttft_ms(self) -> Optional[int]:
        if self.first_token_time_ms is not None:
            return self.first_token_time_ms - self.arrival_time_ms
        return None

    @property
    def ttft_s(self) -> Optional[float]:
        t = self.ttft_ms
        return t / 1000.0 if t is not None else None

    @property
    def total_time_s(self) -> Optional[float]:
        if self.finish_time_ms is not None:
            return (self.finish_time_ms - self.arrival_time_ms) / 1000.0
        return None


@dataclass
class ExperimentDef:
    """Definition for one of the 6 experiment configurations."""
    experiment_id: int
    efo_type: str          # "sp1", "lru", "dlora"
    control_type: str      # "sp2", "random", "lru", "dlora"
    metadata_file: str     # "lora_metadata.json" or "lora_metadata_without_substitutes.json"
    disk_capacity_gb: float = 5.0
    dispatch_strategy: str = "lyapunov"  # lyapunov, random, greedy


EXPERIMENT_CONFIGS = {
    1: ExperimentDef(1, "sp1", "sp2", "lora_metadata.json", 5, "lyapunov"),
    2: ExperimentDef(2, "sp1", "sp2", "lora_metadata_without_substitutes.json", 5, "lyapunov"),
    3: ExperimentDef(3, "sp1", "random", "lora_metadata_without_substitutes.json", 5, "random"),
    4: ExperimentDef(4, "lru", "lru", "lora_metadata_without_substitutes.json", 5, "random"),
    5: ExperimentDef(5, "dlora", "dlora", "lora_metadata_without_substitutes.json", 5, "greedy"),
    6: ExperimentDef(6, "lru", "lru", "lora_metadata_without_substitutes.json", 5, "greedy"),
}


@dataclass
class SimulationConfig:
    experiment_id: int
    cluster_topology: Dict[str, int]  # {cluster_name: num_compute_nodes}
    start_offset: int = 172800        # CSV trace start offset (seconds)
    duration_hours: int = 8
    target_clusters: Optional[List[str]] = None
    disk_capacity_gb: Optional[float] = None  # override from experiment
    dispatch_strategy: Optional[str] = None   # override from experiment
    seed: int = 42
    output_dir: str = "./results/"
    trace_csv: str = "./information/simulation_data.csv"
    metadata_dir: str = "./information/"

    def get_experiment_def(self) -> ExperimentDef:
        return EXPERIMENT_CONFIGS[self.experiment_id]

    def get_target_clusters(self) -> List[str]:
        if self.target_clusters:
            return self.target_clusters
        return list(self.cluster_topology.keys())

    def get_disk_capacity_gb(self) -> float:
        if self.disk_capacity_gb is not None:
            return self.disk_capacity_gb
        return self.get_experiment_def().disk_capacity_gb

    def get_dispatch_strategy(self) -> str:
        if self.dispatch_strategy is not None:
            return self.dispatch_strategy
        return self.get_experiment_def().dispatch_strategy

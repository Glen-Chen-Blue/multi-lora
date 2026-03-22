import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict


class SimTraceReader:
    """Reads a CSV trace file and provides requests at correct simulation times.

    The CSV is expected to have columns: cluster, lora_id, arrive_timestamp
    where arrive_timestamp is in seconds.
    """

    def __init__(
        self,
        csv_path: str,
        start_offset_s: int,
        duration_s: int,
        target_clusters: List[str],
    ):
        df = pd.read_csv(csv_path)

        # Filter by target clusters
        df = df[df["cluster"].isin(target_clusters)].copy()

        # Filter by start_offset: keep rows at or after the offset
        df = df[df["arrive_timestamp"] >= start_offset_s].copy()

        # Normalize arrival times to 0
        df["arrive_timestamp"] = df["arrive_timestamp"] - start_offset_s

        # Filter by duration: keep rows within the window
        df = df[df["arrive_timestamp"] <= duration_s].copy()

        # Sort by arrival time
        df = df.sort_values("arrive_timestamp").reset_index(drop=True)

        # Convert arrival times to milliseconds (int)
        df["arrive_ms"] = (df["arrive_timestamp"] * 1000).astype(int)

        # Build lookup: time_ms -> list of (cluster, lora_id)
        self._events: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        for _, row in df.iterrows():
            self._events[int(row["arrive_ms"])].append(
                (str(row["cluster"]), int(row["lora_id"]))
            )

        self._total_requests: int = len(df)
        self._max_time_ms: int = int(df["arrive_ms"].max()) if len(df) > 0 else 0

    def get_requests_at(self, time_ms: int) -> List[Tuple[str, int]]:
        """Return list of (cluster, lora_id) for requests arriving at time_ms."""
        return self._events.get(time_ms, [])

    @property
    def total_requests(self) -> int:
        """Total number of requests in the filtered trace."""
        return self._total_requests

    @property
    def max_time_ms(self) -> int:
        """Maximum arrival time in milliseconds across all filtered requests."""
        return self._max_time_ms

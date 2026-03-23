import os
import json


class SimLogger:
    """JSONL logger that outputs efo_global_metrics.log in the format cost2.py expects."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "efo_global_metrics.log")
        self._fh = open(self._log_path, "w")

    def log_global_metrics(
        self,
        timestamp_s: float,
        step_id: int,
        sub_step: int,
        cluster_data: dict,
        efo_totals: dict,
    ) -> None:
        """Write one JSONL line matching the exact schema cost2.py parses.

        Parameters
        ----------
        timestamp_s : float
            Simulation wall-clock time in seconds.
        step_id : int
            Discrete simulation step number.
        sub_step : int
            Sub-step within the current step.
        cluster_data : dict
            Per-cluster metrics snapshot (opaque to the logger, stored as-is).
        efo_totals : dict
            Aggregated totals dict.  Must contain the keys that cost2.py reads:
              total_stored_loras, artifact_downloads, total_inference_time,
              total_offloads, total_drops, total_local_completed,
              total_offload_completed, total_requests.
        """
        entry = {
            "timestamp": timestamp_s,
            "step_id": step_id,
            "sub_step": sub_step,
            "cluster_data": cluster_data,
            "efo_totals": {
                "total_stored_loras": int(efo_totals.get("total_stored_loras", 0)),
                "artifact_downloads": int(efo_totals.get("artifact_downloads", 0)),
                "total_inference_time": float(efo_totals.get("total_inference_time", 0.0)),
                "total_offloads": int(efo_totals.get("total_offloads", 0)),
                "total_drops": int(efo_totals.get("total_drops", 0)),
                "total_local_completed": int(efo_totals.get("total_local_completed", 0)),
                "total_offload_completed": int(efo_totals.get("total_offload_completed", 0)),
                "total_requests": int(efo_totals.get("total_requests", 0)),
            },
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self):
        """Close the underlying file handle."""
        if self._fh and not self._fh.closed:
            self._fh.close()

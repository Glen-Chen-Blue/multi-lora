"""Network delay simulation (shifted lognormal)."""

import math
import random as _random
from typing import Dict, Tuple

# Default params from config.py
DEFAULT_NETWORK_PARAMS = {
    ("cluster_1", "cluster_2"): (20, 4.0, 0.5),
    ("cluster_2", "cluster_3"): (40, 5.0, 1.0),
    ("cluster_1", "cluster_3"): (60, 6.0, 1.1),
}


class SimNetwork:
    """Deterministic network delay model using seeded RNG."""

    def __init__(self, seed: int = 42, params: Dict[Tuple[str, str], Tuple[float, float, float]] = None):
        self._rng = _random.Random(seed)
        self._params = params or DEFAULT_NETWORK_PARAMS
        self._matrix: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for (c1, c2), vals in self._params.items():
            self._matrix[(c1, c2)] = vals
            self._matrix[(c2, c1)] = vals

    def get_delay_ms(self, src: str, dest: str) -> float:
        """Returns a shifted lognormal delay in ms."""
        if src == dest:
            return 0.0
        if (src, dest) not in self._matrix:
            return 50.0  # default
        d_prop, mu, sigma = self._matrix[(src, dest)]
        jitter = math.exp(self._rng.gauss(mu, sigma))
        return d_prop + jitter

    def get_p95_info(self, cluster_names) -> Dict[str, Dict[str, float]]:
        """Compute P95 delay matrix for routing."""
        p95_delays = {}
        for c1 in cluster_names:
            p95_delays[c1] = {}
            for c2 in cluster_names:
                if c1 == c2:
                    p95_delays[c1][c2] = 0.0
                else:
                    if (c1, c2) in self._matrix:
                        d_prop, mu, sigma = self._matrix[(c1, c2)]
                        p95_jitter = math.exp(mu + 1.645 * sigma)
                        p95_delays[c1][c2] = round(d_prop + p95_jitter, 2)
                    else:
                        p95_delays[c1][c2] = 100.0
        return p95_delays

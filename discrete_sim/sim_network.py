"""Network delay simulation (shifted lognormal)."""

import math
import random as _random
from typing import Dict, Tuple

def generate_network_params(max_clusters: int = 10) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """動態生成支援到 max_clusters 的全連通網路延遲矩陣"""
    params = {}
    # 固定 generator seed 確保每次跑不同演算法 (S-LoRA, dLoRA, Ours...) 時，網路拓撲(延遲)皆相同
    rng = _random.Random(42) 
    
    for i in range(1, max_clusters + 1):
        for j in range(i + 1, max_clusters + 1):
            c1 = f"cluster_{i}"
            c2 = f"cluster_{j}"
            
            # 保留原本 1~3 號 cluster 的設定，向下相容舊實驗
            if (c1, c2) == ("cluster_1", "cluster_2"):
                vals = (20.0, 4.0, 0.5)
            elif (c1, c2) == ("cluster_2", "cluster_3"):
                vals = (40.0, 5.0, 1.0)
            elif (c1, c2) == ("cluster_1", "cluster_3"):
                vals = (60.0, 6.0, 1.1)
            else:
                # 動態生成其他新增叢集間的延遲 (模擬 20ms 到 80ms 的基礎廣域網延遲)
                d_prop = rng.uniform(20.0, 80.0) 
                mu = rng.uniform(3.0, 5.0)       
                sigma = rng.uniform(0.5, 1.1)    
                vals = (round(d_prop, 1), round(mu, 2), round(sigma, 2))
            
            params[(c1, c2)] = vals
    return params

# 預設生成 10 個叢集的網路參數
DEFAULT_NETWORK_PARAMS = generate_network_params(10)


class SimNetwork:
    """Deterministic network delay model using seeded RNG."""

    def __init__(self, seed: int = 42, params: Dict[Tuple[str, str], Tuple[float, float, float]] = None):
        self._rng = _random.Random(seed)
        self._params = params or DEFAULT_NETWORK_PARAMS
        self._matrix: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        # 建立雙向對稱延遲矩陣
        for (c1, c2), vals in self._params.items():
            self._matrix[(c1, c2)] = vals
            self._matrix[(c2, c1)] = vals

    def get_delay_ms(self, src: str, dest: str) -> float:
        """Returns a shifted lognormal delay in ms."""
        if src == dest:
            return 0.0
        if (src, dest) not in self._matrix:
            return 50.0  # 萬一超過 10 個叢集，提供安全 fallback
        
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
                        # Lognormal P95 近似公式
                        p95_jitter = math.exp(mu + 1.645 * sigma)
                        p95_delays[c1][c2] = round(d_prop + p95_jitter, 2)
                    else:
                        p95_delays[c1][c2] = 100.0 # 萬一超過 10 個叢集，提供安全 fallback
        return p95_delays
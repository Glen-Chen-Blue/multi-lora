import json
import random
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

class SimSyntheticGenerator:
    """Generates synthetic requests using Poisson arrivals and Zipf LoRA selection.
    
    This class serves as a drop-in replacement for SimTraceReader.
    """

    def __init__(
        self,
        lora_mapping_path: str,
        duration_s: int,
        target_clusters: List[str],
        rps_per_cluster: float,
        zipf_s: float = 1.2,
        seed: int = 42,
        **kwargs  # 吸收原先 SimTraceReader 可能傳入的 csv_path, start_offset_s 等參數
    ):
        """
        Args:
            lora_mapping_path: lora_mapping.json 的檔案路徑
            duration_s: 模擬總時長 (秒)
            target_clusters: 參與模擬的 cluster 列表 (例如 ['cluster_1', 'cluster_2', 'cluster_3'])
            rps_per_cluster: 每個 cluster 各自的每秒請求數 (Poisson lambda)
            zipf_s: Zipf 分佈的傾斜參數 (預設 1.2，越大頭部越集中)
            seed: 亂數種子，確保實驗可重現
        """
        random.seed(seed)
        np.random.seed(seed)

        # 讀取 lora_mapping.json
        with open(lora_mapping_path, 'r') as f:
            lora_mapping = json.load(f)

        self._events: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self._total_requests: int = 0
        max_time = 0

        # 為每一個目標 Cluster 獨立生成 Request 軌跡
        for cluster in target_clusters:
            if cluster not in lora_mapping:
                print(f"[Warning] {cluster} not found in lora_mapping.json. Skipping.")
                continue

            # 1. 解析並排序 LoRA ID
            # 將 {"LoRA_71": "1", ...} 轉換成 (71, 1) 的 Tuple 列表並依排名排序
            cluster_loras = []
            for lora_name, rank_str in lora_mapping[cluster].items():
                lora_id = int(lora_name.replace("LoRA_", ""))
                rank = int(rank_str)
                cluster_loras.append((lora_id, rank))
            
            # 依照 rank 由小到大排序 (rank=1 最前面)
            cluster_loras.sort(key=lambda x: x[1])
            ordered_lora_ids = [x[0] for x in cluster_loras]
            num_loras = len(ordered_lora_ids)

            # 2. 建立此 Cluster 專屬的 Zipf 機率分佈表
            # 公式: P(k) = (1/k^s) / sum(1/i^s)
            ranks = np.arange(1, num_loras + 1)
            weights = 1.0 / (ranks ** zipf_s)
            probabilities = weights / weights.sum()

            # 3. 使用 Poisson 分佈 (Exponential inter-arrival) 獨立生成請求時間軸
            current_time_s = 0.0
            while True:
                # 取得下一個 request 的間隔時間
                inter_arrival = random.expovariate(rps_per_cluster)
                current_time_s += inter_arrival

                # 如果超過模擬時間，就停止這個 cluster 的生成
                if current_time_s > duration_s:
                    break

                # 根據 Zipf 機率表抽樣 LoRA ID
                chosen_lora = np.random.choice(ordered_lora_ids, p=probabilities)

                # 轉換為毫秒並存入 events 字典中
                time_ms = int(current_time_s * 1000)
                self._events[time_ms].append((cluster, int(chosen_lora)))
                
                self._total_requests += 1
                if time_ms > max_time:
                    max_time = time_ms

        self._max_time_ms = max_time

    def get_requests_at(self, time_ms: int) -> List[Tuple[str, int]]:
        """Return list of (cluster, lora_id) for requests arriving at time_ms."""
        return self._events.get(time_ms, [])

    @property
    def total_requests(self) -> int:
        """Total number of requests generated."""
        return self._total_requests

    @property
    def max_time_ms(self) -> int:
        """Maximum arrival time in milliseconds across all generated requests."""
        return self._max_time_ms
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert generated events to a DataFrame compatible with EFO forecasting."""
        import pandas as pd
        records = []
        for t_ms, reqs in self._events.items():
            arr_sec = t_ms / 1000.0
            for cluster, lid in reqs:
                records.append({
                    "arrival_sec": arr_sec,
                    "cluster": cluster,
                    # EFO 預設會去找 "lora_id" 欄位，它可以是整數或字串
                    "lora_id": lid 
                })
        return pd.DataFrame(records)
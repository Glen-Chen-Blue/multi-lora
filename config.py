# ============================================================
# config.py - 全域常數與系統設定檔
# ============================================================

import os

# ------------------------------------------------------------
# 1. 系統環境與路徑設定 (Paths & Environment)
# ------------------------------------------------------------
LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
LORA_METADATA_PATH = os.environ.get("LORA_METADATA_PATH", "./information/lora_metadata.json")
LORA_MAPPING_PATH = os.environ.get("LORA_MAPPING", "./information/lora_mapping.json")
LORA_HOURLY_COUNTS_PATH = os.environ.get("LORA_HOURLY_COUNTS", "./information/lora_hourly_counts.json")
SIMULATION_DATA_CSV_PATH = os.environ.get("SIMULATION_DATA_CSV", "./information/simulation_data.csv") # [新增] CSV 檔案路徑
LOG_PATH = os.environ.get("LOG_PATH", "./logs")

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Meta-Llama-3.1-8B")

# ------------------------------------------------------------
# 2. 系統容量與長度限制 (Capacity & Token Limits)
# ------------------------------------------------------------
FIXED_INPUT_LEN = 512        # 標準化輸入長度 (Prompt tokens)
FIXED_OUTPUT_LEN = 256       # 標準化輸出長度 (Generated tokens)

MERGED_CAPACITY = 12         # Merged 模式 (Dedicated) 的最大 Batch Size
UNMERGED_CAPACITY = 10       # Unmerged 模式 (Shared) 的基礎 Batch Size
MAX_CPU_LORAS = 30           # Host Memory (CPU RAM) 最多可快取的 LoRA 數量

# ------------------------------------------------------------
# 3. 模擬引擎延遲與速度常數 (Simulation Engine Latencies)
# ------------------------------------------------------------
SIM_LOAD_DELAY = 0.066           # 從 Disk 載入到 Host Memory 的 I/O 延遲 (秒)
SIM_PREFILL_BASE_TIME = 0.065    # Prefill 階段處理單一請求的基礎時間 (秒)
SIM_DECODE_BASE_TIME = 0.025     # Decode 階段產生單一 Token 的基礎時間 (秒)
SIM_DECODE_SLOPE = 0.0010        # Decode 階段受 Batch Size 影響的斜率 (增加的干擾時間)
MERGE_SPEED_MULTIPLIER = 0.861   # 當處於 Merged 模式時的運算加速/時間折扣乘數

# [衍生常數] 供 Control Node 預估 TTFT 使用
SCHEDULER_OVERHEAD = 0.010       # Control Node 排程與派發的基礎開銷

# ------------------------------------------------------------
# 4. Control Node - Lyapunov、QoE 與自動擴縮容超參數
# ------------------------------------------------------------
T_MAX = 6.0                  # Control Node 視角的 SLO 最大首字延遲限制 (秒)
EPSILON = 0.05               # Lyapunov 虛擬佇列扣除常數 (漂移參數)
PSI_DROP = 10.0              # 容量耗盡被迫丟棄請求的巨大懲罰權重 (Z_debt)

HTTP_MAX_CONNECTIONS = 500         # 控制節點與運算節點之間的 HTTP 連線池上限 (應對高併發)
SCALE_UP_DROP_THRESHOLD = 2        # 觸發擴容的近期 Drop 數量閾值 (調低以加速反應)
SCALE_DOWN_SURPLUS_THRESHOLD = 10  # 觸發縮容的閒置 Slot 數量閾值 (一台基準容量12)
EDGE_SYNC_TIMEOUT = 300.0          # [新增] 等待節點排空與重置的 HTTP 請求逾時限制 (秒)

# ------------------------------------------------------------
# 5. EFO Server - SP1 全域最佳化與預測模型常數
# ------------------------------------------------------------
# 成本權重設定 (抽象單位 Credit)
COST_STORE_PER_GB = 0.005     # kappa_store: 1GB 模型本地存放 1 個時隙的成本
COST_DOWNLOAD_PER_GB = 3   # kappa_inter: 跨區下載 1GB 模型的頻寬成本
COST_INST_LOCAL = 0.001      # kappa_inst: 本地處理 1 個 Request 的算力成本
COST_NET_TRAFFIC = 0.001     # kappa_net: 跨 Cluster 處理 1 個 Request 的流量成本
COST_DROP_PENALTY = 0.01*10      # SP1 視角的 Drop 懲罰 (對應 Psi_drop)
COST_DROP_PENALTY2 = 0.01*1      # SP1 視角的 Drop 懲罰 (對應 Psi_drop)
COST_COMPUTE_PER_SEC = 0.001 # [新增] kappa_compute: 每秒的算力成本 (用於預測模型的成本估計)

# 物理容量與限制
LORA_SIZE_GB = 0.1           # S_lora: 單一 LoRA Adapter 的檔案大小估算 (GB)
DISK_CAPACITY_GB = 5.0       # 每個 Cluster 硬碟的 LoRA 儲存容量上限 (GB)
T_MAX_SLO = 6.0              # EFO 全域視角的 SLO 承諾最大端到端延遲 (秒)
SWAP_EPSILON = 0           # 演算法微調常數：新模型多帶來的淨效用必須大於此門檻才允許替換 (防震盪)

# 時序與 LSTM 預測模型常數
T_TOTAL_HOURS = 336          # 總模擬時間週期 (例如 14 天)
SEQ_LENGTH = 48              # LSTM 歷史觀測序列長度

# ------------------------------------------------------------
# 6. EFO Server - 網路拓撲延遲模擬 (Network Simulator)
# ------------------------------------------------------------
# 格式: (cluster_a, cluster_b): (d_prop, mu, sigma)
# d_prop: 基礎傳輸延遲, mu/sigma: Lognormal 抖動參數
NETWORK_SIM_PARAMS = {
    ("cluster_1", "cluster_2"): (20, 4.0, 0.5),  # Cloud to Near Edge (光纖/5G MEC)
    ("cluster_2", "cluster_3"): (40, 5.0, 1.0),  # Edge to Edge (跨區 WAN/微波通訊)
    ("cluster_1", "cluster_3"): (60, 6.0, 1.1),  # Cloud to Remote Edge (受限頻寬/4G IoT邊緣)
}


SP1_INTERVAL_SECONDS = 3600          # SP1 全局優化的執行間隔 (秒)
SP2_INTERVAL_SECONDS = 3             # SP2 局部優化的執行間隔 (秒)
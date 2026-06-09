# Ablation Experiments — README

新增了兩個實驗腳本，**不修改任何既有檔案**。

## run_ablation.py  （Figure 1：Component Ablation Bar Chart）

### 執行方式
```bash
cd /home/glenchen/multi-lora
/home/glenchen/miniconda3/envs/myenv/bin/python discrete_sim/run_ablation.py
```

### 五個 Variant（X 軸）
| Variant | 說明 | 關閉的 Knob |
|---|---|---|
| `ours` | Full System（動態 auto-scaling） | — |
| `no_semantic` | 無語意替換 | K1 |
| `no_provision` | 無預測性 LoRA 部署（SP1 noop） | K2 |
| `no_merge` | 強制 always-unmerge | K3 |
| `no_autoscale` | 固定全開（原 sim_control_node.py 行為） | K4 |

### 三個指標（三個 Subplot）
| Subplot | 指標 | 定義 |
|---|---|---|
| (a) | Max Stable Throughput | 最高 RPS 且 P95 TTFT ≤ 6.0s |
| (b) | Network Cost / Request | `(downloads×3.0 + offloads×0.001) / total_req`，正規化到 Ours=1.0 |
| (c) | GPU Compute Time / Request | `total_inference_time_ms / total_req` (ms/req) |

圖表底部另標注各 variant 的 **GPU Node-Active Hours**（K4 資源消耗）。

### 輸出
- `discrete_sim/results/ablation_results.json` — 數值結果
- `discrete_sim/results/ablation_bar.png` — Bar Chart 圖表

---

## run_tau_sweep.py  （Figure 2：τ_sim Sensitivity Line Chart）

### 執行方式
```bash
cd /home/glenchen/multi-lora
/home/glenchen/miniconda3/envs/myenv/bin/python discrete_sim/run_tau_sweep.py
```

### 七個閾值點（X 軸）
| DISTANCE_THRESHOLD | 近似 τ_sim | 說明 |
|---|---|---|
| 0.00 | 1.000 | 無替換（等同 -Semantic） |
| 0.04 | 0.999 | 極少替換 |
| 0.07 | 0.998 | 少量替換 |
| **0.10** | **0.995** | **論文預設** |
| 0.15 | 0.989 | 中高替換 |
| 0.20 | 0.980 | 高替換 |
| 0.30 | 0.955 | 很高替換 |

### 兩個指標（雙 Y 軸折線）
| Y 軸 | 指標 |
|---|---|
| 左（橘線） | Semantic Substitution Rate（%） |
| 右（藍線） | Network Download Cost / Request |

### 輸出
- `discrete_sim/results/tau_sweep_results.json` — 數值結果
- `discrete_sim/results/tau_sweep.png` — 雙線圖

---

## 技術架構

兩個腳本都使用 **subprocess + monkey-patch** 架構（仿照現有的 `run_lyapunov_variable.py`）：
- 每個 `(variant, rps)` 或 `threshold` 在獨立 subprocess 中執行，完全隔離
- 透過 `--worker` flag 啟動 worker 模式
- 結果透過 `RESULT_JSON:` 前綴的 stdout 傳回 master process
- 並行度：`run_ablation.py` 最多 12 個並行，`run_tau_sweep.py` 最多 8 個

所有原有腳本（`run_experiments.py`、`run_max_throughput.py` 等）**完全不受影響**。

# Multi-LoRA Orchestration Simulation - Conversation Summary

本文件統整了當前對 **「語意感知多 LoRA 邊緣聯邦編排系統」**（Semantic-Aware Multi-LoRA Orchestration in Edge Federation）離散事件模擬器 (`discrete_sim`) 進行的所有修改與實驗進度，方便您在下次對話中直接交給 AI 讀取，無縫接軌繼續執行。

---

## 1. 核心需求與設計邏輯

教授希望對系統進行 **自我消融實驗 (Self-Ablation Study)** 與 **語意相似度閾值敏感度實驗 ($\tau_{sim}$ Sweep)**，以展示各項機制（Knob）對系統的貢獻度與實質影響。

### 四個控制 Knob
* **K1 (Semantic Substitution)**：是否啟用語意相似度替換。
* **K2 (Predictive Provisioning)**：EFO (SP1) 是否執行預測性部署。
* **K3 (Dynamic Merge/Unmerge)**：是否允許動態切換合併模式（影響批次大小 $B_{max}$）。
* **K4 (Auto-scaling)**：是否動態調整開關 GPU 節點（李雅普諾夫決策）。

---

## 2. 已實作並驗證的檔案

我們**完全沒有修改任何原有模擬器的檔案**（保持 `run_experiments.py` 等舊實驗正常運作），僅新建了以下三個檔案於 `discrete_sim/` 目錄中：

### A. 消融實驗腳本：[`run_ablation.py`](file:///home/glenchen/multi-lora/discrete_sim/run_ablation.py)
* **X 軸：5 個 Variant**
  1. `ours`：完整系統（K1+K2+K3+K4 全開）。
  2. `no_semantic` (K1 關閉)：關閉語意替換。
  3. `no_provision` (K2 關閉)：關閉預載，改為 on-demand 載入。
  4. `no_merge` (K3 關閉)：強制始終為 unmerge 狀態。
  5. `no_autoscale` (K4 關閉)：所有計算節點固定全開（不待機）。
* **Y 軸：3 個指標（產出 Subplots）**
  * **(a) Max Stable Throughput**：在 P95 TTFT $\le 6.0\text{ s}$ 的 SLO 下，系統所能承受的最大總 RPS。
  * **(b) Network Cost / Request**：正規化至 Ours=1.0。
  * **(c) GPU Compute Time / Request**：推理開銷。
  * *附註：圖表下方標註各 Variant 的 **GPU Node-Active Hours**（代表實際運作消耗的 GPU 總時間，非僅開關狀態）。*

### B. 敏感度實驗腳本：[`run_tau_sweep.py`](file:///home/glenchen/multi-lora/discrete_sim/run_tau_sweep.py)
* **X 軸：12 個 $\tau_{sim}$ 門檻點**
  * 平均分散：`1.00`, `0.95`, `0.90`, `0.85`, `0.80`, `0.75`, `0.70`, `0.65`, `0.60`, `0.55`, `0.50`
  * 系統預設：`0.995` (對應原本的歐氏距離 `0.10`)
* **設計要點 (嚴謹控制單一變數與快取壓力)**：
  1. **固定預載關係**：SP1 的預測性部署 (Provisioning) 固定使用 $d=0.10$ ($\tau_{sim}\approx 0.995$)。只有在 Scheduler 派遣 (Dispatch) 替代決策時採用變動的 $\tau_{sim}$。這避免了 SP1 選取不同 LoRA 集合造成的變數混淆。
  2. **快取壓力限制**：限制 `disk_capacity_gb = 4.0` (約 40 個 LoRA slots)。在 70 個 global LoRA 的情況下，保證有 30 個 LoRA 無法被預載，強迫系統在低 $\tau_{sim}$ 下使用語意替換，或在 $\tau_{sim}$ 高時承受 download cost。
* **Y 軸：雙 Y 軸**
  * 左軸：語意替換率 (Semantic Substitution Rate %)
  * 右軸：Network Cost/Req 以及 P95 TTFT (s)

### C. 實驗說明文件：[`README_ablation.md`](file:///home/glenchen/multi-lora/discrete_sim/README_ablation.md)
* 簡述了以上兩個腳本的 Variant 參數、指標計算與技術架構（均採用並行 Subprocess + Monkey-patch 運行，安全乾淨）。

---

## 3. 當前執行進度與狀態

1. **語法與環境驗證**：已確認新腳本在虛擬環境 `/home/glenchen/miniconda3/envs/myenv/bin/python` 中無語法錯誤。
2. **Smoke Test 驗證**：
   * `run_ablation.py` 與 `run_tau_sweep.py` 的單一 Worker 模擬測試均已通過，能正常模擬 4 小時的流量並正確回傳以下指標：
     * `p95_ttft`
     * `network_cost_per_req`
     * `gpu_compute_per_req`
     * `gpu_active_node_ms` (GPU 實際運作毫秒數，已精確測量)
3. **待跑完整模擬**：因前次 quota 限制中斷，目前尚未運行完整的 Multi-process 實驗數據收集。

---

## 4. 下次繼續時的具體操作指令

請在啟用 `myenv` 環境下，於專案根目錄 `/home/glenchen/multi-lora` 執行以下操作：

### 步驟 1：執行消融實驗 (Ablation Study)
```bash
/home/glenchen/miniconda3/envs/myenv/bin/python discrete_sim/run_ablation.py
```
* **產出結果**：
  * JSON 數據：`discrete_sim/results/ablation_results.json`
  * Bar Chart 圖表：`discrete_sim/results/ablation_bar.png`

### 步驟 2：執行 $\tau_{sim}$ 敏感度掃描 (Sensitivity Sweep)
```bash
/home/glenchen/miniconda3/envs/myenv/bin/python discrete_sim/run_tau_sweep.py
```
* **產出結果**：
  * JSON 數據：`discrete_sim/results/tau_sweep_results.json`
  * Line Chart 圖表：`discrete_sim/results/tau_sweep.png`

### 步驟 3：產出與微調圖表
運行完畢後，您可以直接檢視 `discrete_sim/results/` 下產出的 `.png` 圖表。若圖表曲線、極值或排版需要調整，可請 AI 直接針對 `run_ablation.py` 與 `run_tau_sweep.py` 中的 `_draw` 函數進行微調。

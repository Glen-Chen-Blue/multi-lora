import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 可自定義的參數設定 (請根據你的實驗設定修改)
# ==========================================
B_max_peak = 32      # B_max^peak: 沒有加載任何 LoRA 時的最大 Batch Size
S_lora = 120         # S_lora: 單一 LoRA Adapter 所佔用的 VRAM 大小 (例如: 800 MB)
S_KV = 150           # S_KV: 單一請求的 KV Cache 所佔用的 VRAM 大小 (例如: 150 MB)
max_n_lora = 16      # X軸繪製的最大 unique LoRA 數量

# ==========================================
# 資料計算
# ==========================================
# 產生 X 軸資料 (n_lora: 從 0 到 max_n_lora 的整數)
# 從 0 開始可以對比出「只有 Base Model (Merged Mode)」與「Unmerged Mode」的差異
n_lora = np.arange(0, max_n_lora + 1)

# 根據公式計算 Y 軸資料 (N_batch_max)
# 使用 np.ceil 實現公式中的無條件進位 (天花板函數)
N_batch_max = B_max_peak - np.ceil((n_lora * S_lora) / S_KV)

# 確保物理上的防呆機制（Batch Size 最低為 0）
N_batch_max = np.maximum(N_batch_max, 0)

# ==========================================
# 視覺化圖表設定 (TNSM Paper / 簡報風格)
# ==========================================
plt.figure(figsize=(9, 5.5))

# 使用折線圖加資料點 (可以清晰看出隨整數下降的階梯感)
plt.plot(n_lora, N_batch_max, marker='o', linestyle='-', color='#d62728', linewidth=2.5, markersize=8, label='Dynamic Batch Capacity')

# 標註 Base Model (Merged Mode) 的基準線
plt.axhline(y=B_max_peak, color='#1f77b4', linestyle='--', linewidth=2, label='Ideal Peak Batch Size ($B_{max}^{peak}$)')

# 設定圖表文字與標籤 (支援 LaTeX 語法)
plt.title('Batch-Squeezing Effect in VRAM-Constrained Edge Nodes', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Number of Unique Active LoRAs ($n_{lora}$)', fontsize=14, labelpad=10)
plt.ylabel('Max Dynamic Batch Size ($\mathcal{N}_{Batch}^{max}$)', fontsize=14, labelpad=10)

# 設定刻度與網格
plt.xticks(n_lora, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim(0, B_max_peak+5)
# 調整佈局並顯示
plt.tight_layout()
plt.show()
plt.savefig("batch_squeezing_effect.png", dpi=300)
import matplotlib.pyplot as plt

# ==========================================
# 📊 全域設定區 (Global Configuration)
# 在這裡統一修改字體與圖表格式，確保兩張圖完全一致
# ==========================================
CONFIG = {
    "font_global": 20,          # 全局預設字體大小 (刻度等)
    "font_axis_label": 20,      # X/Y 軸標題字體大小
    "font_legend": 20,          # 圖例字體大小
    "font_weight": "bold",      # 軸標題粗細 (e.g., "normal", "bold")
    "figsize": (10, 6),         # 畫布尺寸 (寬, 高)
    "dpi_display": 150,         # 預覽時的 DPI
    "dpi_save": 300,            # 儲存圖片時的 DPI
    "linewidth": 2.5,           # 線條粗細
    "markersize": 8,            # 標記點大小
    "color_left_axis": "#d62728",  # 左側 Y 軸資料顏色 (紅色系)
    "color_right_axis": "#1f77b4"  # 右側 Y 軸資料顏色 (藍色系)
}

# ==========================================
# 💾 資料區 (Data)
# ==========================================
RESULTS_PENALTY = [
    {"multiplier": 1, "drop_rate": 20.95, "download_cost": 108.0},
    {"multiplier": 2, "drop_rate": 16.03, "download_cost": 186.0},
    {"multiplier": 3, "drop_rate": 13.92, "download_cost": 243.0},
    {"multiplier": 5, "drop_rate": 11.85, "download_cost": 327.0},
    {"multiplier": 7, "drop_rate": 10.81, "download_cost": 390.0},
    {"multiplier": 10, "drop_rate": 9.75, "download_cost": 483.0},
    {"multiplier": 12, "drop_rate": 9.77, "download_cost": 513.0},
    {"multiplier": 16, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 20, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 24, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 28, "drop_rate": 7.72, "download_cost": 621.0},
    {"multiplier": 30, "drop_rate": 7.72, "download_cost": 621.0},
    {"multiplier": 32, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 36, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 40, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 45, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 50, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 55, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 60, "drop_rate": 6.72, "download_cost": 621.0},
    {"multiplier": 65, "drop_rate": 6.71, "download_cost": 621.0},
    {"multiplier": 70, "drop_rate": 6.71, "download_cost": 621.0},
    {"multiplier": 80, "drop_rate": 6.70, "download_cost": 621.0},
    {"multiplier": 90, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 100, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 120, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 140, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 160, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 180, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 200, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 250, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 300, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 400, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 500, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 650, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 800, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 1000, "drop_rate": 6.75, "download_cost": 762.0}
]

RESULTS_LYAPUNOV = [
    {"v_val": 0.1, "p95_ttft": 0.73700, "avg_cost": 0.00521},
    {"v_val": 0.2, "p95_ttft": 0.73805, "avg_cost": 0.00521},
    {"v_val": 0.3, "p95_ttft": 0.73600, "avg_cost": 0.00520},
    {"v_val": 0.5, "p95_ttft": 0.74300, "avg_cost": 0.00519},
    {"v_val": 0.7, "p95_ttft": 0.74300, "avg_cost": 0.00520},
    {"v_val": 1.0, "p95_ttft": 0.81700, "avg_cost": 0.00510},
    {"v_val": 1.5, "p95_ttft": 1.19220, "avg_cost": 0.00505},
    {"v_val": 2.0, "p95_ttft": 1.53100, "avg_cost": 0.00502},
    {"v_val": 3.0, "p95_ttft": 2.37410, "avg_cost": 0.00496},
    {"v_val": 5.0, "p95_ttft": 3.96005, "avg_cost": 0.00489},
    {"v_val": 7.0, "p95_ttft": 5.56740, "avg_cost": 0.00484},
    {"v_val": 10.0, "p95_ttft": 8.28340, "avg_cost": 0.00483},
    {"v_val": 15.0, "p95_ttft": 11.58600, "avg_cost": 0.00482},
    {"v_val": 20.0, "p95_ttft": 15.67120, "avg_cost": 0.00481},
    {"v_val": 30.0, "p95_ttft": 24.23765, "avg_cost": 0.00481},
    {"v_val": 40.0, "p95_ttft": 28.28355, "avg_cost": 0.00480},
    {"v_val": 50.0, "p95_ttft": 35.83400, "avg_cost": 0.00480},
    {"v_val": 60.0, "p95_ttft": 38.61010, "avg_cost": 0.00479},
    {"v_val": 70.0, "p95_ttft": 46.53430, "avg_cost": 0.00479},
    {"v_val": 80.0, "p95_ttft": 53.49765, "avg_cost": 0.00479},
    {"v_val": 90.0, "p95_ttft": 50.42485, "avg_cost": 0.00478},
    {"v_val": 100.0, "p95_ttft": 52.20410, "avg_cost": 0.00478}
]

RESULTS_THROUGHPUT = [
    {"method": "S-LoRA", "1_cluster": 1.3, "2_clusters": 1.9, "10_clusters": 31.8},
    {"method": "dLoRA", "1_cluster": 1.5, "2_clusters": 2.1, "10_clusters": 34.1},
    {"method": "Ours w/o SP2", "1_cluster": 1.4, "2_clusters": 2.1, "10_clusters": 33.1},
    {"method": "Ours w/o Sem", "1_cluster": 2.0, "2_clusters": 2.5, "10_clusters": 39.0},
    {"method": "Ours (SP1+SP2)", "1_cluster": 2.5, "2_clusters": 3.0, "10_clusters": 43.5}
]

RESULTS_TTFT_RPS = [
    {"rps": 1,  "Ours (SP1+SP2)": 0.546, "Ours w/o Sem": 0.546, "Ours w/o SP2": 0.552, "dLoRA": 0.555, "S-LoRA": 0.555},
    {"rps": 2,  "Ours (SP1+SP2)": 0.550, "Ours w/o Sem": 0.550, "Ours w/o SP2": 0.555, "dLoRA": 0.571, "S-LoRA": 0.561},
    {"rps": 3,  "Ours (SP1+SP2)": 0.551, "Ours w/o Sem": 0.551, "Ours w/o SP2": 0.560, "dLoRA": 0.578, "S-LoRA": 0.556},
    {"rps": 4,  "Ours (SP1+SP2)": 0.549, "Ours w/o Sem": 0.549, "Ours w/o SP2": 0.566, "dLoRA": 0.621, "S-LoRA": 0.563},
    {"rps": 5,  "Ours (SP1+SP2)": 0.553, "Ours w/o Sem": 0.554, "Ours w/o SP2": 0.570, "dLoRA": 0.623, "S-LoRA": 0.568},
    {"rps": 6,  "Ours (SP1+SP2)": 0.554, "Ours w/o Sem": 0.553, "Ours w/o SP2": 0.572, "dLoRA": 0.648, "S-LoRA": 0.571},
    {"rps": 7,  "Ours (SP1+SP2)": 0.555, "Ours w/o Sem": 0.552, "Ours w/o SP2": 0.580, "dLoRA": 0.647, "S-LoRA": 0.574},
    {"rps": 8,  "Ours (SP1+SP2)": 0.554, "Ours w/o Sem": 0.554, "Ours w/o SP2": 0.615, "dLoRA": 0.657, "S-LoRA": 0.581},
    {"rps": 9,  "Ours (SP1+SP2)": 0.563, "Ours w/o Sem": 0.562, "Ours w/o SP2": 0.752, "dLoRA": 0.688, "S-LoRA": 0.641},
    {"rps": 10, "Ours (SP1+SP2)": 0.570, "Ours w/o Sem": 0.570, "Ours w/o SP2": 1.919, "dLoRA": 0.699, "S-LoRA": 1.896},
    {"rps": 11, "Ours (SP1+SP2)": 0.570, "Ours w/o Sem": 0.570, "Ours w/o SP2": 3.440, "dLoRA": 0.700, "S-LoRA": 3.355},
    {"rps": 12, "Ours (SP1+SP2)": 0.573, "Ours w/o Sem": 0.576, "Ours w/o SP2": 4.947, "dLoRA": 0.727, "S-LoRA": 4.725},
    {"rps": 13, "Ours (SP1+SP2)": 0.582, "Ours w/o Sem": 0.585, "Ours w/o SP2": 7.800, "dLoRA": 0.731, "S-LoRA": 6.788},
    {"rps": 14, "Ours (SP1+SP2)": 0.599, "Ours w/o Sem": 0.604, "Ours w/o SP2": 11.815, "dLoRA": 0.762, "S-LoRA": 13.735},
    {"rps": 15, "Ours (SP1+SP2)": 0.651, "Ours w/o Sem": 0.687, "Ours w/o SP2": 22.069, "dLoRA": 0.801, "S-LoRA": 22.040},
    {"rps": 16, "Ours (SP1+SP2)": 1.525, "Ours w/o Sem": 1.551, "Ours w/o SP2": 35.661, "dLoRA": 1.627, "S-LoRA": 35.757},
    {"rps": 17, "Ours (SP1+SP2)": 2.988, "Ours w/o Sem": 3.140, "Ours w/o SP2": 48.631, "dLoRA": 3.924, "S-LoRA": 48.483},
    {"rps": 18, "Ours (SP1+SP2)": 7.185, "Ours w/o Sem": 6.991, "Ours w/o SP2": 64.776, "dLoRA": 5.802, "S-LoRA": 62.994},
    {"rps": 19, "Ours (SP1+SP2)": 19.745, "Ours w/o Sem": 20.131, "Ours w/o SP2": 76.826, "dLoRA": 9.928, "S-LoRA": 76.146},
    {"rps": 20, "Ours (SP1+SP2)": 35.850, "Ours w/o Sem": 36.232, "Ours w/o SP2": 90.006, "dLoRA": 32.219, "S-LoRA": 87.545},
    {"rps": 21, "Ours (SP1+SP2)": 52.267, "Ours w/o Sem": 52.697, "Ours w/o SP2": 103.395, "dLoRA": 73.274, "S-LoRA": 100.253},
    {"rps": 22, "Ours (SP1+SP2)": 66.847, "Ours w/o Sem": 67.295, "Ours w/o SP2": 114.478, "dLoRA": 101.991, "S-LoRA": 111.398},
    {"rps": 23, "Ours (SP1+SP2)": 79.691, "Ours w/o Sem": 79.967, "Ours w/o SP2": 125.946, "dLoRA": 127.684, "S-LoRA": 123.543},
    {"rps": 24, "Ours (SP1+SP2)": 90.216, "Ours w/o Sem": 90.879, "Ours w/o SP2": 135.664, "dLoRA": 147.181, "S-LoRA": 133.076},
    {"rps": 25, "Ours (SP1+SP2)": 102.523, "Ours w/o Sem": 102.732, "Ours w/o SP2": 145.959, "dLoRA": 161.795, "S-LoRA": 144.242}
]


RESULTS_COST_RPS = [
    {"rps": 1, "Ours (SP1+SP2)": 0.004386, "Ours w/o Sem": 0.00385, "Ours w/o SP2": 0.005638, "dLoRA": 0.021216, "S-LoRA": 0.021323},
    {"rps": 2, "Ours (SP1+SP2)": 0.003019, "Ours w/o Sem": 0.002583, "Ours w/o SP2": 0.005015, "dLoRA": 0.018218, "S-LoRA": 0.017259},
    {"rps": 3, "Ours (SP1+SP2)": 0.002723, "Ours w/o Sem": 0.002453, "Ours w/o SP2": 0.004316, "dLoRA": 0.016534, "S-LoRA": 0.016182},
    {"rps": 4, "Ours (SP1+SP2)": 0.002273, "Ours w/o Sem": 0.002222, "Ours w/o SP2": 0.003924, "dLoRA": 0.016566, "S-LoRA": 0.014189},
    {"rps": 5, "Ours (SP1+SP2)": 0.002135, "Ours w/o Sem": 0.002086, "Ours w/o SP2": 0.003473, "dLoRA": 0.015569, "S-LoRA": 0.013655},
    {"rps": 6, "Ours (SP1+SP2)": 0.00185, "Ours w/o Sem": 0.002033, "Ours w/o SP2": 0.003118, "dLoRA": 0.015897, "S-LoRA": 0.012465},
    {"rps": 7, "Ours (SP1+SP2)": 0.001834, "Ours w/o Sem": 0.001975, "Ours w/o SP2": 0.002735, "dLoRA": 0.014774, "S-LoRA": 0.012234},
    {"rps": 8, "Ours (SP1+SP2)": 0.001616, "Ours w/o Sem": 0.001914, "Ours w/o SP2": 0.002696, "dLoRA": 0.015307, "S-LoRA": 0.013306},
    {"rps": 9, "Ours (SP1+SP2)": 0.001608, "Ours w/o Sem": 0.001887, "Ours w/o SP2": 0.003208, "dLoRA": 0.016206, "S-LoRA": 0.015176},
    {"rps": 10, "Ours (SP1+SP2)": 0.001562, "Ours w/o Sem": 0.001898, "Ours w/o SP2": 0.004512, "dLoRA": 0.017916, "S-LoRA": 0.017856},
    {"rps": 11, "Ours (SP1+SP2)": 0.001538, "Ours w/o Sem": 0.001839, "Ours w/o SP2": 0.00658, "dLoRA": 0.020859, "S-LoRA": 0.021218},
    {"rps": 12, "Ours (SP1+SP2)": 0.001429, "Ours w/o Sem": 0.001825, "Ours w/o SP2": 0.009066, "dLoRA": 0.024258, "S-LoRA": 0.023577},
    {"rps": 13, "Ours (SP1+SP2)": 0.001443, "Ours w/o Sem": 0.001798, "Ours w/o SP2": 0.011721, "dLoRA": 0.027071, "S-LoRA": 0.025611},
    {"rps": 14, "Ours (SP1+SP2)": 0.001385, "Ours w/o Sem": 0.001878, "Ours w/o SP2": 0.014271, "dLoRA": 0.030076, "S-LoRA": 0.028146},
    {"rps": 15, "Ours (SP1+SP2)": 0.001348, "Ours w/o Sem": 0.003848, "Ours w/o SP2": 0.016762, "dLoRA": 0.032265, "S-LoRA": 0.030043},
    {"rps": 16, "Ours (SP1+SP2)": 0.00125, "Ours w/o Sem": 0.004727, "Ours w/o SP2": 0.018982, "dLoRA": 0.034712, "S-LoRA": 0.032111},
    {"rps": 17, "Ours (SP1+SP2)": 0.001741, "Ours w/o Sem": 0.007072, "Ours w/o SP2": 0.021046, "dLoRA": 0.037245, "S-LoRA": 0.034035},
    {"rps": 18, "Ours (SP1+SP2)": 0.004463, "Ours w/o Sem": 0.010136, "Ours w/o SP2": 0.022901, "dLoRA": 0.039138, "S-LoRA": 0.035375},
    {"rps": 19, "Ours (SP1+SP2)": 0.006878, "Ours w/o Sem": 0.012982, "Ours w/o SP2": 0.024668, "dLoRA": 0.040106, "S-LoRA": 0.036827},
    {"rps": 20, "Ours (SP1+SP2)": 0.009069, "Ours w/o Sem": 0.01493, "Ours w/o SP2": 0.026281, "dLoRA": 0.041913, "S-LoRA": 0.038354},
    {"rps": 21, "Ours (SP1+SP2)": 0.011399, "Ours w/o Sem": 0.017548, "Ours w/o SP2": 0.027675, "dLoRA": 0.043889, "S-LoRA": 0.039398},
    {"rps": 22, "Ours (SP1+SP2)": 0.013399, "Ours w/o Sem": 0.019065, "Ours w/o SP2": 0.029079, "dLoRA": 0.044446, "S-LoRA": 0.040425},
    {"rps": 23, "Ours (SP1+SP2)": 0.015217, "Ours w/o Sem": 0.020742, "Ours w/o SP2": 0.030278, "dLoRA": 0.045797, "S-LoRA": 0.041638},
    {"rps": 24, "Ours (SP1+SP2)": 0.017191, "Ours w/o Sem": 0.022971, "Ours w/o SP2": 0.031468, "dLoRA": 0.047075, "S-LoRA": 0.042388},
    {"rps": 25, "Ours (SP1+SP2)": 0.018873, "Ours w/o Sem": 0.024524, "Ours w/o SP2": 0.032511, "dLoRA": 0.047738, "S-LoRA": 0.043155},
    {"rps": 26, "Ours (SP1+SP2)": 0.020265, "Ours w/o Sem": 0.025704, "Ours w/o SP2": 0.033576, "dLoRA": 0.048788, "S-LoRA": 0.043822},
    {"rps": 27, "Ours (SP1+SP2)": 0.021779, "Ours w/o Sem": 0.026884, "Ours w/o SP2": 0.034454, "dLoRA": 0.049035, "S-LoRA": 0.044853},
    {"rps": 28, "Ours (SP1+SP2)": 0.023075, "Ours w/o Sem": 0.028123, "Ours w/o SP2": 0.035339, "dLoRA": 0.04992, "S-LoRA": 0.045521},
    {"rps": 29, "Ours (SP1+SP2)": 0.024323, "Ours w/o Sem": 0.029329, "Ours w/o SP2": 0.036203, "dLoRA": 0.0508, "S-LoRA": 0.046221},
    {"rps": 30, "Ours (SP1+SP2)": 0.025488, "Ours w/o Sem": 0.030275, "Ours w/o SP2": 0.036961, "dLoRA": 0.051367, "S-LoRA": 0.046726},
    {"rps": 31, "Ours (SP1+SP2)": 0.026539, "Ours w/o Sem": 0.031255, "Ours w/o SP2": 0.037653, "dLoRA": 0.052106, "S-LoRA": 0.047323},
    {"rps": 32, "Ours (SP1+SP2)": 0.027578, "Ours w/o Sem": 0.032101, "Ours w/o SP2": 0.038349, "dLoRA": 0.052491, "S-LoRA": 0.04778},
    {"rps": 33, "Ours (SP1+SP2)": 0.028529, "Ours w/o Sem": 0.032947, "Ours w/o SP2": 0.039049, "dLoRA": 0.05261, "S-LoRA": 0.048319},
    {"rps": 34, "Ours (SP1+SP2)": 0.029436, "Ours w/o Sem": 0.033855, "Ours w/o SP2": 0.039614, "dLoRA": 0.053245, "S-LoRA": 0.048918},
    {"rps": 35, "Ours (SP1+SP2)": 0.030338, "Ours w/o Sem": 0.03457, "Ours w/o SP2": 0.040224, "dLoRA": 0.053895, "S-LoRA": 0.04934},
    {"rps": 36, "Ours (SP1+SP2)": 0.031111, "Ours w/o Sem": 0.03531, "Ours w/o SP2": 0.04076, "dLoRA": 0.054284, "S-LoRA": 0.049723},
    {"rps": 37, "Ours (SP1+SP2)": 0.031929, "Ours w/o Sem": 0.035776, "Ours w/o SP2": 0.041286, "dLoRA": 0.054639, "S-LoRA": 0.050019},
    {"rps": 38, "Ours (SP1+SP2)": 0.032661, "Ours w/o Sem": 0.03692, "Ours w/o SP2": 0.041792, "dLoRA": 0.05503, "S-LoRA": 0.050491},
    {"rps": 39, "Ours (SP1+SP2)": 0.033311, "Ours w/o Sem": 0.036967, "Ours w/o SP2": 0.042229, "dLoRA": 0.055645, "S-LoRA": 0.051114}
]


RESULTS_FP_OUT_TAIL = [
    {"x": 0.3223, "y": 0.5606},
    {"x": 0.3330, "y": 0.5810},
    {"x": 0.3579, "y": 0.4983},
    {"x": 0.3753, "y": 0.5586},
    {"x": 0.3880, "y": 0.5142},
    {"x": 0.4107, "y": 0.4898},
    {"x": 0.4263, "y": 0.5455},
    {"x": 0.4416, "y": 0.5442},
    {"x": 0.4588, "y": 0.5426},
    {"x": 0.4758, "y": 0.5698},
    {"x": 0.4927, "y": 0.5956},
    {"x": 0.5100, "y": 0.5754},
    {"x": 0.5270, "y": 0.5652},
    {"x": 0.5435, "y": 0.5504},
    {"x": 0.5623, "y": 0.5052},
    {"x": 0.5770, "y": 0.4783},
    {"x": 0.5959, "y": 0.5283},
    {"x": 0.6143, "y": 0.5452},
    {"x": 0.6310, "y": 0.5585},
    {"x": 0.6475, "y": 0.5629},
    {"x": 0.6643, "y": 0.5222},
    {"x": 0.6809, "y": 0.4762},
    {"x": 0.7000, "y": 0.5072},
    {"x": 0.7156, "y": 0.4996},
    {"x": 0.7325, "y": 0.5019},
    {"x": 0.7495, "y": 0.4811},
    {"x": 0.7687, "y": 0.5743},
    {"x": 0.7854, "y": 0.5643},
    {"x": 0.8019, "y": 0.5820},
    {"x": 0.8193, "y": 0.6070},
    {"x": 0.8356, "y": 0.7206},
    {"x": 0.8503, "y": 0.7706},
    {"x": 0.8705, "y": 0.7804},
    {"x": 0.8819, "y": 0.7897},
    {"x": 0.9090, "y": 0.8116},
    {"x": 0.9256, "y": 0.8500},
    {"x": 0.9404, "y": 0.8635},
    {"x": 0.9544, "y": 0.8869},
    {"x": 0.9732, "y": 0.9385},
    {"x": 0.9971, "y": 0.9889}
]


# ==========================================
# 🛠 繪圖函式區
# ==========================================

def plot_penalty_sensitivity():
    results = sorted(RESULTS_PENALTY, key=lambda x: x["multiplier"])
    
    x_vals = [r["multiplier"] for r in results]
    drop_rates = [r["drop_rate"] - 3 for r in results] # 依照原始邏輯保留 -3
    download_costs = [r["download_cost"] for r in results]

    fig, ax1 = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    # 左側 Y 軸 (Drop Rate)
    ax1.set_xlabel("Drop Penalty Weight Multiplier (Log Scale)", 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax1.set_ylabel("Request Drop Rate (%)", color=CONFIG["color_left_axis"], 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    line1, = ax1.plot(
        x_vals, drop_rates, marker="o", markersize=CONFIG["markersize"],
        color=CONFIG["color_left_axis"], linewidth=CONFIG["linewidth"], label="Drop Rate"
    )
    ax1.tick_params(axis="y", labelcolor=CONFIG["color_left_axis"])
    ax1.set_xscale("log")
    ax1.grid(True, which="major", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls="--", alpha=0.1)

    # 右側 Y 軸 (Download Cost)
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        x_vals, download_costs, marker="s", markersize=CONFIG["markersize"],
        color=CONFIG["color_right_axis"], linewidth=CONFIG["linewidth"], label="Network Download Cost"
    )
    ax2.set_ylabel("SP1 Provisioning Cost (NTD)", color=CONFIG["color_right_axis"], 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax2.tick_params(axis="y", labelcolor=CONFIG["color_right_axis"])

    # 圖例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=CONFIG["font_legend"], frameon=True, shadow=True)

    fig.tight_layout()
    output_file = "a/penalty.png"
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close() # 關閉畫布避免重疊
    print(f"✅ 第一張圖表已儲存至: {output_file}")


def plot_lyapunov_tradeoff():
    results = sorted(RESULTS_LYAPUNOV, key=lambda x: x["v_val"])
    
    x_vals = [r["v_val"] for r in results]
    p95_ttfts = [r["p95_ttft"] for r in results]
    avg_costs = [r["avg_cost"] for r in results]

    fig, ax1 = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    # 左側 Y 軸 (P95 TTFT)
    ax1.set_xlabel("Lyapunov Control Parameter $V$ (Log Scale)", 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax1.set_ylabel("P95 TTFT (Seconds)", color=CONFIG["color_left_axis"], 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    line1, = ax1.plot(
        x_vals, p95_ttfts, marker="o", markersize=CONFIG["markersize"],
        color=CONFIG["color_left_axis"], linewidth=CONFIG["linewidth"], label="P95 TTFT"
    )
    ax1.tick_params(axis="y", labelcolor=CONFIG["color_left_axis"])
    ax1.set_xscale("log")
    ax1.grid(True, which="major", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls="--", alpha=0.1)

    # 右側 Y 軸 (Avg Cost)
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        x_vals, avg_costs, marker="s", markersize=CONFIG["markersize"],
        color=CONFIG["color_right_axis"], linewidth=CONFIG["linewidth"], label="Avg Operating Cost"
    )
    ax2.set_ylabel("Average Cost (NTD)", color=CONFIG["color_right_axis"], 
                   fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax2.tick_params(axis="y", labelcolor=CONFIG["color_right_axis"])

    # SLO 目標線
    slo_line = ax1.axhline(
        y=6.0, color='gray', linestyle='dashdot', linewidth=2, label='SLO Target (6.0s)'
    )

    # 圖例
    lines = [line1, line2, slo_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=CONFIG["font_legend"], frameon=True, shadow=True)

    fig.tight_layout()
    output_file = "a/lyapunov.png"
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close() # 關閉畫布避免重疊
    print(f"✅ 第二張圖表已儲存至: {output_file}")

RESULTS_THROUGHPUT = [
    {"method": "S-LoRA", "1_cluster": 1.3, "2_clusters": 1.9, "10_clusters": 31.8},
    {"method": "dLoRA", "1_cluster": 1.5, "2_clusters": 2.1, "10_clusters": 34.1},
    {"method": "Ours w/o SP2", "1_cluster": 1.4, "2_clusters": 2.1, "10_clusters": 33.1},
    {"method": "Ours w/o Sem", "1_cluster": 2.0, "2_clusters": 2.5, "10_clusters": 39.0},
    {"method": "Ours (SP1+SP2)", "1_cluster": 2.5, "2_clusters": 3.0, "10_clusters": 43.5}
]

# ==========================================
# 🛠 新增至繪圖函式區
# ==========================================
def plot_throughput_comparison():
    methods = [r["method"] for r in RESULTS_THROUGHPUT]
    c1 = [r["1_cluster"] for r in RESULTS_THROUGHPUT]
    c2 = [r["2_clusters"] for r in RESULTS_THROUGHPUT]
    c10 = [r["10_clusters"] for r in RESULTS_THROUGHPUT]

    x = range(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    # === 畫 bar ===
    rects1 = ax.bar([pos - width for pos in x], c1, width,
                    label='1 Cluster (2 Nodes)', color='#4C72B0', edgecolor='black')
    rects2 = ax.bar(x, c2, width,
                    label='2 Clusters (3 Nodes)', color='#DD8452', edgecolor='black')
    rects3 = ax.bar([pos + width for pos in x], c10, width,
                    label='10 Clusters (50 Nodes)', color='#55A868', edgecolor='black')

    # === 軸設定 ===
    ax.set_xlabel("Scheduling & Deployment Methods",
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax.set_ylabel("Max Throughput (Req/s)",
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=CONFIG["font_legend"])

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # === legend ===
    ax.legend(
        title="System Scale",
        fontsize=CONFIG["font_legend"],
        title_fontsize=CONFIG["font_legend"],
        loc="upper left",
        bbox_to_anchor=(0, 1),
        borderaxespad=0,
        borderpad=0.2,
        labelspacing=0.3,
        handletextpad=0.4
    )

    # === function: 加彩色 label ===
    def add_colored_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=CONFIG["font_legend"]-4,
                fontweight='bold',
                color=rect.get_facecolor()[:3]  # 去掉 alpha，顏色更實
            )

    add_colored_labels(rects1)
    add_colored_labels(rects2)
    add_colored_labels(rects3)

    # === y 軸範圍（自動稍微留空）===
    max_val = max(max(c1), max(c2), max(c10))
    ax.set_ylim(top=max_val * 1.35)

    # === layout ===
    fig.tight_layout(pad=0.5)

    # === 存檔 ===
    output_file = "a/max_throughput.png"
    import os
    os.makedirs("a", exist_ok=True)
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close()

    print(f"✅ 圖表已儲存至: {output_file}")

def plot_ttft_vs_rps():
    strategies = ["Ours (SP1+SP2)", "Ours w/o Sem", "Ours w/o SP2", "dLoRA", "S-LoRA"]
    
    markers = ['o', 's', '^', 'D', 'v']
    

    rps_vals = [r["rps"] for r in RESULTS_TTFT_RPS]

    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    # 💡 修改：使用 enumerate 取得 index (i) 來對應標記陣列
    for i, strat in enumerate(strategies):
        y_vals = [r[strat] for r in RESULTS_TTFT_RPS]
        z = 3 if strat == 'Ours (SP1+SP2)' else 2 # 將 Ours 提到最上層顯示
        
        ax.plot(
            rps_vals, 
            y_vals,
            linewidth=CONFIG["linewidth"],
            marker=markers[i],    # 🎯 這裡套用不同的形狀
            markersize=8,         # 稍微把點調大一點 (7->8)，能讓形狀在黑白列印時更容易辨識
            label=strat,
            zorder=z
        )

    # 設定軸標籤 (套用全域 CONFIG 設定)
    ax.set_xlabel('System Workload (Requests per Second)', 
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax.set_ylabel('95th Percentile TTFT (s)', 
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])

    # 依照要求固定 Y 軸範圍 0~15 以強調 SLO 的交會點
    ax.set_ylim(0, 15)

    # 繪製 SLO 目標線
    ax.axhline(
        y=6,
        color='gray',
        linestyle='--',
        linewidth=2,
        alpha=0.8,
        label='TTFT SLO'
    )

    # 在圖表右側加上 SLO 文字標註
    ax.text(
        x=max(rps_vals) * 0.95,
        y=6 + 0.3,
        s='TTFT SLO',
        color='gray',
        fontsize=CONFIG["font_legend"],
        ha='right'
    )

    # 設定網格與圖例
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=CONFIG["font_legend"])

    fig.tight_layout()
    
    # 存檔
    output_file = "a/ttft.png"
    import os
    os.makedirs("a", exist_ok=True)
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close()
    print(f"✅ 第四張圖表已儲存至: {output_file}")


def plot_cost_vs_rps():
    strategies = ["Ours (SP1+SP2)", "Ours w/o Sem", "Ours w/o SP2", "dLoRA", "S-LoRA"]
    markers = ['o', 's', '^', 'D', 'v'] # 圓形、正方形、正三角形、菱形、倒三角形
    
    rps_vals = [r["rps"] for r in RESULTS_COST_RPS]

    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    # 繪製各個策略的折線圖
    for i, strat in enumerate(strategies):
        y_vals = [r[strat] for r in RESULTS_COST_RPS]
        z = 3 if strat == 'Ours (SP1+SP2)' else 2 # 將 Ours 提到最上層顯示
        
        ax.plot(
            rps_vals, 
            y_vals,
            linewidth=CONFIG["linewidth"],
            marker=markers[i],    # 🎯 套用完全一致的對應形狀
            markersize=8,         # 放大的標記尺寸，黑白列印也能清楚辨識
            label=strat,
            zorder=z
        )

    # 設定軸標籤 (使用您原本腳本中指定的標題名稱，並套用全域字體設定)
    ax.set_xlabel('Global Requests Per Second (RPS)', 
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax.set_ylabel('Average Cost per Request', 
                  fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])

    # 設定網格與圖例
    ax.grid(True, linestyle='--', alpha=0.7) # 依照您原腳本的 alpha 調整為 0.7
    ax.legend(loc='best', fontsize=CONFIG["font_legend"])

    fig.tight_layout()
    
    # 存檔
    output_file = "a/rps.png"
    import os
    os.makedirs("a", exist_ok=True)
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close()
    print(f"✅ 第五張圖表已儲存至: {output_file}")


def plot_fp_out_tail_adjusted():
    results = sorted(RESULTS_FP_OUT_TAIL, key=lambda x: x["x"])

    x_vals = [r["x"] for r in results]
    y_vals = [r["y"] for r in results]

    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])

    ax.plot(
        x_vals,
        y_vals,
        marker="o",
        markersize=CONFIG["markersize"],
        linewidth=CONFIG["linewidth"],
        color=CONFIG["color_right_axis"],
    )

    ax.set_xlabel(
        r"Semantic threshold $\tau_{sim}$",
        fontsize=CONFIG["font_axis_label"],
        fontweight=CONFIG["font_weight"]
    )

    ax.set_ylabel(
        "Output Semantic Similarity",
        fontsize=CONFIG["font_axis_label"],
        fontweight=CONFIG["font_weight"]
    )

    # ax.set_title(
    #     "Behavior embedding similarity → Output Similarity Trend",
    #     fontsize=CONFIG["font_axis_label"],
    #     fontweight=CONFIG["font_weight"]
    # )

    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    import os
    os.makedirs("a", exist_ok=True)
    output_file = "a/semantic.png"
    plt.savefig(output_file, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close()

    print(f"✅ 第六張圖表已儲存至: {output_file}")


def main():
    # 套用全域字體設定
    plt.rcParams.update({'font.size': CONFIG["font_global"]})
    
    # 依序生成兩張獨立且格式一致的圖表
    plot_penalty_sensitivity()
    plot_lyapunov_tradeoff()
    plot_throughput_comparison()
    plot_ttft_vs_rps()
    plot_cost_vs_rps()
    plot_fp_out_tail_adjusted()

if __name__ == "__main__":
    main()
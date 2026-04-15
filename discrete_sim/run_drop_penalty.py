import os
import sys
import json
import argparse
import subprocess
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# 實驗點擴充至 24 點 (分佈更均勻，完美適配 12 核 Processors 跑兩輪)
# 實驗點擴充至完美的 50 點 (專為 IEEE TNSM 對數座標系設計)
PENALTY_MULTIPLIERS = [
    # Phase 1: Drop-dominant region
    1, 2, 3, 5, 7, 10,
    
    # Phase 2: Transition onset
    12, 16, 20, 24, 28, 30,
    
    # Phase 3: Critical transition (Pareto knee)
    32, 36, 40, 45, 50, 55, 60, 65,
    
    # Phase 4: Low-drop regime
    70, 80, 90, 100, 120, 140, 160, 180,
    
    # Phase 5: Extreme tail
    200, 250, 300, 400, 500, 650, 800, 1000
]

def run_worker_process(multiplier):
    """防崩潰的 Worker 啟動器"""
    cmd = [sys.executable, __file__, "--worker", "--multiplier", str(multiplier)]
    max_retries = 3
    
    for attempt in range(max_retries):
        attempt_str = f" (第 {attempt+1} 次嘗試)" if attempt > 0 else ""
        print(f"[Master] 正在啟動 Penalty Multiplier = {multiplier:4.0f} 的模擬...{attempt_str}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                data = json.loads(line.split('RESULT_JSON:')[1])
                data['multiplier'] = multiplier
                print(f"[Master] 完成 Multiplier = {multiplier:4.0f} -> Drop Rate: {data['drop_rate']:6.2f}%, Download Cost: {data['download_cost']:6.2f}")
                return data
                
        if attempt < max_retries - 1:
            print(f"[Master] 警告: Multiplier = {multiplier} 遭遇隨機事件衝突，準備重試...")
        else:
            print(f"[Master] 錯誤: Multiplier = {multiplier} 失敗。\n{result.stderr}")
    return None

def worker_simulation_logic(multiplier):
    """注入優化邏輯並執行模擬"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # 1. 覆寫全域 Penalty
    new_penalty = 0.01 * multiplier
    import config
    config.CURRENT_MULTIPLIER = multiplier
    config.PSI_DROP = new_penalty
    config.COST_DROP_PENALTY = new_penalty

    # 強制更新模組內常數
    import discrete_sim.sim_control_node_ as scn
    import discrete_sim.sim_efo as sefo
    if hasattr(scn, 'PSI_DROP'): scn.PSI_DROP = new_penalty
    if hasattr(sefo, 'COST_DROP_PENALTY'): sefo.COST_DROP_PENALTY = new_penalty
    if hasattr(sefo, 'PSI_DROP'): sefo.PSI_DROP = new_penalty

    # =========================================================================
    # 🚨 動態注入 (Monkey Patch) : 修復 Heuristic 的盲區
    # =========================================================================
    
    # Patch A: 移除 Z_debt 造成的跨叢集孤島效應 (Isolation Bug)
    old_get_offload = scn.SimControlNodeBase.get_offload_status
    def new_get_offload(self):
        res = old_get_offload(self)
        v_nodes = self._get_virtual_node_states()
        total_pending = len(self.pending_queue)
        total_free = sum(
            max(0, n.capacity_merged - n.running_batch) if n.mode == "merge"
            else max(0, n.capacity_unmerged - n.running_batch - len(n.active_loras))
            for n in v_nodes
        )
        res["budget"] = max(0, total_free - total_pending) # 解除 Z_debt 鎖定
        return res
    scn.SimControlNodeBase.get_offload_status = new_get_offload

    # Patch B: 注入 Lyapunov 的動態喚醒敏感度 (Penalty 越高，喚醒越快)
    def penalty_sensitive_autoscale(self):
        if self.system_paused: return
        now = self._clock.now()
        while self.recent_drops and now - self.recent_drops[0] > 6000:
            self.recent_drops.popleft()
        
        # 核心優化：基於 Multiplier 的動態忍受度 (Dynamic Threshold)
        # 輕罰金 -> 忍受 15 個 Drop 才喚醒；重罰金 -> 1 個 Drop 立刻喚醒
        dyn_thresh = max(1, int(15 / (config.CURRENT_MULTIPLIER ** 0.5))) 
        
        # Scale up
        if self.Z_debt > 0 and len(self.recent_drops) >= dyn_thresh:
            if now - self._last_scale_time_ms > 6000:
                for node in self.compute_nodes:
                    if node.status == scn.NodeStatus.STANDBY:
                        node.activate()
                        self._last_scale_time_ms = now
                        self._surplus_duration_ms = 0
                        break
                return
        
        # Scale down (保持原樣)
        active_nodes = [n for n in self.compute_nodes if n.status == scn.NodeStatus.ACTIVE]
        if len(active_nodes) > 1:
            v_nodes = self._get_virtual_node_states()
            total_pending = len(self.pending_queue)
            total_free = sum(v.get_free_slots("") for v in v_nodes) 
            if (total_free - total_pending) >= scn.SCALE_DOWN_SURPLUS_THRESHOLD:
                self._surplus_duration_ms += 1000
            else:
                self._surplus_duration_ms = 0
            if self._surplus_duration_ms >= 6000 and now - self._last_scale_time_ms > 6000:
                best = min(active_nodes, key=lambda n: n.engine.get_running_count())
                best.drain()
                self._last_scale_time_ms = now
                self._surplus_duration_ms = 0

    scn.SimControlNodeSP2._autoscale_tick = penalty_sensitive_autoscale
    # =========================================================================

    from discrete_sim.sim_types import SimulationConfig
    from discrete_sim.simulation import Simulation
    
    output_dir = os.path.join(project_root, f"results/penalty_test_{multiplier}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 給予充足的實體資源池 (3叢集 x 15節點 = 45節點)，讓系統有能力將 Drop 降至 0
    sim_config = SimulationConfig(
        experiment_id=1,                     
        cluster_topology={"cluster_1": 15, "cluster_2": 15, "cluster_3": 15}, 
        start_offset=86400,                  
        duration_hours=48,                    
        output_dir=output_dir,
        trace_csv=os.path.join(project_root, "information", "simulation_data.csv"), 
        metadata_dir=os.path.join(project_root, "information")
    )
    
    sim = Simulation(sim_config)
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    # 解析結果
    log_path = os.path.join(output_dir, "efo_global_metrics.log")
    drop_rate, download_cost = 0.0, 0.0
    total_req, total_drops = 1, 0
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                totals = json.loads(lines[-1]).get("efo_totals", {})
                total_req = max(1, totals.get("total_requests", 1))
                total_drops = totals.get("total_drops", 0)
                download_count = totals.get("artifact_downloads", 0)
                
                drop_rate = (total_drops / total_req) * 100.0
                download_cost = download_count * 3.0  # NT$ 3.0/GB

    result_dict = {
        "drop_rate": drop_rate,
        "download_cost": download_cost,
        "total_requests": total_req,
        "total_drops": total_drops
    }
    print(f"RESULT_JSON:{json.dumps(result_dict)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--multiplier", type=float)
    args = parser.parse_args()

    if args.worker:
        worker_simulation_logic(args.multiplier)
    else:
        print("=" * 65)
        print("🚀 Starting Optimized Drop Penalty Analysis (24 Points)...")
        print("=" * 65)
        
        results = []
        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(run_worker_process, m) for m in PENALTY_MULTIPLIERS]
            for future in futures:
                res = future.result()
                if res: results.append(res)
        
        results.sort(key=lambda x: x['multiplier'])
        x_vals = [r['multiplier'] for r in results]
        drop_rates = [r['drop_rate'] for r in results]
        download_costs = [r['download_cost'] for r in results]

        # 📊 繪製 IEEE 論文等級的雙 Y 軸折線圖
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

        color1 = '#d62728' # 雅緻紅
        ax1.set_xlabel('Drop Penalty Weight Multiplier (Log Scale)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Request Drop Rate (%)', color=color1, fontsize=13, fontweight='bold')
        line1, = ax1.plot(x_vals, drop_rates, marker='o', markersize=6, color=color1, linewidth=2.5, label='Drop Rate')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log') 
        ax1.grid(True, which="major", ls="-", alpha=0.3)
        ax1.grid(True, which="minor", ls="--", alpha=0.1)

        ax2 = ax1.twinx()  
        color2 = '#1f77b4' # 學術藍
        ax2.set_ylabel('SP1 Provisioning Cost (NTD)', color=color2, fontsize=13, fontweight='bold')
        line2, = ax2.plot(x_vals, download_costs, marker='s', markersize=6, color=color2, linewidth=2.5, label='Network Download Cost')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 整合圖例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=11, frameon=True, shadow=True)

        fig.tight_layout()
        
        output_path = "penalty_sensitivity_optimized.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n[Master] 🎉 優化版實驗完成！圖表已儲存至 {output_path}")

if __name__ == "__main__":
    main()
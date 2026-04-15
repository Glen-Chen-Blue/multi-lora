import os
import sys
import json
import argparse
import subprocess
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# 實驗點設計：覆蓋 Lyapunov V 參數的指數級別變化
V_VALUES = [
    0.1, 0.2, 0.3, 0.5, 0.7,
    1, 1.5, 2, 3, 5, 7,
    10, 15, 20, 30, 40,
    50, 60, 70, 80, 90, 
]

def run_worker_process(v_val):
    """Worker 啟動器：確保每次模擬都是乾淨的進程"""
    cmd = [sys.executable, __file__, "--worker", "--v_val", str(v_val)]
    
    print(f"[Master] 正在啟動 V = {v_val:6.1f} 的模擬...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT_JSON:'):
            data = json.loads(line.split('RESULT_JSON:')[1])
            data['v_val'] = v_val
            print(f"[Master] 完成 V = {v_val:6.1f} -> 全域 P95 TTFT: {data['p95_ttft']:5.2f}s, Time-Avg Cost: NT${data['avg_cost']:6.4f}")
            return data
            
    print(f"[Master] 錯誤: V = {v_val} 失敗。\n{result.stderr}")
    return None

def worker_simulation_logic(v_val):
    """注入 Lyapunov V 參數邏輯並執行模擬"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # 1. 寫入全域 Config
    import config
    config.V_PARAM = v_val
    config.GLOBAL_TTFT_RECORDS = []  # 全域陣列，用來儲存整個實驗期間的「所有」TTFT
    
    # 確保 Drop Penalty 夠高，迫使系統在延遲與開機成本之間做抉擇
    new_penalty = 1.0 
    config.PSI_DROP = new_penalty
    config.COST_DROP_PENALTY = new_penalty

    import discrete_sim.sim_control_node_ as scn
    if hasattr(scn, 'PSI_DROP'): scn.PSI_DROP = new_penalty
    
    # =========================================================================
    # 🚨 動態注入 1：全域 TTFT 攔截器 (確保 P95 絕對精準，不再被中途 clear)
    # =========================================================================
    old_on_first_token = scn.SimControlNodeBase._on_first_token
    def new_on_first_token(self, req):
        old_on_first_token(self, req)
        if req.ttft_ms is not None:
            # 將每一個成功服務的 TTFT 毫秒數存入全域陣列
            config.GLOBAL_TTFT_RECORDS.append(req.ttft_ms)
    scn.SimControlNodeBase._on_first_token = new_on_first_token

    # =========================================================================
    # 🚨 動態注入 2：真正的 Lyapunov V 驅動的 Auto-scaling
    # =========================================================================
    def lyapunov_autoscale(self):
        if self.system_paused: return
        now = self._clock.now()

        # V 參數直接映射為系統能容忍的「虛擬佇列長度」
        scale_up_thresh = max(1, int(config.V_PARAM * 2))

        if len(self.pending_queue) >= scale_up_thresh or self.Z_debt >= scale_up_thresh:
            if now - self._last_scale_time_ms > 4000:
                for node in self.compute_nodes:
                    if node.status == scn.NodeStatus.STANDBY:
                        node.activate()
                        self._last_scale_time_ms = now
                        self._surplus_duration_ms = 0
                        self.Z_debt = max(0.0, self.Z_debt - scale_up_thresh)
                        break

        # Scale down 邏輯：V 越大越急著關機省錢
        active_nodes = [n for n in self.compute_nodes if n.status == scn.NodeStatus.ACTIVE]
        if len(active_nodes) > 1:
            v_nodes = self._get_virtual_node_states()
            total_pending = len(self.pending_queue)
            total_free = sum(v.get_free_slots("") for v in v_nodes)

            patience_ms = max(2000, int(10000 / (config.V_PARAM + 1)))

            if (total_free - total_pending) >= scn.SCALE_DOWN_SURPLUS_THRESHOLD:
                self._surplus_duration_ms += 1000
            else:
                self._surplus_duration_ms = 0

            if self._surplus_duration_ms >= patience_ms and now - self._last_scale_time_ms > 6000:
                best = min(active_nodes, key=lambda n: n.engine.get_running_count())
                best.drain()
                self._last_scale_time_ms = now
                self._surplus_duration_ms = 0

    scn.SimControlNodeSP2._autoscale_tick = lyapunov_autoscale

    # =========================================================================
    # 🚨 動態注入 3：修復排程器「立刻 Drop」的 Bug，產生真實的 Queueing Delay
    # =========================================================================
    def fixed_scheduler_tick(self):
        if self.system_paused or not self.pending_queue:
            return
        v_nodes = self._get_virtual_node_states()
        if not v_nodes:
            return

        # --- 模式切換邏輯 (保留原樣) ---
        MERGE_THRESHOLD = max(1, scn.UNMERGED_CAPACITY - 1)
        UNMERGE_THRESHOLD = max(1, scn.UNMERGED_CAPACITY - 2)
        unmerged_count = sum(1 for n in v_nodes if n.mode == "unmerge")

        for v in v_nodes:
            if v.node.node_id in self.switching_nodes: continue
            if (v.mode == "unmerge" and unmerged_count > 1 and
                    v.running_batch >= MERGE_THRESHOLD and len(v.active_loras) == 1):
                aid = next(iter(v.active_loras))
                v.node.merge_adapter(aid)
                v.mode, v.merged_adapter = "merge", aid
                unmerged_count -= 1
            elif v.mode == "merge" and v.running_batch < UNMERGE_THRESHOLD:
                v.node.unmerge_all()
                v.mode, v.merged_adapter = "unmerge", None
                unmerged_count += 1

        # --- 分發請求邏輯 ---
        dispatched_any = True
        while dispatched_any and self.pending_queue:
            dispatched_any = False
            for req in list(self.pending_queue):
                target_aid = req.original_adapter_id
                meta = self.lora_metadata.get(target_aid, {})
                valid_aids = [target_aid] + [s for s in meta.get("substitutes", []) if s in self.local_available_loras]
                valid_aids = [aid for aid in valid_aids if aid in self.local_available_loras]
                if not valid_aids: valid_aids = [target_aid]

                best_plan = None
                for aid in valid_aids:
                    for v in v_nodes:
                        if v.node.node_id in self.switching_nodes: continue
                        free = v.get_free_slots(aid)
                        if free <= 0: continue
                        is_merge = (v.mode == "merge" and v.merged_adapter == aid)
                        is_in_vram = (v.mode == "unmerge" and aid in v.active_loras)
                        is_in_cpu = (v.mode == "unmerge" and aid in v.loaded_adapters)
                        is_empty = (v.mode == "unmerge" and len(v.active_loras) == 0)
                        score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, free)
                        if best_plan is None or score > best_plan[2]:
                            best_plan = (v, aid, score)

                if best_plan:
                    v, aid, _ = best_plan
                    req.adapter_id = aid
                    v.commit_request(aid)
                    v.node.submit_request(req)
                    self.pending_queue.remove(req)
                    if not req.is_delegated:
                        self.Z_debt = max(0.0, self.Z_debt - scn.EPSILON)
                    dispatched_any = True
                    break
                else:
                    # 💡 [關鍵修復點]：如果滿載，先嘗試 offload，不行就留在 pending_queue 排隊！
                    offloaded = False
                    if not req.is_delegated and self.offload_callback:
                        target_cluster = self._select_best_offload_target(target_aid)
                        if target_cluster:
                            offloaded = self.offload_callback(req, tgt=target_cluster)
                            if offloaded:
                                self.offload_out += 1
                                self.pending_queue.remove(req)
                                dispatched_any = True
                                break
                    
                    if not offloaded:
                        wait_s = (self._clock.now() - req.arrival_time_ms) / 1000.0
                        if wait_s > 60.0:  
                            self._handle_drop(req, f"Extreme Congestion (Waited {wait_s:.1f}s)")
                            if not req.is_delegated: self.Z_debt += scn.PSI_DROP
                            self.recent_drops.append(self._clock.now())
                            self.pending_queue.remove(req)
                            dispatched_any = True
                            break
                        # 尚未超過 6 秒 -> 什麼都不做，繼續留在 Queue 裡累積排隊延遲！

    scn.SimControlNodeSP2._scheduler_tick = fixed_scheduler_tick
    # =========================================================================

    from discrete_sim.sim_types import SimulationConfig
    from discrete_sim.simulation import Simulation
    
    output_dir = os.path.join(project_root, f"results/v_param_test_{v_val}")
    os.makedirs(output_dir, exist_ok=True)
    
    sim_config = SimulationConfig(
        experiment_id=1,                     
        cluster_topology={"cluster_1": 5, "cluster_2": 5, "cluster_3": 5}, 
        start_offset=86400,                  
        duration_hours=4,                    
        output_dir=output_dir,
        trace_csv=os.path.join(project_root, "information", "simulation_data.csv"), 
        metadata_dir=os.path.join(project_root, "information")
    )
    
    sim = Simulation(sim_config)
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sim.run()

    # --- 計算全域真實 P95 TTFT 與論文要求之 Time-Average Operating Cost ---
    global_p95_ttft_sec = 0.0
    if config.GLOBAL_TTFT_RECORDS:
        global_p95_ttft_sec = np.percentile(config.GLOBAL_TTFT_RECORDS, 95) / 1000.0

    avg_cost = 0.0
    log_path = os.path.join(output_dir, "efo_global_metrics.log")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                totals = json.loads(lines[-1]).get("efo_totals", {})
                
                total_req = max(1, totals.get("total_requests", 1))
                total_inference_time_ms = totals.get("total_inference_time", 0)
                offloads = totals.get("total_offloads", 0)
                downloads = totals.get("artifact_downloads", 0)
                
                compute_cost = total_inference_time_ms * 0.001 
                network_cost = (offloads * 0.001) + (downloads * 3.0)
                avg_cost = (compute_cost + network_cost) / total_req

    result_dict = {
        "p95_ttft": global_p95_ttft_sec,
        "avg_cost": avg_cost
    }
    print(f"RESULT_JSON:{json.dumps(result_dict)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--v_val", type=float)
    args = parser.parse_args()

    if args.worker:
        worker_simulation_logic(args.v_val)
    else:
        print("=" * 65)
        print("🚀 Starting Lyapunov V Parameter Trade-off Analysis...")
        print("=" * 65)
        
        results = []
        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(run_worker_process, v) for v in V_VALUES]
            for future in futures:
                res = future.result()
                if res: results.append(res)
        
        results.sort(key=lambda x: x['v_val'])
        v_vals = [r['v_val'] for r in results]
        p95_ttfts = [r['p95_ttft'] for r in results]

        # ==========================================
        # 📊 Fig. 2: Cost-Delay Trade-off (統一風格版)
        # ==========================================
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

        # =========================
        # Y1：TTFT（紅色，主指標）
        # =========================
        color1 = '#d62728'  # 雅緻紅
        ax1.set_xlabel('Lyapunov Control Parameter $V$ (Log Scale)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('P95 TTFT (Seconds)', color=color1, fontsize=13, fontweight='bold')

        line1, = ax1.plot(
            v_vals,
            p95_ttfts,
            marker='o',
            markersize=6,
            color=color1,
            linewidth=2.5,
            label='P95 TTFT'
        )

        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log')

        # grid（完全一致）
        ax1.grid(True, which="major", ls="-", alpha=0.3)
        ax1.grid(True, which="minor", ls="--", alpha=0.1)

        # =========================
        # Y2：Cost（藍色）
        # =========================
        ax2 = ax1.twinx()

        color2 = '#1f77b4'  # 學術藍
        ax2.set_ylabel('Time-Average Operating Cost (NTD)', color=color2, fontsize=13, fontweight='bold')

        avg_costs = [r['avg_cost'] for r in results]

        line2, = ax2.plot(
            v_vals,
            avg_costs,
            marker='s',
            markersize=6,
            color=color2,
            linewidth=2.5,
            label='Avg Operating Cost'
        )

        ax2.tick_params(axis='y', labelcolor=color2)

        # =========================
        # SLO 線（保留但不干擾主視覺）
        # =========================
        slo_line = ax1.axhline(
            y=6.0,
            color='gray',
            linestyle='dashdot',
            linewidth=2,
            label='SLO Target (6.0s)'
        )

        # =========================
        # Legend（完全對齊你第二張圖）
        # =========================
        lines = [line1, line2, slo_line]
        labels = [l.get_label() for l in lines]

        ax1.legend(
            lines,
            labels,
            loc='center right',
            fontsize=11,
            frameon=True,
            shadow=True
        )

        fig.tight_layout()

        output_path = "lyapunov_v_tradeoff.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"\n[Master] 🎉 實驗完成！圖表已儲存至 {output_path}")

if __name__ == "__main__":
    main()
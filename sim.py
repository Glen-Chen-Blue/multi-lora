import os
import json
import time
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from copy import deepcopy

# 匯入原本的設定
from config import (
    FIXED_INPUT_LEN, FIXED_OUTPUT_LEN, MERGED_CAPACITY, UNMERGED_CAPACITY, MAX_CPU_LORAS,
    SIM_LOAD_DELAY, SIM_PREFILL_BASE_TIME, SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    MERGE_SPEED_MULTIPLIER, SCHEDULER_OVERHEAD, T_MAX, EPSILON, PSI_DROP,
    SCALE_UP_DROP_THRESHOLD, SCALE_DOWN_SURPLUS_THRESHOLD, NETWORK_SIM_PARAMS,
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB, COST_INST_LOCAL, COST_NET_TRAFFIC,
    COST_DROP_PENALTY, LORA_SIZE_GB, DISK_CAPACITY_GB, T_MAX_SLO, SWAP_EPSILON,
    SP1_INTERVAL_SECONDS, SP2_INTERVAL_SECONDS, SIMULATION_DATA_CSV_PATH
)

# =========================================================================
# 1. 核心：離散事件引擎 (Discrete-Event Simulator Engine)
# =========================================================================
class Event:
    def __init__(self, time, event_type, target, data=None):
        self.time = time
        self.type = event_type
        self.target = target
        self.data = data

    def __lt__(self, other):
        return self.time < other.time

class Simulator:
    def __init__(self):
        self.now = 0.0
        self.events = []

    def schedule(self, delay, event_type, target, data=None):
        heapq.heappush(self.events, Event(self.now + delay, event_type, target, data))

    def run(self, until):
        while self.events and self.events[0].time <= until:
            event = heapq.heappop(self.events)
            self.now = event.time
            event.target.handle_event(event)

# =========================================================================
# 2. 引擎層：對應 _multilora_system.py (完全還原 Token-by-Token 邏輯)
# =========================================================================
class DesMultiLoRAEngine:
    def __init__(self, sim, node_id):
        self.sim = sim
        self.node_id = node_id
        
        self.merged_capacity = MERGED_CAPACITY
        self.unmerged_capacity = UNMERGED_CAPACITY
        self.adapter_slots = self.unmerged_capacity 
        self.max_cpu_loras = MAX_CPU_LORAS
        
        self.known_adapters = set()
        self.cpu_cache = {} 
        self.gpu_slots = {} 
        self.adapter_to_slot = {}
        
        self.request_queue = []
        self.running_queue = []
        self.current_merged_adapter = None
        
        self.is_running_step = False
        self.on_token = None
        self.on_finish = None

    def update_known_adapters(self, adapters):
        new_set = set(adapters)
        current = list(self.known_adapters)
        for aid in current:
            if aid not in new_set:
                self.known_adapters.remove(aid)
                self.cpu_cache.pop(aid, None)
                if aid in self.adapter_to_slot:
                    self._evict_from_gpu(aid)
        self.known_adapters = new_set

    def _evict_from_gpu(self, adapter_id):
        if adapter_id in self.adapter_to_slot:
            slot = self.adapter_to_slot.pop(adapter_id)
            del self.gpu_slots[slot]

    def _load_adapter_to_slot(self, adapter_id, slot_id, load_delay_accumulator):
        if adapter_id not in self.cpu_cache:
            load_delay_accumulator[0] += SIM_LOAD_DELAY
            if len(self.cpu_cache) >= self.max_cpu_loras:
                self.cpu_cache.pop(next(iter(self.cpu_cache))) # 簡單 FIFO
            self.cpu_cache[adapter_id] = True

        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id:
            self._evict_from_gpu(self.gpu_slots[slot_id])
            
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id

    def _ensure_adapters_resident(self, required_adapters, load_delay_accumulator):
        for aid in required_adapters:
            if aid not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    raise RuntimeError(f"VRAM Capacity logic failure for {aid}.")
                self._load_adapter_to_slot(aid, available_slots[0], load_delay_accumulator)

    def merge_adapter(self, adapter_id, force=False):
        load_delay = [0.0]
        if adapter_id not in self.adapter_to_slot:
            available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
            if not available_slots:
                self._cleanup_unused_adapters()
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
            if available_slots:
                self._load_adapter_to_slot(adapter_id, available_slots[0], load_delay)
        
        if self.current_merged_adapter != adapter_id:
            self.current_merged_adapter = adapter_id

    def unmerge_all(self):
        self.current_merged_adapter = None

    def add_request(self, prompt, adapter_id, request_id, max_new_tokens=256):
        self.request_queue.append({
            "request_id": request_id,
            "adapter_id": adapter_id,
            "past_key_values": None,
            "tokens_gen": 0,
            "max_new_tokens": FIXED_OUTPUT_LEN,
            "done": False,
            "arrival_time": self.sim.now
        })
        self.wakeup()

    def is_idle(self):
        return len(self.request_queue) == 0 and len(self.running_queue) == 0

    def _cleanup_unused_adapters(self):
        needed_aids = {r["adapter_id"] for r in self.running_queue + self.request_queue}
        if self.current_merged_adapter: needed_aids.add(self.current_merged_adapter)
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in needed_aids:
                self._evict_from_gpu(aid)

    def full_reset(self):
        self.request_queue.clear()
        self.running_queue.clear()
        self.current_merged_adapter = None
        self.gpu_slots.clear()
        self.adapter_to_slot.clear()
        self.known_adapters.clear()
        self.cpu_cache.clear()
        self.is_running_step = False

    def wakeup(self):
        if not self.is_running_step:
            self.is_running_step = True
            self.sim.schedule(0.0, "ENGINE_STEP", self)

    def handle_event(self, event):
        if event.type == "ENGINE_STEP":
            self._execute_step()
        elif event.type == "ENGINE_FINISH_PHASE":
            self._process_outputs(event.data)
            self.sim.schedule(0.0, "ENGINE_STEP", self) # 繼續下一輪

    def _execute_step(self):
        self._cleanup_unused_adapters()
        self.running_queue = [r for r in self.running_queue if not r["done"]]

        if not self.running_queue and not self.request_queue:
            self.is_running_step = False
            return

        multiplier = 1.0
        target_group = []
        required_adapters = []
        load_delay_accumulator = [0.0]

        if self.current_merged_adapter:
            target_group = [r for r in self.running_queue if r["adapter_id"] == self.current_merged_adapter]
            remaining_slots = self.merged_capacity - len(target_group)
            if remaining_slots > 0 and self.request_queue:
                move_indices = [i for i, req in enumerate(self.request_queue) if req["adapter_id"] == self.current_merged_adapter][:remaining_slots]
                for i in sorted(move_indices, reverse=True):
                    req = self.request_queue.pop(i)
                    self.running_queue.append(req)
                    target_group.append(req)
            if not target_group:
                self.is_running_step = False
                return
            multiplier = MERGE_SPEED_MULTIPLIER
        else:
            current_reqs = len(self.running_queue)
            active_loras = {r["adapter_id"] for r in self.running_queue}
            current_lora_count = len(active_loras)
            
            idx_to_remove = []
            for i, req in enumerate(self.request_queue):
                aid = req["adapter_id"]
                new_lora_cost = 1 if aid not in active_loras else 0
                if (current_reqs + 1) + (current_lora_count + new_lora_cost) <= self.unmerged_capacity:
                    self.running_queue.append(req)
                    current_reqs += 1
                    if new_lora_cost > 0:
                        active_loras.add(aid)
                        current_lora_count += 1
                    idx_to_remove.append(i)
                else:
                    break
            
            for i in reversed(idx_to_remove):
                self.request_queue.pop(i)

            if not self.running_queue:
                self.is_running_step = False
                return
            target_group = self.running_queue
            required_adapters = sorted(list({r["adapter_id"] for r in self.running_queue}))

        # Phase 2 & 3: 計算總推論與 I/O 延遲
        step_sleep_time = 0.0
        
        # 處理載入 CPU
        for aid in required_adapters:
            if aid not in self.cpu_cache:
                load_delay_accumulator[0] += SIM_LOAD_DELAY
                if len(self.cpu_cache) >= self.max_cpu_loras: self.cpu_cache.pop(next(iter(self.cpu_cache)))
                self.cpu_cache[aid] = True

        if not self.current_merged_adapter:
            self._ensure_adapters_resident(required_adapters, load_delay_accumulator)
        
        step_sleep_time += load_delay_accumulator[0]

        prefill_reqs = [r for r in target_group if r["past_key_values"] is None]
        decode_reqs = [r for r in target_group if r["past_key_values"] is not None]

        if prefill_reqs:
            step_sleep_time += (SIM_PREFILL_BASE_TIME * len(prefill_reqs)) * multiplier
            for r in prefill_reqs:
                r["past_key_values"] = True
                decode_reqs.append(r)
        
        if decode_reqs:
            step_sleep_time += (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * len(decode_reqs)) * multiplier

        # 模擬時間推進
        if step_sleep_time > 0:
            self.sim.schedule(step_sleep_time, "ENGINE_FINISH_PHASE", self, decode_reqs)
        else:
            self._process_outputs(decode_reqs)
            self.sim.schedule(0.0, "ENGINE_STEP", self)

    def _process_outputs(self, reqs):
        for req in reqs:
            req["tokens_gen"] += 1
            if self.on_token: self.on_token(req["request_id"])
            if req["tokens_gen"] >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: self.on_finish(req["request_id"], "finished")

# =========================================================================
# 3. 節點層：VirtualNode & ControlNode
# =========================================================================
class VirtualNodeState:
    def __init__(self, compute_node):
        self.node = compute_node
        self.url = compute_node.node_id
        self.mode = compute_node.engine.current_merged_adapter and "merge" or "unmerge"
        self.running_batch = len(compute_node.engine.running_queue) + len(compute_node.engine.request_queue)
        self.merged_adapter = compute_node.engine.current_merged_adapter
        self.active_loras = set({r["adapter_id"] for r in compute_node.engine.running_queue + compute_node.engine.request_queue})
        self.loaded_adapters = set(compute_node.engine.cpu_cache.keys())
        self.capacity_merged = MERGED_CAPACITY
        self.capacity_unmerged_base = UNMERGED_CAPACITY
        self.request_set = [{"adapter_id": r["adapter_id"], "remaining_tokens": r["max_new_tokens"] - r["tokens_gen"]} for r in compute_node.engine.running_queue + compute_node.engine.request_queue]

    def get_free_slots(self, target_lora):
        if self.mode == "merge":
            return max(0, self.capacity_merged - self.running_batch) if self.merged_adapter == target_lora else 0
        current_cost = self.running_batch + len(self.active_loras)
        margin = self.capacity_unmerged_base - current_cost
        if target_lora not in self.active_loras:
            return (margin - 1) if margin >= 2 else 0 
        return max(0, margin)

    def commit_request(self, target_lora):
        self.running_batch += 1
        self.active_loras.add(target_lora)

class ComputeNodeServer:
    def __init__(self, sim, node_id):
        self.node_id = node_id
        self.status = "active"
        self.engine = DesMultiLoRAEngine(sim, node_id)
        
        # 綁定 Callbacks
        self.control_node = None
        self.engine.on_token = self._on_token
        self.engine.on_finish = self._on_finish
        self.first_token_flags = {}

    def _on_token(self, rid):
        if rid not in self.first_token_flags and self.control_node:
            self.first_token_flags[rid] = True
            req = self.control_node.active_requests[rid]
            ttft = self.engine.sim.now - req["arrival_time"]
            self.control_node.record_ttft(ttft, req["is_delegated"])

    def _on_finish(self, rid, reason):
        if self.control_node:
            req = self.control_node.active_requests.pop(rid, None)
            if req:
                self.control_node.record_finish(req, elapsed=self.engine.sim.now - req["arrival_time"])

class ControlNodeServer:
    def __init__(self, sim, cluster_name, efo, strategy="dynamic"):
        self.sim = sim
        self.cluster_name = cluster_name
        self.efo = efo
        self.strategy = strategy
        self.nodes = [ComputeNodeServer(sim, f"{cluster_name}-n{i+1}") for i in range(3)]
        for n in self.nodes: n.control_node = self
        
        self.local_loras = set()
        self.z_debt = 0.0
        self.system_paused = False
        
        self.request_queues = defaultdict(deque)
        self.global_request_list = []
        self.active_requests = {}
        
        # 統計
        self.metrics = {"local_completed": 0, "offload_completed": 0, "dropped": 0, "offload_out": 0, "ttft_records": []}
        
        # 啟動背景迴圈
        self.sim.schedule(0.5, "SCHEDULER_LOOP", self)

    def record_ttft(self, ttft, is_delegated):
        self.metrics["ttft_records"].append(ttft)
        self.efo.global_stats["ttft_records"].append(ttft)

    def record_finish(self, req, elapsed):
        if req["is_delegated"]: self.metrics["offload_completed"] += 1
        else: self.metrics["local_completed"] += 1
        self.efo.global_stats["finished"] += 1

    def apply_sp1_and_reset(self, loras):
        self.system_paused = True
        self.pending_sp1_loras = loras
        self.sim.schedule(0.5, "CHECK_DRAIN", self)

    def handle_event(self, event):
        if event.type == "SCHEDULER_LOOP":
            self.scheduler_loop()
            self.sim.schedule(0.5, "SCHEDULER_LOOP", self)
        elif event.type == "CHECK_DRAIN":
            if len(self.global_request_list) == 0 and all(len(n.engine.running_queue) == 0 for n in self.nodes):
                for n in self.nodes: n.engine.full_reset()
                self.local_loras = set(self.pending_sp1_loras)
                self.system_paused = False
                self.efo.sp1_ack_count += 1
                if self.efo.sp1_ack_count == len(self.efo.clusters):
                    pass # EFO SP1 完畢
            else:
                self.sim.schedule(0.5, "CHECK_DRAIN", self)
        elif event.type == "RECEIVE_REQUEST":
            self.send_request(event.data)

    def predict_cluster_ttft(self, v_nodes, target_lora, global_pending_ahead):
        # [完全還原原本的預測公式]
        node_scores, total_free, cluster_concurrent_capacity, has_merged_node = [], 0, 0, False
        for node in v_nodes:
            is_merge = (node.mode == "merge" and node.merged_adapter == target_lora)
            if node.mode == "merge" and not is_merge: continue 
            if is_merge: has_merged_node = True
            
            is_in_vram = (node.mode == "unmerge" and target_lora in node.active_loras)
            is_in_cpu = (node.mode == "unmerge" and target_lora in node.loaded_adapters)
            is_empty = (node.mode == "unmerge" and len(node.active_loras) == 0)
            
            free_slots = node.get_free_slots(target_lora)
            if is_merge: cluster_concurrent_capacity += node.capacity_merged
            elif node.mode == "unmerge": cluster_concurrent_capacity += max(0, node.capacity_unmerged_base - 1)

            if free_slots > 0:
                score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, free_slots)
                node_scores.append({"score": score, "free": free_slots})
                total_free += free_slots

        my_position = global_pending_ahead + 1

        if my_position <= total_free:
            node_scores.sort(key=lambda x: x["score"], reverse=True)
            allocated, landing_score, take_at_landing = 0, None, 0
            for ns in node_scores:
                take = min(ns["free"], my_position - allocated)
                allocated += take
                if allocated == my_position:
                    landing_score, take_at_landing = ns["score"], take
                    break
            is_merge_landing = landing_score[0] == 1
            is_in_cpu_landing = landing_score[2] == 1
            multiplier = MERGE_SPEED_MULTIPLIER if is_merge_landing else 1.0
            load_delay = 0.0 if (is_in_cpu_landing or is_merge_landing) else SIM_LOAD_DELAY
            prefill_time = SIM_PREFILL_BASE_TIME * take_at_landing * multiplier
            return SCHEDULER_OVERHEAD + load_delay + prefill_time
        else:
            needed_to_finish = my_position - total_free
            if has_merged_node:
                merge_node = next((n for n in v_nodes if n.mode == "merge"), v_nodes[0])
                assumed_batch = merge_node.capacity_merged 
                multiplier = MERGE_SPEED_MULTIPLIER
            else:
                assumed_batch = max(1, v_nodes[0].capacity_unmerged_base - 2)
                multiplier = 1.0
                
            dynamic_decode_speed = (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * assumed_batch) * multiplier
            if cluster_concurrent_capacity == 0: 
                cluster_concurrent_capacity = v_nodes[0].capacity_merged if has_merged_node else v_nodes[0].capacity_unmerged_base
                
            full_cycles = needed_to_finish // cluster_concurrent_capacity
            remainder = needed_to_finish % cluster_concurrent_capacity
            
            all_remains = []
            for node in v_nodes:
                if (node.mode == "merge" and node.merged_adapter == target_lora) or (node.mode == "unmerge"):
                    all_remains.extend([r["remaining_tokens"] for r in node.request_set])
            
            if not all_remains: current_wait = 1.0
            else:
                all_remains.sort()
                idx = min(len(all_remains) - 1, max(0, remainder - 1))
                current_wait = all_remains[idx] * dynamic_decode_speed
                
            total_wait_time = current_wait + (full_cycles * 256 * dynamic_decode_speed)
            return SCHEDULER_OVERHEAD + total_wait_time + (SIM_PREFILL_BASE_TIME * multiplier)

    def send_request(self, req):
        if self.system_paused:
            return # Drop
            
        rid = req["request_id"]
        meta = self.efo.lora_metadata.get(req["adapter_id"], {})
        is_local = (meta.get("type") == "local")
        
        # Sovereignty
        if not meta or (is_local and meta.get("cluster") != self.cluster_name):
            self.efo.global_stats["dropped"] += 1; return

        valid_subs = [req["adapter_id"]] + [s for s in meta.get("substitutes", []) if s in self.local_loras]
        actual_valid = [s for s in valid_subs if s in self.local_loras]

        target_ttft = T_MAX - (req.get("network_delay", 0.0) * 2)
        v_nodes = [VirtualNodeState(n) for n in self.nodes if n.status == "active"]
        
        best_ttft = 999.0
        global_pending = len(self.global_request_list)
        if v_nodes and actual_valid:
            for aid in actual_valid:
                ttft = self.predict_cluster_ttft(v_nodes, aid, global_pending)
                if ttft < best_ttft: best_ttft = ttft

        s_eff = 1.0 if (best_ttft <= target_ttft and actual_valid) else -1.0

        if s_eff < 0:
            if req["is_delegated"] or is_local:
                self.z_debt = max(0.0, self.z_debt + 1.0 - EPSILON)
                self.metrics["dropped"] += 1
                self.efo.global_stats["dropped"] += 1
                return

            target_cluster = self.efo.get_offload_target(self.cluster_name, req["adapter_id"])
            if target_cluster:
                self.z_debt = max(0.0, self.z_debt + 1.0 - EPSILON)
                self.metrics["offload_out"] += 1
                req["is_delegated"] = True
                delay = self.efo.network_params.get((self.cluster_name, target_cluster), (50,0,0))[0] / 1000.0
                req["network_delay"] = delay
                self.sim.schedule(delay, "RECEIVE_REQUEST", self.efo.control_nodes[target_cluster], req)
                return
            else:
                self.z_debt = max(0.0, self.z_debt + 1.0 - EPSILON)
                self.metrics["dropped"] += 1
                self.efo.global_stats["dropped"] += 1
                return
        else:
            self.z_debt = max(0.0, self.z_debt - EPSILON)

        self.request_queues[req["adapter_id"]].append(req)
        self.global_request_list.append(req)

    def scheduler_loop(self):
        if self.system_paused: return
        
        v_nodes = [VirtualNodeState(n) for n in self.nodes if n.status == "active"]
        if not v_nodes: return

        # Merge/Unmerge 狀態切換
        MERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 1)
        UNMERGE_THRESHOLD = max(1, UNMERGED_CAPACITY - 2)
        unmerged_count = sum(1 for n in v_nodes if n.mode == "unmerge")

        for v in v_nodes:
            if v.mode == "unmerge" and unmerged_count > 1 and v.running_batch >= MERGE_THRESHOLD and len(v.active_loras) == 1:
                aid = next(iter(v.active_loras))
                if len(self.request_queues[aid]) > 0:
                    v.node.engine.merge_adapter(aid)
                    unmerged_count -= 1
            elif v.mode == "merge" and v.running_batch < UNMERGE_THRESHOLD:
                if len(self.request_queues[v.merged_adapter]) == 0:
                    v.node.engine.unmerge_all()
                    unmerged_count += 1

        # 分派邏輯
        dispatched_any = True
        while dispatched_any and self.global_request_list:
            dispatched_any = False
            for req in list(self.global_request_list):
                target_aid = req["original_aid"]
                meta = self.efo.lora_metadata.get(target_aid, {})
                
                # [實驗切換邏輯]
                valid_aids = [target_aid]
                if self.strategy != "random" and self.strategy != "greedy": # Dynamic Strategy
                    valid_aids += [s for s in meta.get("substitutes", []) if s in self.local_loras]
                valid_aids = [aid for aid in valid_aids if aid in self.local_loras]
                if not valid_aids: valid_aids = [target_aid]

                best_plan = None
                for aid in valid_aids:
                    candidate_reqs = []
                    for q_aid, q_reqs in self.request_queues.items():
                        if aid == q_aid or aid in meta.get("substitutes", []):
                            candidate_reqs.extend(q_reqs)
                    if not candidate_reqs: continue
                    candidate_reqs.sort(key=lambda x: x["arrival_time"])
                    
                    for v in v_nodes:
                        free_slots = v.get_free_slots(aid)
                        if free_slots <= 0: continue
                        can_take = min(free_slots, len(candidate_reqs))
                        
                        is_merge = (v.mode == "merge" and v.merged_adapter == aid)
                        is_in_vram = (v.mode == "unmerge" and aid in v.active_loras)
                        is_in_cpu = (v.mode == "unmerge" and aid in v.loaded_adapters)
                        is_empty = (v.mode == "unmerge" and len(v.active_loras) == 0)
                        
                        # [策略]
                        if self.strategy == "random":
                            score = np.random.uniform(0, 1)
                        elif self.strategy == "greedy":
                            score = free_slots # 只看空位
                        else:
                            score = (1 if is_merge else 0, 1 if is_in_vram else 0, 1 if is_in_cpu else 0, 1 if is_empty else 0, can_take)
                        
                        if best_plan is None or score > best_plan["score"]:
                            best_plan = {"node": v.node, "lora": aid, "requests": candidate_reqs[:can_take], "score": score}
                
                if best_plan:
                    node, aid, reqs = best_plan["node"], best_plan["lora"], best_plan["requests"]
                    for r in reqs:
                        q_aid = r["original_aid"]
                        self.request_queues[q_aid] = deque([x for x in self.request_queues[q_aid] if x["request_id"] != r["request_id"]])
                        self.global_request_list = [x for x in self.global_request_list if x["request_id"] != r["request_id"]]
                        
                        r["adapter_id"] = aid
                        self.active_requests[r["request_id"]] = r
                        node.engine.add_request(prompt="test", adapter_id=aid, request_id=r["request_id"])
                    dispatched_any = True
                    break 

# =========================================================================
# 4. 全域層：EFO_server.py 的精確移植
# =========================================================================
class EFOServer:
    def __init__(self, sim, meta_path):
        self.sim = sim
        self.clusters = ["cluster_1", "cluster_2", "cluster_3"]
        self.control_nodes = {}
        self.network_params = NETWORK_SIM_PARAMS
        
        with open(meta_path, "r", encoding="utf-8") as f:
            self.lora_metadata = json.load(f)
            
        self.predicted_demand = defaultdict(lambda: defaultdict(float))
        self.global_lora_disk_inventory = defaultdict(list)
        self.sp1_ack_count = 0
        
        self.global_stats = {"sent": 0, "finished": 0, "dropped": 0, "ttft_records": []}

    def load_forecasts(self):
        df = pd.read_csv(SIMULATION_DATA_CSV_PATH)
        df["arrival_sec"] = df["arrive_timestamp"].astype(float) - 86400*2
        for _, row in df.iterrows():
            cluster = str(row["cluster"]).strip()
            if cluster not in self.clusters: continue
            lora_id = f"LoRA_{int(float(row['lora_id']))}"
            step = int(row["arrival_sec"] // SP1_INTERVAL_SECONDS)
            if step >= 0:
                self.predicted_demand[(step, cluster)][lora_id] += 1.0

    def get_offload_target(self, src_cluster, adapter_id):
        best_cluster, best_score = None, float('inf')
        valid_aids = [adapter_id] + self.lora_metadata.get(adapter_id, {}).get("substitutes", [])
        for c, cnode in self.control_nodes.items():
            if c == src_cluster or cnode.z_debt >= PSI_DROP * 0.9: continue
            if any(aid in cnode.local_loras for aid in valid_aids):
                delay = self.network_params.get((src_cluster, c), (50,0,0))[0] / 1000.0
                if delay < best_score:
                    best_score, best_cluster = delay, c
        return best_cluster

    def handle_event(self, event):
        if event.type == "TRIGGER_SP1":
            self.run_sp1(event.data)

    def run_sp1(self, step_id):
        # [完全拷貝 EFO_server.py Phase 1 & 2 CSG-Swap 演算法]
        cluster_targets = {}
        serves_map = defaultdict(set)
        for lid in self.lora_metadata.keys(): serves_map[lid].add(lid)
        for lid, info in self.lora_metadata.items():
            for parent in info.get("substitutes", []):
                serves_map[parent].add(lid)

        def is_covered(target_id, stored_set):
            if target_id in stored_set: return True
            subs = self.lora_metadata.get(target_id, {}).get("substitutes", [])
            for s in subs:
                if s in stored_set: return True
            return False

        def calc_marginal_demand(cluster, candidate_id, current_set):
            total_new_demand = 0.0
            for tid in serves_map.get(candidate_id, set()):
                if not is_covered(tid, current_set):
                    total_new_demand += self.predicted_demand[(step_id, cluster)].get(tid, 0.0)
            return total_new_demand

        # Phase 1: Local Provisioning
        for c in self.clusters:
            target_disk = set()
            mandatory_set = set()
            c_valid = [l for l, info in self.lora_metadata.items() if info.get("type") == "global" or info.get("cluster") == c]
            
            for l in c_valid:
                if self.lora_metadata[l].get("type") == "local":
                    mandatory_set.add(l); target_disk.add(l)
            
            current_disk = set(self.global_lora_disk_inventory.get(c, []))
            for l in current_disk:
                if l in mandatory_set or l not in c_valid: continue
                if is_covered(l, target_disk.union(current_disk) - {l}): continue
                
                gain_per_req = max(0.0, COST_DROP_PENALTY - COST_INST_LOCAL) # 簡化
                u_retention = (self.predicted_demand[(step_id, c)].get(l, 0.0) * gain_per_req) - (LORA_SIZE_GB * COST_STORE_PER_GB)
                if u_retention >= 0: target_disk.add(l)

            candidates = [l for l in c_valid if l not in target_disk]
            capacity = int(DISK_CAPACITY_GB / LORA_SIZE_GB)
            
            while True:
                best_cand, max_u = None, -float('inf')
                for cand in candidates:
                    new_dem = calc_marginal_demand(c, cand, target_disk)
                    net_u = (new_dem * COST_DROP_PENALTY) - (LORA_SIZE_GB * (COST_STORE_PER_GB + COST_DOWNLOAD_PER_GB))
                    if net_u > max_u: max_u, best_cand = net_u, cand
                
                if not best_cand or max_u <= 0: break
                
                if len(target_disk) < capacity:
                    target_disk.add(best_cand); candidates.remove(best_cand)
                else:
                    victim, min_loss = None, float('inf')
                    swappable = [t for t in target_disk if t not in mandatory_set]
                    if not swappable: break
                    for t in swappable:
                        loss_val = (calc_marginal_demand(c, t, target_disk - {t}) * COST_DROP_PENALTY) - (LORA_SIZE_GB * COST_STORE_PER_GB)
                        if loss_val < min_loss: min_loss, victim = loss_val, t
                    
                    if max_u > min_loss + SWAP_EPSILON:
                        target_disk.remove(victim); target_disk.add(best_cand)
                        candidates.remove(best_cand); candidates.append(victim)
                    else: break
            cluster_targets[c] = list(target_disk)

        # Apply and Reset (Block in original, events here)
        self.sp1_ack_count = 0
        for c, targets in cluster_targets.items():
            self.control_nodes[c].apply_sp1_and_reset(targets)

        self.sim.schedule(SP1_INTERVAL_SECONDS, "TRIGGER_SP1", self, step_id + 1)


# =========================================================================
# 5. 執行器 (Player) & 主程式
# =========================================================================
def run_experiment(config):
    print(f"\n{'-'*60}\n🚀 Starting Experiment: {config['name']} (Strategy: {config['strategy']})\n{'-'*60}")
    
    sim = Simulator()
    efo = EFOServer(sim, f"./information/{config['meta_path']}")
    
    for c in efo.clusters:
        efo.control_nodes[c] = ControlNodeServer(sim, c, efo, strategy=config['strategy'])
        
    efo.load_forecasts()
    sim.schedule(0.0, "TRIGGER_SP1", efo, 0)
    
    df = pd.read_csv(SIMULATION_DATA_CSV_PATH)
    df["arrival_sec"] = df["arrive_timestamp"].astype(float) - 86400*2
    run_duration = SP1_INTERVAL_SECONDS * 8
    df = df[(df["arrival_sec"] >= 0) & (df["arrival_sec"] <= run_duration)]
    
    for _, row in df.iterrows():
        c = str(row["cluster"]).strip()
        if c not in efo.clusters: continue
        req = {
            "request_id": str(uuid.uuid4()),
            "adapter_id": f"LoRA_{int(float(row['lora_id']))}",
            "original_aid": f"LoRA_{int(float(row['lora_id']))}",
            "arrival_time": row["arrival_sec"],
            "is_delegated": False
        }
        efo.global_stats["sent"] += 1
        sim.schedule(row["arrival_sec"], "RECEIVE_REQUEST", efo.control_nodes[c], req)
        
    print(f"📥 Loaded {efo.global_stats['sent']} requests. Simulating at Max CPU Speed...")
    sim.run(until=run_duration + 100)
    
    # [輸出完全對齊 test_simulation.py 的報表]
    stats = efo.global_stats
    print(f"\n=== Summary: Sent {stats['sent']} / Fin {stats['finished']} / Drop {stats['dropped']} / Err 0 ===")
    if stats['ttft_records']:
        avg = sum(stats['ttft_records']) / len(stats['ttft_records'])
        p95 = sorted(stats['ttft_records'])[int(len(stats['ttft_records']) * 0.95)]
        print(f"Average TTFT: {avg:.4f}s")
        print(f"P95 TTFT: {p95:.4f}s")

if __name__ == "__main__":
    import uuid
    # 對應 start_experinment3_1 ~ 3_4.sh 的環境參數
    experiments = [
        {"name": "Exp3_1 (Dynamic, with Sub)", "meta_path": "lora_metadata.json", "strategy": "dynamic"},
        {"name": "Exp3_2 (Dynamic, NO Sub)",   "meta_path": "lora_metadata_without_substitutes.json", "strategy": "dynamic"},
        {"name": "Exp3_3 (Random, NO Sub)",    "meta_path": "lora_metadata_without_substitutes.json", "strategy": "random"},
        {"name": "Exp3_4 (Greedy, NO Sub)",    "meta_path": "lora_metadata_without_substitutes.json", "strategy": "greedy"},
    ]
    
    for cfg in experiments:
        run_experiment(cfg)
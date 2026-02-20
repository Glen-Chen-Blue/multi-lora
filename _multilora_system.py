import os
import time
import threading
from typing import Dict, Optional, Any, List, Set, Callable
from collections import OrderedDict

# ============================================================
# Simulated Tokenizer
# ============================================================
class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def decode(self, tokens, skip_special_tokens=True):
        return "word " * len(tokens)

# ============================================================
# Multi-LoRA Engine Core (Pure Simulation Mode)
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, device: Optional[str] = None, torch_dtype: Any = None, enable_monitor: bool = True, adapter_fetcher: Optional[Callable[[str], bytes]] = None):
        self.device = "simulated_cpu"
        self.merged_capacity = 15
        self.unmerged_capacity = 12
        self.adapter_slots = self.unmerged_capacity 
        self.max_cpu_loras = 30
        self.adapter_fetcher = adapter_fetcher 

        self.FIXED_INPUT_LEN = 512
        self.FIXED_OUTPUT_LEN = 256

        print(f"🟢 [Engine] Running in PURE SIMULATION MODE (No GPU/PyTorch required).")
        
        self.tokenizer = DummyTokenizer()
        self.known_adapters: Set[str] = set()
        self.cpu_cache: OrderedDict[str, bool] = OrderedDict() 
        self.gpu_slots: Dict[int, str] = {} 
        self.adapter_to_slot: Dict[str, int] = {} 
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        
        self.current_merged_adapter: Optional[str] = None
        self.lock = threading.RLock()
        
        self.on_token = None
        self.on_finish = None

    def update_known_adapters(self, adapters: List[str]):
        new_set = set(adapters)
        current = list(self.known_adapters)
        for aid in current:
            if aid not in new_set:
                self.known_adapters.remove(aid)
                if aid in self.cpu_cache: 
                    del self.cpu_cache[aid]
                if aid in self.adapter_to_slot:
                    self._evict_from_gpu(aid)
        self.known_adapters = new_set

    def _ensure_cpu_loaded(self, adapter_id: str):
        # [修改] 先用鎖檢查是否需要載入
        with self.lock:
            needs_load = adapter_id not in self.cpu_cache
            
        if needs_load:
            # ⏳ [修改] 在鎖「外面」模擬延遲，這樣 /metrics 就不會被卡住！
            time.sleep(0.200) 
            
            with self.lock:
                # Double-check 避免被其他 Thread 搶先載入
                if adapter_id not in self.cpu_cache:
                    while len(self.cpu_cache) >= self.max_cpu_loras:
                        evicted_aid, _ = self.cpu_cache.popitem(last=False)
                    self.cpu_cache[adapter_id] = True

    def _evict_from_gpu(self, adapter_id: str):
        if adapter_id in self.adapter_to_slot:
            slot = self.adapter_to_slot.pop(adapter_id)
            del self.gpu_slots[slot]
            self.slot_lru.move_to_end(slot, last=False) 

    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        self._ensure_cpu_loaded(adapter_id)
        if adapter_id in self.cpu_cache:
            self.cpu_cache.move_to_end(adapter_id)
        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id: 
            self._evict_from_gpu(self.gpu_slots[slot_id])
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id

    def _ensure_adapters_resident(self, required_adapters: List[str]):
        for aid in required_adapters:
            if aid not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    raise RuntimeError(f"No Simulated VRAM slots available for {aid}.")
                self._load_adapter_to_slot(aid, available_slots[0])

    def merge_adapter(self, adapter_id: str, force: bool = False):
        self._ensure_cpu_loaded(adapter_id)
        with self.lock:
            conflicting_reqs = [r for r in self.running_queue + self.request_queue if r["adapter_id"] != adapter_id]
            if conflicting_reqs and not force:
                raise RuntimeError(f"Cannot merge {adapter_id}: Found {len(conflicting_reqs)} conflicting requests from other LoRAs.")

            if adapter_id not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    self._cleanup_unused_adapters() 
                    available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                    if not available_slots:
                        raise RuntimeError("No VRAM slots available to load adapter for merging.")
                self._load_adapter_to_slot(adapter_id, available_slots[0])
            
            if self.current_merged_adapter and self.current_merged_adapter != adapter_id:
                self.unmerge_all()
            
            if self.current_merged_adapter != adapter_id:
                print(f"🔀 [Simulated Seamless] Merging {adapter_id} into base model on-the-fly...")
                self.current_merged_adapter = adapter_id

    def unmerge_all(self):
        with self.lock:
            if self.current_merged_adapter:
                print(f"🔀 [Simulated Seamless] Unmerging {self.current_merged_adapter}...")
                self.current_merged_adapter = None

    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 256):
        with self.lock:
            self.request_queue.append({
                "request_id": str(request_id),
                "adapter_id": str(adapter_id),
                "seq_len": self.FIXED_INPUT_LEN,
                "past_key_values": None,
                "tokens_gen": [],
                "max_new_tokens": self.FIXED_OUTPUT_LEN,
                "done": False
            })

    def is_idle(self) -> bool:
        with self.lock: 
            return len(self.request_queue) == 0 and len(self.running_queue) == 0

    def _cleanup_unused_adapters(self):
        active_aids = {r["adapter_id"] for r in self.running_queue}
        pending_aids = {r["adapter_id"] for r in self.request_queue}
        needed_aids = active_aids.union(pending_aids)
        if self.current_merged_adapter:
            needed_aids.add(self.current_merged_adapter)
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in needed_aids:
                self._evict_from_gpu(aid)

    def step(self) -> bool:
        # ==========================================
        # Phase 1: 鎖定狀態，挑選本次要執行的 Request
        # ==========================================
        with self.lock:
            self._cleanup_unused_adapters()
            self.running_queue = [r for r in self.running_queue if not r["done"]]

            if not self.running_queue and not self.request_queue:
                return False

            multiplier = 1.0
            required_adapters = []
            target_group = []

            if self.current_merged_adapter:
                target_group = [r for r in self.running_queue if r["adapter_id"] == self.current_merged_adapter]
                remaining_slots = self.merged_capacity - len(target_group)
                if remaining_slots > 0 and self.request_queue:
                    move_indices = []
                    for i, req in enumerate(self.request_queue):
                        if req["adapter_id"] == self.current_merged_adapter:
                            move_indices.append(i)
                            if len(move_indices) >= remaining_slots: break
                    for i in sorted(move_indices, reverse=True):
                        req = self.request_queue.pop(i)
                        self.running_queue.append(req)
                        target_group.append(req)

                if not target_group: return False
                multiplier = 0.8
            else:
                current_reqs = len(self.running_queue)
                active_loras = {r["adapter_id"] for r in self.running_queue}
                current_lora_count = len(active_loras)
                
                idx_to_remove = []
                for i, req in enumerate(self.request_queue):
                    aid = req["adapter_id"]
                    new_req_cost = 1
                    new_lora_cost = 1 if aid not in active_loras else 0
                    total_cost = (current_reqs + new_req_cost) + (current_lora_count + new_lora_cost)
                    
                    if total_cost <= self.unmerged_capacity:
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

                if not self.running_queue: return False
                target_group = self.running_queue
                required_adapters = sorted(list({r["adapter_id"] for r in self.running_queue}))

        # ==========================================
        # Phase 2: 載入 LoRA (在鎖的外面執行，不卡 API)
        # ==========================================
        for aid in required_adapters:
            self._ensure_cpu_loaded(aid)

        # ==========================================
        # Phase 3: 鎖定狀態，結算需要睡眠的時間
        # ==========================================
        with self.lock:
            # 防呆：可能在 Phase 2 期間被 reset 清空了
            if not target_group or all(r["done"] for r in target_group):
                return True

            if not self.current_merged_adapter:
                self._ensure_adapters_resident(required_adapters)
            
            prefill_reqs = [r for r in target_group if r["past_key_values"] is None]
            decode_reqs = [r for r in target_group if r["past_key_values"] is not None]
            
            step_sleep_time = 0.0
            
            if prefill_reqs:
                step_sleep_time += (0.050 * len(prefill_reqs)) * multiplier
                for r in prefill_reqs:
                    r["past_key_values"] = True
                    decode_reqs.append(r)
            
            if decode_reqs:
                batch_size = len(decode_reqs)
                step_sleep_time += (0.025 + 0.0012 * batch_size) * multiplier

        # ==========================================
        # Phase 4: 模擬 GPU 運算 (在鎖的外面執行，不卡 API)
        # ==========================================
        if step_sleep_time > 0:
            time.sleep(step_sleep_time)

        # ==========================================
        # Phase 5: 鎖定狀態，產出結果
        # ==========================================
        if decode_reqs:
            with self.lock:
                self._process_outputs(decode_reqs)

        return True

    def _process_outputs(self, reqs):
        for req in reqs:
            req["tokens_gen"].append(1)
            if self.on_token: 
                self.on_token(req["request_id"], req["tokens_gen"])
            if len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: 
                    self.on_finish(req["request_id"], "finished")
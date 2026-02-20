import os
import time
import threading
from typing import Dict, Optional, Any, List, Set, Callable
from collections import OrderedDict

# ============================================================
# Simulated Tokenizer
# ============================================================
class DummyTokenizer:
    """
    假 Tokenizer：不需要載入真實模型，只負責計算長度與假字串轉換
    供 compute_node_server 的 on_token 呼叫使用
    """
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        # 簡單回傳長度相符的 dummy list
        return [1] * len(text.split())

    def decode(self, tokens, skip_special_tokens=True):
        # 簡單回傳等比例的長度字串，讓前端看得到吐字
        return "word " * len(tokens)

# ============================================================
# Multi-LoRA Engine Core (Pure Simulation Mode)
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, device: Optional[str] = None, torch_dtype: Any = None, enable_monitor: bool = True, adapter_fetcher: Optional[Callable[[str], bytes]] = None):
        self.device = "simulated_cpu"
        
        # [Capacity Configuration]
        self.merged_capacity = 15
        self.unmerged_capacity = 12
        self.adapter_slots = self.unmerged_capacity 
        
        self.max_cpu_loras = 30
        self.adapter_fetcher = adapter_fetcher 

        # [Constraints]
        self.FIXED_INPUT_LEN = 512
        self.FIXED_OUTPUT_LEN = 256

        print(f"🟢 [Engine] Running in PURE SIMULATION MODE (No GPU/PyTorch required).")
        print(f"   - CPU Load Time: 200ms")
        print(f"   - Unmerged Prefill: 50ms per request | Decode: 25ms + 1.2*B ms")
        print(f"   - Merged Speed: 1.25x (Execution Time * 0.8)")

        self.tokenizer = DummyTokenizer()

        # [LRU System]
        self.known_adapters: Set[str] = set()
        self.cpu_cache: OrderedDict[str, bool] = OrderedDict() # 簡化為 boolean 標記
        
        # GPU Management (Simulated Slots)
        self.gpu_slots: Dict[int, str] = {} 
        self.adapter_to_slot: Dict[str, int] = {} 
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        
        self.current_merged_adapter: Optional[str] = None
        self.lock = threading.RLock()
        
        # Callbacks
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
        if adapter_id in self.cpu_cache:
            return

        try:
            # ⏳ [模擬延遲] 載入 LoRA 到 CPU 需要 200ms
            time.sleep(0.200) 
            
            while len(self.cpu_cache) >= self.max_cpu_loras:
                evicted_aid, _ = self.cpu_cache.popitem(last=False)

            # 模擬載入成功
            self.cpu_cache[adapter_id] = True
                
        except Exception as e:
            print(f"❌ [Engine] Failed to simulate loading {adapter_id}: {e}")
            raise e

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
        """
        Seamless Merge (Simulated)
        Convert the model to dedicated mode WITHOUT interrupting running requests.
        """
        self._ensure_cpu_loaded(adapter_id)
        
        with self.lock:
            # 🚀 [防卡死安全機制]
            # 檢查是否有不屬於目標 adapter_id 的請求存在於佇列中。
            conflicting_reqs = [r for r in self.running_queue + self.request_queue if r["adapter_id"] != adapter_id]
            if conflicting_reqs and not force:
                raise RuntimeError(f"Cannot merge {adapter_id}: Found {len(conflicting_reqs)} conflicting requests from other LoRAs.")

            # 確保該 Adapter 已經載入虛擬 VRAM
            if adapter_id not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    self._cleanup_unused_adapters() 
                    available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                    if not available_slots:
                        raise RuntimeError("No VRAM slots available to load adapter for merging.")
                
                self._load_adapter_to_slot(adapter_id, available_slots[0])
            
            # 如果已經有別的 Merged，先 Unmerge 它
            if self.current_merged_adapter and self.current_merged_adapter != adapter_id:
                self.unmerge_all()
            
            if self.current_merged_adapter != adapter_id:
                print(f"🔀 [Simulated Seamless] Merging {adapter_id} into base model on-the-fly...")
                self.current_merged_adapter = adapter_id
                # 不 Evict Slot，保留供 Unmerge 使用

    def unmerge_all(self):
        """
        Seamless Unmerge (Simulated)
        """
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
                "past_key_values": None, # None 代表這是一個新的 Request (尚未 Prefill)
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
        
        # [Protection] Merged Adapter is needed for Unmerge
        if self.current_merged_adapter:
            needed_aids.add(self.current_merged_adapter)
        
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in needed_aids:
                self._evict_from_gpu(aid)

    def step(self) -> bool:
        with self.lock:
            self._cleanup_unused_adapters()
            self.running_queue = [r for r in self.running_queue if not r["done"]]

            if not self.running_queue and not self.request_queue:
                return False

            if self.current_merged_adapter:
                # === [Merged Mode Path] ===
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

                self._execute_batch(target_group)
                return True

            else:
                # === [Unmerged Mode Path] ===
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

                required = sorted(list({r["adapter_id"] for r in self.running_queue}))
                self._ensure_adapters_resident(required)
                
                self._execute_batch(self.running_queue)
                return True

    def _execute_batch(self, target_group):
        """
        核心時間模擬邏輯 (已修正：Prefill 完緊接著 Decode 形成完整 Step)
        """
        prefill_reqs = [r for r in target_group if r["past_key_values"] is None]
        decode_reqs = [r for r in target_group if r["past_key_values"] is not None]

        # ⏳ Merge 的速度是 Unmerge 的 5/4 倍，因此花費時間為 4/5 (0.8)
        multiplier = 0.8 if self.current_merged_adapter else 1.0
        step_sleep_time = 0.0

        # =================================================
        # 1. Prefill 階段
        # =================================================
        if prefill_reqs:
            # ⏳ [模擬延遲] 每個進入的 Request 的 Prefill 為 50ms
            step_sleep_time += (0.050 * len(prefill_reqs)) * multiplier
            
            # 將剛 prefill 完的 request 標記並加入 decode_reqs
            # 確保它們在這個 step 中也能產出第一個 Token
            for r in prefill_reqs:
                r["past_key_values"] = True
                decode_reqs.append(r)

        # =================================================
        # 2. Decode 階段
        # =================================================
        if decode_reqs:
            batch_size = len(decode_reqs)
            # ⏳ [模擬延遲] Decode 時間：25 + 1.2 * batch_size ms
            step_sleep_time += (0.025 + 0.0012 * batch_size) * multiplier

        # =================================================
        # 3. 執行等待與產出
        # =================================================
        if step_sleep_time > 0:
            time.sleep(step_sleep_time)

        if decode_reqs:
            self._process_outputs(decode_reqs)

    def _process_outputs(self, reqs):
        for req in reqs:
            # 加入 1 個假 Token
            req["tokens_gen"].append(1)
            
            if self.on_token: 
                self.on_token(req["request_id"], req["tokens_gen"])
            
            # 檢查是否到達固定輸出長度 (256)
            if len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: 
                    self.on_finish(req["request_id"], "finished")
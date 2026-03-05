import os
import time
import threading
from typing import Dict, Optional, Any, List, Set, Callable
from collections import OrderedDict

# 匯入集中管理的設定
from config import (
    FIXED_INPUT_LEN, FIXED_OUTPUT_LEN,
    MERGED_CAPACITY, UNMERGED_CAPACITY, MAX_CPU_LORAS,
    SIM_LOAD_DELAY, SIM_PREFILL_BASE_TIME,
    SIM_DECODE_BASE_TIME, SIM_DECODE_SLOPE,
    MERGE_SPEED_MULTIPLIER
)

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
        # 簡單模擬：用空格分割長度，每個詞當作一個 token
        return [1] * len(text.split())

    def decode(self, tokens, skip_special_tokens=True):
        return "word " * len(tokens)

# ============================================================
# Multi-LoRA Engine Core (Pure Simulation Mode)
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, device: Optional[str] = None, torch_dtype: Any = None, enable_monitor: bool = True, adapter_fetcher: Optional[Callable[[str], bytes]] = None):
        self.device = "simulated_cpu"
        
        # [Capacity Configuration]
        self.merged_capacity = MERGED_CAPACITY
        self.unmerged_capacity = UNMERGED_CAPACITY
        self.adapter_slots = self.unmerged_capacity 
        
        self.max_cpu_loras = MAX_CPU_LORAS
        self.adapter_fetcher = adapter_fetcher 

        self.FIXED_INPUT_LEN = FIXED_INPUT_LEN
        self.FIXED_OUTPUT_LEN = FIXED_OUTPUT_LEN

        print(f"🟢 [Engine] Running in PURE SIMULATION MODE (No GPU/PyTorch required).")
        
        self.tokenizer = DummyTokenizer()
        
        # [LRU System]
        self.known_adapters: Set[str] = set()
        self.cpu_cache: OrderedDict[str, bool] = OrderedDict() # 模擬 Cache，Value 存 True 即可
        
        # GPU Management (Simulated)
        self.gpu_slots: Dict[int, str] = {} 
        self.adapter_to_slot: Dict[str, int] = {} 
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        
        self.current_merged_adapter: Optional[str] = None
        
        # [鎖定機制同步] 
        self.lock = threading.RLock()       # 隊列狀態鎖
        self.gpu_lock = threading.Lock()    # 模擬 GPU 資源鎖 (即使是模擬，也要模擬鎖競爭)
        
        self.on_token = None
        self.on_finish = None

    def update_known_adapters(self, adapters: List[str]):
        new_set = set(adapters)
        with self.lock:
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
        # Phase 1: 快速檢查
        with self.lock:
            if adapter_id in self.cpu_cache:
                return

        # Phase 2: 無鎖模擬 I/O 延遲
        time.sleep(SIM_LOAD_DELAY) 
            
        # Phase 3: 上鎖寫入 Cache
        with self.lock:
            if adapter_id not in self.cpu_cache: # Double check
                while len(self.cpu_cache) >= self.max_cpu_loras:
                    self.cpu_cache.popitem(last=False)
                self.cpu_cache[adapter_id] = True

    def _evict_from_gpu(self, adapter_id: str):
        if adapter_id in self.adapter_to_slot:
            slot = self.adapter_to_slot.pop(adapter_id)
            del self.gpu_slots[slot]
            self.slot_lru.move_to_end(slot, last=False) 

    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        self._ensure_cpu_loaded(adapter_id)
        
        with self.lock:
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
                    raise RuntimeError(f"No Simulated VRAM slots available for {aid}. Capacity logic failure.")
                self._load_adapter_to_slot(aid, available_slots[0])

    def merge_adapter(self, adapter_id: str, force: bool = False):
        self._ensure_cpu_loaded(adapter_id)
        with self.lock:
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
                # 模擬真實系統的 GPU 寫入鎖
                with self.gpu_lock:
                    self.current_merged_adapter = adapter_id

    def unmerge_all(self):
        with self.lock:
            if self.current_merged_adapter:
                print(f"🔀 [Simulated Seamless] Unmerging {self.current_merged_adapter}...")
                with self.gpu_lock:
                    self.current_merged_adapter = None

    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 256):
        # 模擬設定：使用 config 中的固定長度
        max_new_tokens = self.FIXED_OUTPUT_LEN
        
        with self.lock:
            self.request_queue.append({
                "request_id": str(request_id),
                "adapter_id": str(adapter_id),
                "seq_len": self.FIXED_INPUT_LEN, # 模擬只看長度
                "past_key_values": None,
                "tokens_gen": [],
                "max_new_tokens": max_new_tokens,
                "done": False
            })

    def is_idle(self) -> bool:
        with self.lock: return len(self.request_queue) == 0 and len(self.running_queue) == 0

    def _cleanup_unused_adapters(self):
        active_aids = {r["adapter_id"] for r in self.running_queue}
        pending_aids = {r["adapter_id"] for r in self.request_queue}
        needed_aids = active_aids.union(pending_aids)
        
        if self.current_merged_adapter:
            needed_aids.add(self.current_merged_adapter)
        
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in needed_aids:
                self._evict_from_gpu(aid)

    # [新增] 模擬完整的重置邏輯，與真實版同步
    def full_reset(self):
        """
        模擬徹底重置引擎狀態。
        """
        print("🧹 [Engine] Starting Full Reset sequence (Simulated)...")
        with self.lock:
            self.request_queue.clear()
            self.running_queue.clear()

            if self.current_merged_adapter:
                with self.gpu_lock:
                    self.current_merged_adapter = None
            
            self.gpu_slots.clear()
            self.adapter_to_slot.clear()
            self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))

            self.known_adapters.clear()
            self.cpu_cache.clear()
            
            print("✨ [Engine] Full Reset Complete.")

    def step(self) -> bool:
        # ==========================================
        # Phase 1: 上鎖管理佇列狀態，分派 Batch
        # ==========================================
        with self.lock:
            self._cleanup_unused_adapters()
            self.running_queue = [r for r in self.running_queue if not r["done"]]

            if not self.running_queue and not self.request_queue:
                return False

            multiplier = 1.0 # Merged 模式下的速度加成
            required_adapters = []
            target_group = []

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
                multiplier = MERGE_SPEED_MULTIPLIER
            else:
                # === [Unmerged Mode Path] ===
                # 這裡的邏輯必須嚴格對齊真實系統的 Unmerged Capacity 計算
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
                        break # 容量已滿
                
                for i in reversed(idx_to_remove):
                    self.request_queue.pop(i)

                if not self.running_queue: return False
                target_group = self.running_queue
                required_adapters = sorted(list({r["adapter_id"] for r in self.running_queue}))

        # ==========================================
        # Phase 2: 無鎖模擬載入 (Disk I/O)
        # ==========================================
        for aid in required_adapters:
            self._ensure_cpu_loaded(aid)

        # ==========================================
        # Phase 3: 鎖定計算延遲與狀態準備
        # ==========================================
        step_sleep_time = 0.0
        decode_reqs = []
        
        with self.lock:
            # 防呆：可能在 Phase 2 期間被 reset 清空了
            if not target_group or all(r["done"] for r in target_group):
                return True

            if not self.current_merged_adapter:
                self._ensure_adapters_resident(required_adapters)
            
            prefill_reqs = [r for r in target_group if r["past_key_values"] is None]
            decode_reqs = [r for r in target_group if r["past_key_values"] is not None]
            
            # 計算模擬延遲
            if prefill_reqs:
                # 模擬 Prefill
                step_sleep_time += (SIM_PREFILL_BASE_TIME * len(prefill_reqs)) * multiplier
                # 標記為已有 Past KV
                for r in prefill_reqs:
                    r["past_key_values"] = True # 模擬物件
                    # 注意：真實系統 Prefill 也會吐出第一個 Token，所以這裡也加入輸出隊列
                    decode_reqs.append(r)
            
            if decode_reqs:
                # 模擬 Decode
                batch_size = len(decode_reqs)
                step_sleep_time += (SIM_DECODE_BASE_TIME + SIM_DECODE_SLOPE * batch_size) * multiplier

        # ==========================================
        # Phase 4: 模擬 GPU 運算 (釋放 self.lock，但持有 gpu_lock)
        # ==========================================
        # 這裡不持有 lock，讓 add_request 可以繼續進來 (模擬 async)
        # 但持有 gpu_lock，模擬 GPU 被佔用
        with self.gpu_lock:
            if step_sleep_time > 0:
                time.sleep(step_sleep_time)

        # ==========================================
        # Phase 5: 上鎖寫回結果
        # ==========================================
        if decode_reqs:
            with self.lock:
                self._process_outputs(decode_reqs)

        return True

    def _process_outputs(self, reqs):
        for req in reqs:
            req["tokens_gen"].append(1) # 模擬生成一個 token
            if self.on_token: 
                self.on_token(req["request_id"], req["tokens_gen"])
            
            if len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: 
                    self.on_finish(req["request_id"], "finished")
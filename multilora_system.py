import os
import torch
import torch.nn as nn
import time
import threading
from typing import Dict, Optional, Any, List, Tuple, Union, Set, Callable
from collections import OrderedDict, deque
from safetensors.torch import load
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Global Context for Dynamic LoRA
# ============================================================
class LoRAContext:
    _current_mapping: Optional[torch.Tensor] = None

    @classmethod
    def set_mapping(cls, mapping: Optional[torch.Tensor]):
        cls._current_mapping = mapping

    @classmethod
    def get_mapping(cls) -> Optional[torch.Tensor]:
        return cls._current_mapping

# ============================================================
# Dynamic LoRA Layer (Optimized)
# ============================================================
class DynamicLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, adapter_slots: int, r: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.adapter_slots = adapter_slots

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # lora_As: (slots, in, r)
        self.lora_As = nn.Parameter(torch.zeros(adapter_slots, base_layer.in_features, r, device=device, dtype=dtype))
        # lora_Bs: (slots, r, out)
        self.lora_Bs = nn.Parameter(torch.zeros(adapter_slots, r, base_layer.out_features, device=device, dtype=dtype))

        self.is_merged = False
        self.merged_idx = -1

        nn.init.kaiming_uniform_(self.lora_As, a=5**0.5)
        nn.init.zeros_(self.lora_Bs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.is_merged:
            return base_out

        adapter_mapping = LoRAContext.get_mapping()
        if adapter_mapping is None:
            return base_out

        x_lora = x.to(self.lora_As.dtype)
        if adapter_mapping.device != self.lora_As.device:
            adapter_mapping = adapter_mapping.to(self.lora_As.device)

        A_selected = self.lora_As.index_select(0, adapter_mapping)
        B_selected = self.lora_Bs.index_select(0, adapter_mapping)

        if x_lora.dim() == 2:
            lora_h = torch.einsum("bi,bir->br", x_lora, A_selected)
            lora_out = torch.einsum("br,bro->bo", lora_h, B_selected)
        else:
            lora_h = torch.einsum("bti,bir->btr", x_lora, A_selected)
            lora_out = torch.einsum("btr,bro->bto", lora_h, B_selected)

        return base_out + (lora_out.to(base_out.dtype) * self.scaling)

    @torch.no_grad()
    def manual_merge(self, slot_id: int):
        if self.is_merged: self.manual_unmerge()
        slot_id = int(slot_id)
        W = self.base_layer.weight.data
        A = self.lora_As.data[slot_id]
        B = self.lora_Bs.data[slot_id]
        W.addmm_(B.T, A.T, alpha=self.scaling)
        self.is_merged = True
        self.merged_idx = slot_id

    @torch.no_grad()
    def manual_unmerge(self):
        if not self.is_merged: return
        W = self.base_layer.weight.data
        A = self.lora_As.data[self.merged_idx]
        B = self.lora_Bs.data[self.merged_idx]
        W.addmm_(B.T, A.T, alpha=-self.scaling)
        self.is_merged = False
        self.merged_idx = -1

# ============================================================
# KV Cache Utilities
# ============================================================
try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    DynamicCache = None
    _HAS_DYNAMIC_CACHE = False

def _to_legacy_cache(past: Any) -> Any:
    if past is None: return None
    if hasattr(past, "to_legacy_cache"): return past.to_legacy_cache()
    return past

def _to_model_cache(past_legacy: Any) -> Any:
    if past_legacy is None: return None
    if hasattr(past_legacy, "get_seq_length"): return past_legacy
    if _HAS_DYNAMIC_CACHE and isinstance(past_legacy, tuple):
        return DynamicCache.from_legacy_cache(past_legacy)
    return past_legacy

def _slice_past_for_sample(past_legacy: Tuple, sample_idx: int, seq_len: int) -> Tuple:
    out = []
    for layer_k, layer_v in past_legacy:
        # standard layout: (batch, heads, seq, dim)
        ks = layer_k[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        vs = layer_v[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        out.append((ks, vs))
    return tuple(out)

def _left_pad_kv(k: torch.Tensor, v: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cur_len = k.shape[2]
    if cur_len >= target_len: return k, v
    pad_len = target_len - cur_len
    k_pad = torch.zeros((k.shape[0], k.shape[1], pad_len, k.shape[3]), device=k.device, dtype=k.dtype)
    v_pad = torch.zeros((v.shape[0], v.shape[1], pad_len, v.shape[3]), device=v.device, dtype=v.dtype)
    return torch.cat([k_pad, k], dim=2), torch.cat([v_pad, v], dim=2)

def _batch_past(past_list: List[Tuple], target_len: int) -> Tuple:
    if not past_list: return ()
    n_layers = len(past_list[0])
    batched = []
    for layer_idx in range(n_layers):
        ks_list, vs_list = [], []
        for p in past_list:
            k, v = p[layer_idx]
            k_aligned, v_aligned = _left_pad_kv(k, v, target_len)
            ks_list.append(k_aligned)
            vs_list.append(v_aligned)
        batched.append((torch.cat(ks_list, dim=0), torch.cat(vs_list, dim=0)))
    return tuple(batched)

# ============================================================
# Multi-LoRA Engine Core (Slot-Based Management)
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, device: Optional[str] = None, torch_dtype: torch.dtype = torch.bfloat16, enable_monitor: bool = True, adapter_fetcher: Optional[Callable[[str], bytes]] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype if self.device.type == "cuda" else torch.float32
        
        # [Capacity Configuration]
        # Merged Mode: 15 slots (Requests only)
        # Unmerged Mode: 12 slots (Requests + LoRA Adapters)
        self.merged_capacity = 15
        self.unmerged_capacity = 12
        
        # Hardware slots matches unmerged capacity
        self.adapter_slots = self.unmerged_capacity 
        
        self.max_cpu_loras = 30
        self.adapter_fetcher = adapter_fetcher 

        # [Constraints]
        self.FIXED_INPUT_LEN = 512
        self.FIXED_OUTPUT_LEN = 256

        print(f"⏳ [Engine] Loading base model: {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left" 
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self._replace_layers(r, alpha)

        # [LRU System]
        self.known_adapters: Set[str] = set()
        self.cpu_cache: OrderedDict[str, Dict] = OrderedDict()
        
        # GPU Management
        self.gpu_slots: Dict[int, str] = {} 
        self.adapter_to_slot: Dict[str, int] = {} 
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        
        self.current_merged_adapter: Optional[str] = None
        self.lock = threading.RLock()
        self.on_token = None
        self.on_finish = None

    def _replace_layers(self, r: int, alpha: int):
        target_suffixes = {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "o_proj"}
        replaced_count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and name.split(".")[-1] in target_suffixes:
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                target_name = name.split(".")[-1]
                new_layer = DynamicLoRALinear(module, self.adapter_slots, r, alpha).to(self.device)
                setattr(parent, target_name, new_layer)
                replaced_count += 1
        print(f"🔧 [Engine] Replaced {replaced_count} layers with DynamicLoRALinear (Slots={self.adapter_slots}).")

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
            if not self.adapter_fetcher:
                raise ValueError("No adapter fetcher configured.")
            
            while len(self.cpu_cache) >= self.max_cpu_loras:
                evicted_aid, _ = self.cpu_cache.popitem(last=False)
                # print(f"♻️ [CPU LRU] Evicted {evicted_aid} from CPU RAM.")

            model_bytes = self.adapter_fetcher(adapter_id)
            weights = load(model_bytes)
            
            adapter_weights = {}
            for n, m in self.model.named_modules():
                if isinstance(m, DynamicLoRALinear):
                    key_A = f"base_model.model.{n}.lora_A.weight"
                    key_B = f"base_model.model.{n}.lora_B.weight"
                    if key_A in weights and key_B in weights:
                        adapter_weights[n] = {
                            "A": weights[key_A].T.contiguous().to(torch.float32).pin_memory(),
                            "B": weights[key_B].T.contiguous().to(torch.float32).pin_memory()
                        }
            if adapter_weights:
                self.cpu_cache[adapter_id] = adapter_weights
                
        except Exception as e:
            print(f"❌ [Engine] Failed to load {adapter_id}: {e}")
            raise e

    def _evict_from_gpu(self, adapter_id: str):
        if adapter_id in self.adapter_to_slot:
            slot = self.adapter_to_slot.pop(adapter_id)
            del self.gpu_slots[slot]
            self.slot_lru.move_to_end(slot, last=False) 

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        self._ensure_cpu_loaded(adapter_id)
        
        if adapter_id in self.cpu_cache:
            self.cpu_cache.move_to_end(adapter_id)

        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id: 
            self._evict_from_gpu(self.gpu_slots[slot_id])

        for n, m in self.model.named_modules():
            if isinstance(m, DynamicLoRALinear) and n in self.cpu_cache[adapter_id]:
                w = self.cpu_cache[adapter_id][n]
                m.lora_As.data[slot_id].copy_(w["A"].to(m.lora_As.device, m.lora_As.dtype, non_blocking=True))
                m.lora_Bs.data[slot_id].copy_(w["B"].to(m.lora_Bs.device, m.lora_Bs.dtype, non_blocking=True))
        
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id

    def _ensure_adapters_resident(self, required_adapters: List[str]):
        for aid in required_adapters:
            if aid not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    raise RuntimeError(f"No VRAM slots available for {aid}. Capacity logic failure.")
                self._load_adapter_to_slot(aid, available_slots[0])

    @torch.no_grad()
    def merge_adapter(self, adapter_id: str, force: bool = False):
        """
        Seamless Merge:
        Convert the model to dedicated mode WITHOUT interrupting running requests.
        The KV cache remains valid.
        """
        self._ensure_cpu_loaded(adapter_id)
        
        with self.lock:
            # 確保該 Adapter 已經載入 VRAM
            if adapter_id not in self.adapter_to_slot:
                available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                if not available_slots:
                    # 嘗試清出空間
                    self._cleanup_unused_adapters() 
                    available_slots = [s for s in range(self.adapter_slots) if s not in self.gpu_slots]
                    if not available_slots:
                        raise RuntimeError("No VRAM slots available to load adapter for merging.")
                
                self._load_adapter_to_slot(adapter_id, available_slots[0])
            
            slot_id = self.adapter_to_slot[adapter_id]

            # 如果已經有別的 Merged，先 Unmerge 它
            if self.current_merged_adapter and self.current_merged_adapter != adapter_id:
                self.unmerge_all()
            
            if self.current_merged_adapter != adapter_id:
                print(f"🔀 [Seamless] Merging {adapter_id} into base model on-the-fly...")
                for m in self.model.modules():
                    if isinstance(m, DynamicLoRALinear):
                        m.manual_merge(slot_id)
                
                self.current_merged_adapter = adapter_id
                # 注意：我們不 Evict 這個 Slot，必須保留它供 Unmerge 使用

    @torch.no_grad()
    def unmerge_all(self):
        """
        Seamless Unmerge:
        Convert back to shared mode WITHOUT interrupting running requests.
        """
        with self.lock:
            if self.current_merged_adapter:
                print(f"🔀 [Seamless] Unmerging {self.current_merged_adapter}...")
                for m in self.model.modules():
                    if isinstance(m, DynamicLoRALinear):
                        m.manual_unmerge()
                
                self.current_merged_adapter = None
                torch.cuda.empty_cache()

    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 256):
        max_new_tokens = self.FIXED_OUTPUT_LEN 
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_len = self.FIXED_INPUT_LEN
        current_len = len(tokens)
        
        if current_len > target_len:
            tokens = tokens[-target_len:]
            input_ids = torch.tensor([tokens], device=self.device)
            attention_mask = torch.ones((1, target_len), device=self.device)
        elif current_len < target_len:
            pad_len = target_len - current_len
            pad_id = self.tokenizer.pad_token_id
            padding = [pad_id] * pad_len
            tokens = padding + tokens
            input_ids = torch.tensor([tokens], device=self.device)
            attention_mask = torch.tensor([[0]*pad_len + [1]*current_len], device=self.device)
        else:
            input_ids = torch.tensor([tokens], device=self.device)
            attention_mask = torch.ones((1, target_len), device=self.device)

        with self.lock:
            self.request_queue.append({
                "request_id": str(request_id),
                "adapter_id": str(adapter_id),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "seq_len": target_len,
                "past_key_values": None,
                "tokens_gen": [],
                "max_new_tokens": max_new_tokens,
                "done": False
            })

    def is_idle(self) -> bool:
        with self.lock: return len(self.request_queue) == 0 and len(self.running_queue) == 0

    def _cleanup_unused_adapters(self):
        """
        Evict LoRAs from VRAM if they are not needed.
        [Modified] Protect the merged adapter.
        """
        active_aids = {r["adapter_id"] for r in self.running_queue}
        pending_aids = {r["adapter_id"] for r in self.request_queue}
        needed_aids = active_aids.union(pending_aids)
        
        # [Protection] Merged Adapter is needed for Unmerge
        if self.current_merged_adapter:
            needed_aids.add(self.current_merged_adapter)
        
        cleaned = False
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in needed_aids:
                self._evict_from_gpu(aid)
                cleaned = True
        
        if cleaned:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self) -> bool:
        with self.lock:
            self._cleanup_unused_adapters()
            self.running_queue = [r for r in self.running_queue if not r["done"]]

            if not self.running_queue and not self.request_queue:
                return False

            if self.current_merged_adapter:
                # === [Merged Mode Path] ===
                # 只處理屬於該 Adapter 的請求
                target_group = [r for r in self.running_queue if r["adapter_id"] == self.current_merged_adapter]
                
                # 補人
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

                # Merged 模式下不需要 map
                LoRAContext.set_mapping(None)
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
        try:
            prefill_reqs = [r for r in target_group if r["past_key_values"] is None]
            decode_reqs = [r for r in target_group if r["past_key_values"] is not None]

            if prefill_reqs:
                batch_reqs = prefill_reqs
                input_ids = torch.cat([r["input_ids"] for r in batch_reqs], dim=0)
                attention_mask = torch.cat([r["attention_mask"] for r in batch_reqs], dim=0)
                
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in batch_reqs], device=self.device)
                    LoRAContext.set_mapping(mapping)

                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

            elif decode_reqs:
                batch_reqs = decode_reqs
                
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in batch_reqs], device=self.device)
                    LoRAContext.set_mapping(mapping)

                past_list = [r["past_key_values"] for r in batch_reqs]
                max_past_len = max(p[0][0].shape[2] for p in past_list)
                batched_past = _to_model_cache(_batch_past(past_list, max_past_len))
                
                input_ids = torch.cat([r["input_ids"][:, -1:] for r in batch_reqs], dim=0)
                attention_mask = torch.ones((len(batch_reqs), max_past_len + 1), device=self.device)
                
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=batched_past, use_cache=True)
            else:
                return

            self._process_outputs(out, batch_reqs)
        except RuntimeError as e:
             if "out of memory" in str(e).lower():
                 print(f"🚨 [OOM] Batch too large.")
                 if self.running_queue:
                    self.running_queue.pop() # Simple drop
                 torch.cuda.empty_cache()
             else:
                 raise e

    def _process_outputs(self, model_out, reqs):
        logits = model_out.logits[:, -1, :] 
        new_tokens = torch.argmax(logits, dim=-1)
        new_past_legacy = _to_legacy_cache(model_out.past_key_values)
        
        for i, req in enumerate(reqs):
            token_id = new_tokens[i].item()
            req["tokens_gen"].append(token_id)
            if self.on_token: self.on_token(req["request_id"], req["tokens_gen"])
            
            req["input_ids"] = torch.cat([req["input_ids"], new_tokens[i:i+1].view(1, 1)], dim=-1)
            current_total_len = req["seq_len"] + len(req["tokens_gen"])
            req["past_key_values"] = _slice_past_for_sample(new_past_legacy, i, current_total_len)
            
            if len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: self.on_finish(req["request_id"], "finished")


import os
import torch
import torch.nn as nn
import time
import threading
from typing import Dict, Optional, Any, List, Tuple, Union, Set
from collections import OrderedDict, deque
from safetensors.torch import load_file
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
        # 如果已經合併，直接回傳 Base Output (因為權重已經包含 LoRA)
        if self.is_merged:
            return base_out

        adapter_mapping = LoRAContext.get_mapping()
        if adapter_mapping is None:
            return base_out

        # 標準 LoRA 計算路徑 (未合併時使用)
        x_lora = x.to(self.lora_As.dtype)
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
        """
        [Optimized] 使用 addmm_ 進行原地合併，避免產生中間矩陣 delta。
        W_new = W + scaling * (A @ B)^T
              = W + scaling * (B^T @ A^T)
        """
        if self.is_merged: self.manual_unmerge()
        slot_id = int(slot_id)
        
        W = self.base_layer.weight.data
        A = self.lora_As.data[slot_id] # (in, r)
        B = self.lora_Bs.data[slot_id] # (r, out)
        
        # W shape: (out, in)
        # B.T shape: (out, r)
        # A.T shape: (r, in)
        # addmm_ 執行: W = 1*W + alpha*(B.T @ A.T)
        W.addmm_(B.T, A.T, alpha=self.scaling)
        
        self.is_merged = True
        self.merged_idx = slot_id

    @torch.no_grad()
    def manual_unmerge(self):
        """
        [Optimized] 使用 addmm_ 進行原地還原。
        W_orig = W_merged - scaling * (B^T @ A^T)
        """
        if not self.is_merged: return
        
        W = self.base_layer.weight.data
        A = self.lora_As.data[self.merged_idx]
        B = self.lora_Bs.data[self.merged_idx]
        
        # alpha 設為負值即為減法
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
# Multi-LoRA Engine Core
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, adapter_slots: int = 2, max_batch_size: int = 4, device: Optional[str] = None, torch_dtype: torch.dtype = torch.bfloat16, enable_monitor: bool = True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype if self.device.type == "cuda" else torch.float32
        self.enable_monitor = enable_monitor
        self.max_batch_size = int(max_batch_size)
        self.adapter_slots = int(adapter_slots)

        print(f"⏳ [Engine] Loading base model: {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left" 
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self._replace_layers(r, alpha)

        self.cpu_cache: Dict[str, Dict] = {}
        self.gpu_slots: Dict[int, str] = {} 
        self.adapter_to_slot: Dict[str, int] = {} 
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        self.finished_results = deque(maxlen=1000)
        
        self.current_merged_adapter: Optional[str] = None
        self.lock = threading.RLock()
        self.is_draining = False
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
        print(f"🔧 [Engine] Replaced {replaced_count} layers with DynamicLoRALinear.")

    def load_adapters_to_cpu(self, base_path: str = "./testLoRA", allowed_adapters: Optional[List[str]] = None):
        if not os.path.exists(base_path): return
        whitelist = set(allowed_adapters) if allowed_adapters is not None else None
        
        if whitelist is not None:
            current_loaded = list(self.cpu_cache.keys())
            for aid in current_loaded:
                if aid not in whitelist:
                    del self.cpu_cache[aid]
                    if aid in self.adapter_to_slot:
                        slot = self.adapter_to_slot.pop(aid)
                        if self.gpu_slots.get(slot) == aid: del self.gpu_slots[slot]
                        self.slot_lru.move_to_end(slot, last=False)

        try:
            dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        except OSError: return

        for d in dirs:
            aid = d.split("_")[-1] if "_" in d else d
            if whitelist is not None and aid not in whitelist: continue
            if aid in self.cpu_cache: continue

            safetensor_path = os.path.join(base_path, d, "adapter_model.safetensors")
            if not os.path.exists(safetensor_path): continue

            try:
                weights = load_file(safetensor_path, device="cpu")
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
                if adapter_weights: self.cpu_cache[aid] = adapter_weights
            except Exception as e:
                print(f"❌ [Engine] Failed to load {aid}: {e}")
        print(f"✅ [Engine] Sync done. Total Resident: {len(self.cpu_cache)}")

    def _evict_slot(self, slot_id: int):
        if slot_id in self.gpu_slots:
            old_adapter = self.gpu_slots.pop(slot_id)
            self.adapter_to_slot.pop(old_adapter, None)
        if slot_id in self.slot_lru:
            self.slot_lru.move_to_end(slot_id, last=False)

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        if adapter_id not in self.cpu_cache: raise KeyError(f"Adapter {adapter_id} not in CPU cache.")
        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id: self._evict_slot(slot_id)

        for n, m in self.model.named_modules():
            if isinstance(m, DynamicLoRALinear) and n in self.cpu_cache[adapter_id]:
                w = self.cpu_cache[adapter_id][n]
                m.lora_As.data[slot_id].copy_(w["A"].to(m.lora_As.device, m.lora_As.dtype, non_blocking=True))
                m.lora_Bs.data[slot_id].copy_(w["B"].to(m.lora_Bs.device, m.lora_Bs.dtype, non_blocking=True))
        
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id
        self.slot_lru.move_to_end(slot_id, last=True)

    def _ensure_adapters_resident(self, required_adapters: List[str]):
        required_set = set(required_adapters)
        missing = [aid for aid in required_adapters if aid not in self.adapter_to_slot]
        if not missing:
            for aid in required_adapters: self.slot_lru.move_to_end(self.adapter_to_slot[aid], last=True)
            return

        evictable_slots = [s for s in self.slot_lru if self.gpu_slots.get(s) not in required_set]
        for aid in missing:
            if not evictable_slots: raise RuntimeError("No available slots to load required adapters!")
            self._load_adapter_to_slot(aid, evictable_slots.pop(0))

    @torch.no_grad()
    def merge_adapter(self, adapter_id: str, force: bool = False):
        if adapter_id not in self.cpu_cache: raise KeyError(f"Unknown adapter {adapter_id}")
        if not force:
            with self.lock: self.is_draining = True
            for _ in range(100):
                if len(self.running_queue) == 0: break
                time.sleep(0.1)
        with self.lock:
            if force:
                kept = []
                for req in self.running_queue:
                    if req["adapter_id"] == adapter_id: kept.append(req)
                    else:
                        req["done"] = True
                        if self.on_finish: self.on_finish(req["request_id"], "aborted_by_merge")
                self.running_queue = kept
            self.is_draining = False
            if adapter_id not in self.adapter_to_slot:
                self._load_adapter_to_slot(adapter_id, next(iter(self.slot_lru)))
            slot_id = self.adapter_to_slot[adapter_id]
            if self.current_merged_adapter and self.current_merged_adapter != adapter_id: self.unmerge_all()
            if self.current_merged_adapter != adapter_id:
                for m in self.model.modules():
                    if isinstance(m, DynamicLoRALinear): m.manual_merge(slot_id)
                self.current_merged_adapter = adapter_id

    @torch.no_grad()
    def unmerge_all(self):
        with self.lock:
            if self.current_merged_adapter:
                for m in self.model.modules():
                    if isinstance(m, DynamicLoRALinear): m.manual_unmerge()
                self.current_merged_adapter = None

    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 128):
        if adapter_id not in self.cpu_cache: raise KeyError(f"Adapter {adapter_id} unavailable.")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.lock:
            self.request_queue.append({
                "request_id": str(request_id),
                "adapter_id": str(adapter_id),
                "input_ids": inputs.input_ids.to(self.device),
                "seq_len": int(inputs.input_ids.shape[1]),
                "past_key_values": None,
                "tokens_gen": [],
                "max_new_tokens": int(max_new_tokens),
                "done": False
            })

    def is_idle(self) -> bool:
        with self.lock: return len(self.request_queue) == 0 and len(self.running_queue) == 0

    @torch.no_grad()
    def step(self) -> bool:
        with self.lock:
            self.running_queue = [r for r in self.running_queue if not r["done"]]

            # 1. 調度：將新請求加入運行隊列 (Batching)
            if not self.is_draining:
                # Merged Mode
                if self.current_merged_adapter:
                    while len(self.running_queue) < self.max_batch_size and self.request_queue:
                        cand_idx = -1
                        for i, req in enumerate(self.request_queue):
                            if req["adapter_id"] == self.current_merged_adapter:
                                cand_idx = i
                                break
                        if cand_idx != -1: self.running_queue.append(self.request_queue.pop(cand_idx))
                        else: break
                # Normal Mode
                else:
                    while len(self.running_queue) < self.max_batch_size and self.request_queue:
                        req = self.request_queue[0]
                        current_aids = {r["adapter_id"] for r in self.running_queue}
                        # Slot Check
                        if req["adapter_id"] not in current_aids and len(current_aids) >= self.adapter_slots:
                            break 
                        self.running_queue.append(self.request_queue.pop(0))

            if not self.running_queue: return False

            # 2. 準備資源
            if self.current_merged_adapter:
                LoRAContext.set_mapping(None)
            else:
                required = sorted(list({r["adapter_id"] for r in self.running_queue}))
                self._ensure_adapters_resident(required)

            # 3. 執行策略：優先處理 Prefill
            prefill_reqs = [r for r in self.running_queue if r["past_key_values"] is None]
            decode_reqs = [r for r in self.running_queue if r["past_key_values"] is not None]

            if prefill_reqs:
                # --- Prefill Phase (Batch Processing) ---
                target_group = prefill_reqs
                
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in target_group], device=self.device)
                    LoRAContext.set_mapping(mapping)

                input_ids_list = [r["input_ids"] for r in target_group]
                max_len = max(x.shape[1] for x in input_ids_list)
                padded_input = torch.full((len(target_group), max_len), self.tokenizer.pad_token_id, device=self.device)
                attention_mask = torch.zeros((len(target_group), max_len), device=self.device)
                
                for i, ids in enumerate(input_ids_list):
                    L = ids.shape[1]
                    padded_input[i, -L:] = ids[0]
                    attention_mask[i, -L:] = 1
                
                out = self.model(input_ids=padded_input, attention_mask=attention_mask, use_cache=True)

            elif decode_reqs:
                # --- Decode Phase ---
                target_group = decode_reqs
                
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in target_group], device=self.device)
                    LoRAContext.set_mapping(mapping)

                past_list = [r["past_key_values"] for r in target_group]
                max_past_len = max(p[0][0].shape[2] for p in past_list)
                batched_past = _to_model_cache(_batch_past(past_list, max_past_len))
                
                input_ids = torch.cat([r["input_ids"][:, -1:] for r in target_group], dim=0)
                attention_mask = torch.ones((len(target_group), max_past_len + 1), device=self.device)
                
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=batched_past, use_cache=True)
            
            else:
                return False

            self._process_outputs(out, target_group)
            LoRAContext.set_mapping(None)
            return True

    def _process_outputs(self, model_out, reqs):
        logits = model_out.logits[:, -1, :] 
        new_tokens = torch.argmax(logits, dim=-1)
        new_past_legacy = _to_legacy_cache(model_out.past_key_values)
        
        for i, req in enumerate(reqs):
            token_id = new_tokens[i].item()
            req["tokens_gen"].append(token_id)
            if self.on_token: self.on_token(req["request_id"], req["tokens_gen"])
            
            req["input_ids"] = torch.cat([req["input_ids"], new_tokens[i:i+1].view(1, 1)], dim=-1)
            current_len = req["seq_len"] + len(req["tokens_gen"])
            req["past_key_values"] = _slice_past_for_sample(new_past_legacy, i, current_len)
            
            if token_id == self.tokenizer.eos_token_id or len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: self.on_finish(req["request_id"], "finished")
import os
import psutil
import torch
import torch.nn as nn
import time
import threading
from typing import Dict, Optional, Any, List, Tuple, Callable
from collections import OrderedDict, deque
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Dynamic LoRA Linear
# ============================================================
class LoRAContext:
    _current_mapping: Optional[torch.Tensor] = None

    @classmethod
    def set_mapping(cls, mapping: Optional[torch.Tensor]):
        cls._current_mapping = mapping

    @classmethod
    def get_mapping(cls) -> Optional[torch.Tensor]:
        return cls._current_mapping

class DynamicLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, adapter_slots: int, r: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_As = nn.Parameter(torch.zeros(adapter_slots, base_layer.in_features, r, device=device, dtype=dtype))
        self.lora_Bs = nn.Parameter(torch.zeros(adapter_slots, r, base_layer.out_features, device=device, dtype=dtype))

        self.is_merged = False
        self.merged_idx = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.is_merged:
            return base_out

        adapter_mapping = LoRAContext.get_mapping()
        if adapter_mapping is None:
            return base_out

        x_lora = x.to(self.lora_As.dtype)
        A = self.lora_As.index_select(0, adapter_mapping)
        B = self.lora_Bs.index_select(0, adapter_mapping)

        if x_lora.dim() == 2:
            lora_h = torch.einsum("bi,bir->br", x_lora, A)
            lora_out = torch.einsum("br,bro->bo", lora_h, B)
        else:
            lora_h = torch.einsum("bti,bir->btr", x_lora, A)
            lora_out = torch.einsum("btr,bro->bto", lora_h, B)

        return base_out + (lora_out.to(base_out.dtype) * self.scaling)

    @torch.no_grad()
    def manual_merge(self, slot_id: int):
        if self.is_merged:
            self.manual_unmerge()
        slot_id = int(slot_id)
        W = self.base_layer.weight.data
        A = self.lora_As.data[slot_id]
        B = self.lora_Bs.data[slot_id]
        delta = (A @ B) * self.scaling
        W.add_(delta.T.to(W.dtype))
        self.is_merged = True
        self.merged_idx = slot_id

    @torch.no_grad()
    def manual_unmerge(self):
        if not self.is_merged:
            return
        W = self.base_layer.weight.data
        A = self.lora_As.data[self.merged_idx]
        B = self.lora_Bs.data[self.merged_idx]
        delta = (A @ B) * self.scaling
        W.sub_(delta.T.to(W.dtype))
        self.is_merged = False
        self.merged_idx = -1

# ============================================================
# KV Cache Helpers
# ============================================================
try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except Exception:
    DynamicCache = None
    _HAS_DYNAMIC_CACHE = False

def _to_legacy_cache(past):
    if past is None: return None
    if hasattr(past, "to_legacy_cache"): return past.to_legacy_cache()
    return past

def _to_model_cache(past_legacy):
    if past_legacy is None: return None
    if hasattr(past_legacy, "get_seq_length"): return past_legacy
    if _HAS_DYNAMIC_CACHE and isinstance(past_legacy, tuple):
        return DynamicCache.from_legacy_cache(past_legacy)
    return past_legacy

def _slice_past_for_sample(past_legacy, sample_idx: int, seq_len: int):
    out = []
    for k, v in past_legacy:
        ks = k[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        vs = v[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        out.append((ks, vs))
    return tuple(out)

def _left_pad_kv(k, v, target_len):
    cur = k.shape[2]
    if cur == target_len: return k, v
    pad = target_len - cur
    k_pad = torch.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), device=k.device, dtype=k.dtype)
    v_pad = torch.zeros((v.shape[0], v.shape[1], pad, v.shape[3]), device=v.device, dtype=v.dtype)
    return torch.cat([k_pad, k], dim=2), torch.cat([v_pad, v], dim=2)

def _batch_past(past_list, target_len):
    n_layers = len(past_list[0])
    batched = []
    for layer in range(n_layers):
        ks, vs = [], []
        for p in past_list:
            k, v = p[layer]
            k2, v2 = _left_pad_kv(k, v, target_len)
            ks.append(k2)
            vs.append(v2)
        batched.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
    return tuple(batched)

# ============================================================
# MultiLoRA Engine
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, adapter_slots: int = 2, max_batch_size: int = 4, device: Optional[str] = None, torch_dtype=torch.bfloat16, enable_monitor: bool = True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype if self.device.type == "cuda" else torch.float32
        self.enable_monitor = enable_monitor

        print(f"⏳ Loading model: {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        
        self.adapter_slots = int(adapter_slots)
        self.max_batch_size = int(max_batch_size)
        self.on_token = None
        self.on_finish = None

        self.cpu_cache = {}
        self.gpu_slots = {}
        self.adapter_to_slot = {}
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue = []
        self.running_queue = []
        self.finished_results = deque(maxlen=1000)
        self.current_merged_adapter = None

        # 狀態管理與鎖
        self.lock = threading.RLock() # 允許 Reentrant
        self.is_draining = False      # 是否正在排空請求

        self._replace_layers(r, alpha)

    def _replace_layers(self, r, alpha):
        target_names = {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"}
        replaced = 0
        for name, module in list(self.model.named_modules()):
            if name.split(".")[-1] in target_names and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                new_layer = DynamicLoRALinear(module, self.adapter_slots, r, alpha).to(self.device)
                setattr(parent, name.split(".")[-1], new_layer)
                replaced += 1
        print(f"🔧 Replaced {replaced} layers with DynamicLoRALinear.")

    def load_adapters_to_cpu(self, base_path: str = "./testLoRA", allowed_adapters: Optional[List[str]] = None):
        """
        [MODIFIED] 支援「多退少補 (Incremental Update)」的載入機制。
        1. 卸載：移除不在 allowed_adapters 白名單中的 LoRA。
        2. 載入：只讀取白名單中且尚未在記憶體內的 LoRA。
        """
        if not os.path.exists(base_path):
            print(f"⚠️ Path {base_path} not found.")
            return

        whitelist = set(allowed_adapters) if allowed_adapters is not None else None
        
        # ==========================================
        # 1. Unload Phase (卸載)
        # ==========================================
        if whitelist is not None:
            # 找出目前在 Cache 中，但不在白名單的 Adapters
            current_loaded = list(self.cpu_cache.keys())
            for aid in current_loaded:
                if aid not in whitelist:
                    print(f"🗑️ [Pruning] Unloading adapter: {aid}")
                    
                    # A. 從 CPU Cache 移除
                    del self.cpu_cache[aid]
                    
                    # B. 如果它正在 GPU 上，強制移除
                    if aid in self.adapter_to_slot:
                        slot_id = self.adapter_to_slot.pop(aid)
                        if slot_id in self.gpu_slots:
                            del self.gpu_slots[slot_id]
                        # 更新 LRU: 讓被釋放的 Slot 保持在 LRU 隊列中
                        self.slot_lru.move_to_end(slot_id, last=False)
        
        # ==========================================
        # 2. Load Phase (載入)
        # ==========================================
        # 掃描硬碟上的 Adapter
        try:
            adapters_on_disk = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        except Exception as e:
            print(f"❌ Error listing directories: {e}")
            return

        loaded_count = 0
        skipped_count = 0

        for folder in adapters_on_disk:
            aid = str(folder.split("_")[-1]) if "_" in folder else str(folder)
            
            # [Filter 1] 白名單過濾
            if whitelist is not None and aid not in whitelist:
                continue

            # [Filter 2] 增量檢查：如果已經載入，就跳過
            if aid in self.cpu_cache:
                skipped_count += 1
                continue

            # 準備載入
            st_path = os.path.join(base_path, folder, "adapter_model.safetensors")
            if not os.path.exists(st_path):
                continue
            
            try:
                # print(f"📥 Loading new adapter: {aid}...")
                weights = load_file(st_path, device="cpu")
                self.cpu_cache[aid] = {}
                
                # 使用 model.named_modules 匹配權重
                for n, m in self.model.named_modules():
                    if isinstance(m, DynamicLoRALinear):
                        a_key = f"base_model.model.{n}.lora_A.weight"
                        b_key = f"base_model.model.{n}.lora_B.weight"
                        if a_key in weights and b_key in weights:
                            self.cpu_cache[aid][n] = {
                                "A": weights[a_key].T.contiguous().to(torch.float32).pin_memory(),
                                "B": weights[b_key].T.contiguous().to(torch.float32).pin_memory(),
                            }
                loaded_count += 1
            except Exception as e:
                print(f"❌ Failed to load {aid}: {e}")

        total_now = len(self.cpu_cache)
        print(f"✅ Incremental update done. Loaded: {loaded_count}, Skipped (Already Cached): {skipped_count}, Total Cached: {total_now}")
        print(f"   Current Adapters: {list(self.cpu_cache.keys())}")

    def _evict_slot(self, slot_id):
        if slot_id in self.gpu_slots:
            old = self.gpu_slots.pop(slot_id)
            self.adapter_to_slot.pop(old, None)
        if slot_id in self.slot_lru:
            self.slot_lru.move_to_end(slot_id)

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id, slot_id):
        if adapter_id not in self.cpu_cache: raise KeyError(f"Adapter {adapter_id} not loaded.")
        if slot_id in self.gpu_slots: self._evict_slot(slot_id)

        for n, m in self.model.named_modules():
            if isinstance(m, DynamicLoRALinear) and n in self.cpu_cache[adapter_id]:
                w = self.cpu_cache[adapter_id][n]
                m.lora_As.data[slot_id].copy_(w["A"].to(m.lora_As.device, m.lora_As.dtype, non_blocking=True))
                m.lora_Bs.data[slot_id].copy_(w["B"].to(m.lora_Bs.device, m.lora_Bs.dtype, non_blocking=True))
        
        if self.device.type == "cuda": torch.cuda.synchronize()
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id
        self.slot_lru.move_to_end(slot_id)

    def _ensure_adapters_resident(self, required):
        required_set = set(required)
        current = set(self.gpu_slots.values())
        
        # Evict unneeded
        for slot, aid in list(self.gpu_slots.items()):
            if aid not in required_set: self._evict_slot(slot)
            
        # Load missing
        missing = [a for a in required if a not in self.gpu_slots.values()]
        free_slots = [s for s in self.slot_lru if s not in self.gpu_slots]
        
        for aid in missing:
            slot = free_slots.pop(0) if free_slots else next(iter(self.slot_lru))
            self._load_adapter_to_slot(aid, slot)
            
        for aid in required:
            self.slot_lru.move_to_end(self.adapter_to_slot[aid])

    @torch.no_grad()
    def merge_adapter(self, adapter_id, force: bool = False):
        """
        Merge 指定的 Adapter 到模型權重中。
        """
        if adapter_id not in self.cpu_cache: 
            raise KeyError(f"Unknown adapter {adapter_id}")

        # 1. Graceful Draining (非強制模式)
        if not force:
            with self.lock:
                self.is_draining = True 
            
            print(f"⏳ [Merge] Draining requests for merge {adapter_id}...")
            while True:
                with self.lock:
                    if len(self.running_queue) == 0:
                        break
                time.sleep(0.1) 
            print(f"✅ [Merge] Drained. Proceeding to merge.")

        # 2. 執行 Merge (加鎖)
        with self.lock:
            if force:
                new_running = []
                aborted = 0
                for req in self.running_queue:
                    if req["adapter_id"] != adapter_id:
                        req["done"] = True
                        if self.on_finish: 
                            self.on_finish(req["request_id"], "aborted_by_merge")
                        aborted += 1
                    else:
                        new_running.append(req)
                self.running_queue = new_running
                if aborted > 0: 
                    print(f"⚠️ [Merge] Force aborted {aborted} requests.")

            self.is_draining = False 

            if adapter_id not in self.adapter_to_slot:
                self._load_adapter_to_slot(adapter_id, next(iter(self.slot_lru)))
            
            slot_id = self.adapter_to_slot[adapter_id]
            
            if self.current_merged_adapter and self.current_merged_adapter != adapter_id:
                self.unmerge_all()
            
            for m in self.model.modules():
                if isinstance(m, DynamicLoRALinear): m.manual_merge(slot_id)
            
            self.current_merged_adapter = adapter_id

    @torch.no_grad()
    def unmerge_all(self):
        with self.lock:
            for m in self.model.modules():
                if isinstance(m, DynamicLoRALinear): m.manual_unmerge()
            self.current_merged_adapter = None

    def add_request(self, prompt, adapter_id, request_id, max_new_tokens=128):
        if adapter_id not in self.cpu_cache: 
            raise KeyError(f"Adapter {adapter_id} not available (Filtered or Not Found)")
        inputs = self.tokenizer(prompt, return_tensors="pt")
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

    def is_idle(self):
        return len(self.request_queue) == 0 and len(self.running_queue) == 0

    @torch.no_grad()
    def step(self):
        with self.lock:
            # Scheduler
            self.running_queue = [r for r in self.running_queue if not r["done"]]
            
            if not self.is_draining:
                if self.current_merged_adapter:
                    pass 

                while len(self.running_queue) < self.max_batch_size and self.request_queue:
                    req = self.request_queue[0]
                    if self.current_merged_adapter and req["adapter_id"] != self.current_merged_adapter:
                        break 
                    
                    current_aids = {r["adapter_id"] for r in self.running_queue}
                    if not self.current_merged_adapter and req["adapter_id"] not in current_aids and len(current_aids) >= self.adapter_slots:
                        break
                        
                    self.running_queue.append(self.request_queue.pop(0))

            if not self.running_queue: return False

            # Load Adapters
            if self.current_merged_adapter:
                required = [self.current_merged_adapter]
            else:
                required = sorted(list({r["adapter_id"] for r in self.running_queue}))
            self._ensure_adapters_resident(required)

            # Batching
            prefill = [r for r in self.running_queue if r["past_key_values"] is None]
            decode = [r for r in self.running_queue if r["past_key_values"] is not None]

            # Prefill Logic
            if prefill:
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in prefill], device=self.device)
                    LoRAContext.set_mapping(mapping)
                else:
                    LoRAContext.set_mapping(None)

                input_ids = [r["input_ids"] for r in prefill]
                max_len = max(x.shape[1] for x in input_ids)
                padded = torch.full((len(prefill), max_len), self.tokenizer.pad_token_id, device=self.device)
                attn = torch.zeros((len(prefill), max_len), device=self.device)
                
                for i, ids in enumerate(input_ids):
                    L = ids.shape[1]
                    padded[i, -L:] = ids[0]
                    attn[i, -L:] = 1

                out = self.model(input_ids=padded, attention_mask=attn, use_cache=True)
                self._process_logits(out, prefill)
                LoRAContext.set_mapping(None)
                return True

            # Decode Logic
            if decode:
                if not self.current_merged_adapter:
                    mapping = torch.tensor([self.adapter_to_slot[r["adapter_id"]] for r in decode], device=self.device)
                    LoRAContext.set_mapping(mapping)
                else:
                    LoRAContext.set_mapping(None)

                past_list = [r["past_key_values"] for r in decode]
                max_past = max(p[0][0].shape[2] for p in past_list)
                batched_past = _to_model_cache(_batch_past(past_list, max_past))
                inputs = torch.cat([r["input_ids"][:, -1:] for r in decode], dim=0)
                attn = torch.ones((len(decode), max_past + 1), device=self.device)

                out = self.model(input_ids=inputs, attention_mask=attn, past_key_values=batched_past, use_cache=True)
                self._process_logits(out, decode)
                LoRAContext.set_mapping(None)
                return True

            return False

    def _process_logits(self, out, batch_reqs):
        logits = out.logits[:, -1, :]
        tokens = torch.argmax(logits, dim=-1)
        new_past = _to_legacy_cache(out.past_key_values)
        
        for i, req in enumerate(batch_reqs):
            tok = tokens[i].item()
            req["tokens_gen"].append(tok)
            
            if self.on_token: 
                self.on_token(req["request_id"], req["tokens_gen"])
            
            # Update State
            req["input_ids"] = torch.cat([req["input_ids"], tokens[i:i+1].view(1,1)], dim=-1)
            req["past_key_values"] = _slice_past_for_sample(new_past, i, req["seq_len"] + len(req["tokens_gen"]))
            
            if tok == self.tokenizer.eos_token_id or len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
                if self.on_finish: self.on_finish(req["request_id"], "finished")
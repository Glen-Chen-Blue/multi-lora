# multilora_system.py
import os
import psutil
from typing import Dict, Optional, Any, List, Tuple, Callable
from collections import OrderedDict, deque

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Cache compatibility (Transformers new Cache API vs legacy tuple)
# ============================================================
try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except Exception:
    DynamicCache = None
    _HAS_DYNAMIC_CACHE = False


def _to_legacy_cache(past):
    """Cache object -> legacy tuple if needed."""
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


def _to_model_cache(past_legacy):
    """legacy tuple -> Cache object if needed."""
    if past_legacy is None:
        return None
    if hasattr(past_legacy, "get_seq_length"):
        return past_legacy
    if _HAS_DYNAMIC_CACHE and isinstance(past_legacy, tuple):
        return DynamicCache.from_legacy_cache(past_legacy)
    return past_legacy


# ============================================================
# Global mapping context (token-level adapter selection)
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
# Dynamic LoRA Linear (per-sample adapter selection)
# ============================================================
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

        A = self.lora_As.index_select(0, adapter_mapping)  # [B, in, r]
        B = self.lora_Bs.index_select(0, adapter_mapping)  # [B, r, out]

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
        A = self.lora_As.data[slot_id]  # [in, r]
        B = self.lora_Bs.data[slot_id]  # [r, out]
        delta = (A @ B) * self.scaling  # [in, out]
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
# KV cache helpers (legacy tuple only)
# ============================================================
LegacyPast = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  # layers -> (k, v)


def _slice_past_for_sample(past_legacy: LegacyPast, sample_idx: int, seq_len: int) -> LegacyPast:
    out = []
    for k, v in past_legacy:
        ks = k[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        vs = v[sample_idx:sample_idx+1, :, -seq_len:, :].contiguous()
        out.append((ks, vs))
    return tuple(out)


def _left_pad_kv(k: torch.Tensor, v: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cur = k.shape[2]
    if cur == target_len:
        return k, v
    pad = target_len - cur
    k_pad = torch.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), device=k.device, dtype=k.dtype)
    v_pad = torch.zeros((v.shape[0], v.shape[1], pad, v.shape[3]), device=v.device, dtype=v.dtype)
    return torch.cat([k_pad, k], dim=2), torch.cat([v_pad, v], dim=2)


def _batch_past(past_list: List[LegacyPast], target_len: int) -> LegacyPast:
    n_layers = len(past_list[0])
    batched = []
    for layer in range(n_layers):
        ks = []
        vs = []
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
    def __init__(
        self,
        model_id: str,
        r: int = 16,
        alpha: int = 64,
        adapter_slots: int = 2,
        max_batch_size: int = 4,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        finished_results_maxlen: int = 10000,
        enable_monitor: bool = True,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            self.dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
        else:
            self.dtype = torch.float32

        self.enable_monitor = bool(enable_monitor)

        print(f"⏳ Loading model: {model_id} on {self.device} dtype={self.dtype} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully.")

        self.adapter_slots = int(adapter_slots)
        self.max_batch_size = int(max_batch_size)

        self.on_token: Optional[Callable[[str, int], None]] = None
        self.on_finish: Optional[Callable[[str, str], None]] = None

        self.cpu_cache: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

        self.gpu_slots: Dict[int, str] = {}          # slot_id -> adapter_id
        self.adapter_to_slot: Dict[str, int] = {}    # adapter_id -> slot_id
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))

        self.request_queue: List[Dict[str, Any]] = []
        self.running_queue: List[Dict[str, Any]] = []

        self.finished_results = deque(maxlen=int(finished_results_maxlen))

        self._replace_layers(r=r, alpha=alpha)

        self.current_merged_adapter: Optional[str] = None

    # ---------------------------
    # Monitoring
    # ---------------------------
    def _monitor_resources(self):
        if not self.enable_monitor:
            return
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / (1024**3)
        vram_info = ""
        if self.device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
            vram_info = f"| VRAM: {allocated_gb:.2f} GB (Reserved: {reserved_gb:.2f} GB)"
        print(f"[Process Monitor] RAM: {ram_gb:.2f} GB {vram_info}")

    # ---------------------------
    # Replace layers
    # ---------------------------
    def _replace_layers(self, r: int, alpha: int):
        target_names = {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"}
        replaced = 0
        for name, module in list(self.model.named_modules()):
            last = name.split(".")[-1]
            if last in target_names and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                new_layer = DynamicLoRALinear(module, self.adapter_slots, r, alpha)
                new_layer.to(self.device)
                setattr(parent, last, new_layer)
                replaced += 1
        print(f"🔧 Replaced {replaced} Linear layers with DynamicLoRALinear.")

    # ---------------------------
    # Adapter loading
    # ---------------------------
    def load_adapters_to_cpu(self, base_path: str = "./testLoRA"):
        adapters = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        self.cpu_cache.clear()

        for folder in adapters:
            aid = str(folder.split("_")[-1]) if "_" in folder else str(folder)
            st_path = os.path.join(base_path, folder, "adapter_model.safetensors")
            if not os.path.exists(st_path):
                print(f"⚠️ Skip {folder}: adapter_model.safetensors not found.")
                continue

            weights = load_file(st_path, device="cpu")
            self.cpu_cache[aid] = {}

            for n, m in self.model.named_modules():
                if isinstance(m, DynamicLoRALinear):
                    a_key = f"base_model.model.{n}.lora_A.weight"
                    b_key = f"base_model.model.{n}.lora_B.weight"
                    if a_key in weights and b_key in weights:
                        A = weights[a_key].T.contiguous()  # [in, r]
                        B = weights[b_key].T.contiguous()  # [r, out]
                        self.cpu_cache[aid][n] = {
                            "A": A.to(torch.float32).pin_memory(),
                            "B": B.to(torch.float32).pin_memory(),
                        }

        print(f"✅ Loaded {len(self.cpu_cache)} adapters into CPU cache. IDs: {list(self.cpu_cache.keys())}")

    # ---------------------------
    # Low-level: evict + load to a specific slot
    # ---------------------------
    def _evict_slot(self, slot_id: int):
        if slot_id in self.gpu_slots:
            old_aid = self.gpu_slots.pop(slot_id)
            self.adapter_to_slot.pop(old_aid, None)
        if slot_id in self.slot_lru:
            self.slot_lru.move_to_end(slot_id)

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        adapter_id = str(adapter_id)
        if adapter_id not in self.cpu_cache:
            raise KeyError(f"Adapter '{adapter_id}' not found in cpu_cache.")

        if slot_id in self.gpu_slots:
            self._evict_slot(slot_id)

        for n, m in self.model.named_modules():
            if isinstance(m, DynamicLoRALinear):
                target_dtype = m.lora_As.dtype
                target_device = m.lora_As.device
                if n in self.cpu_cache[adapter_id]:
                    A_cpu = self.cpu_cache[adapter_id][n]["A"]
                    B_cpu = self.cpu_cache[adapter_id][n]["B"]
                    m.lora_As.data[slot_id].copy_(A_cpu.to(dtype=target_dtype, device=target_device, non_blocking=True))
                    m.lora_Bs.data[slot_id].copy_(B_cpu.to(dtype=target_dtype, device=target_device, non_blocking=True))
                else:
                    m.lora_As.data[slot_id].zero_()
                    m.lora_Bs.data[slot_id].zero_()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id
        self.slot_lru.move_to_end(slot_id)

    # ---------------------------
    # Step-wise residency management (多退少補)
    # ---------------------------
    def _ensure_adapters_resident(self, required_adapters: List[str]):
        required = [str(a) for a in required_adapters]
        required_set = set(required)

        missing_cpu = [a for a in required if a not in self.cpu_cache]
        if missing_cpu:
            raise KeyError(f"Required adapters not in cpu_cache: {missing_cpu}")

        current_set = set(self.gpu_slots.values())

        # Evict extras
        extras = [aid for aid in current_set if aid not in required_set]
        for slot_id, aid in list(self.gpu_slots.items()):
            if aid in extras:
                self._evict_slot(slot_id)

        current_set = set(self.gpu_slots.values())
        missing = [aid for aid in required if aid not in current_set]
        if not missing:
            for aid in required:
                sid = self.adapter_to_slot.get(aid)
                if sid is not None:
                    self.slot_lru.move_to_end(sid)
            return

        if len(required_set) > self.adapter_slots:
            required = required[: self.adapter_slots]
            required_set = set(required)
            for slot_id, aid in list(self.gpu_slots.items()):
                if aid not in required_set:
                    self._evict_slot(slot_id)
            current_set = set(self.gpu_slots.values())
            missing = [aid for aid in required if aid not in current_set]

        occupied_slots = set(self.gpu_slots.keys())
        free_slots = [sid for sid in self.slot_lru.keys() if sid not in occupied_slots]

        mi = 0
        while mi < len(missing) and free_slots:
            aid = missing[mi]
            slot_id = free_slots.pop(0)
            self._load_adapter_to_slot(aid, slot_id)
            mi += 1

        while mi < len(missing):
            aid = missing[mi]
            lru_slot = next(iter(self.slot_lru))
            self._load_adapter_to_slot(aid, lru_slot)
            mi += 1

        for aid in required:
            sid = self.adapter_to_slot.get(aid)
            if sid is not None:
                self.slot_lru.move_to_end(sid)

    # ---------------------------
    # Merge / Unmerge
    # ---------------------------
    @torch.no_grad()
    def merge_adapter(self, adapter_id: str):
        adapter_id = str(adapter_id)
        if adapter_id not in self.cpu_cache:
            raise KeyError(f"Adapter '{adapter_id}' not in cpu_cache. Load first.")

        if adapter_id not in self.adapter_to_slot:
            slot_id = next(iter(self.slot_lru))
            self._load_adapter_to_slot(adapter_id, slot_id)
        slot_id = self.adapter_to_slot[adapter_id]

        if self.current_merged_adapter is not None and self.current_merged_adapter != adapter_id:
            self.unmerge_all()

        for m in self.model.modules():
            if isinstance(m, DynamicLoRALinear):
                m.manual_merge(slot_id)

        self.current_merged_adapter = adapter_id

    @torch.no_grad()
    def unmerge_all(self):
        for m in self.model.modules():
            if isinstance(m, DynamicLoRALinear):
                m.manual_unmerge()
        self.current_merged_adapter = None

    # ---------------------------
    # Add request
    # ---------------------------
    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 128):
        adapter_id = str(adapter_id)
        if adapter_id not in self.cpu_cache:
            raise KeyError(f"Adapter '{adapter_id}' not found in CPU cache. Load it first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)  # [1, L]
        prompt_len = int(input_ids.shape[1])

        self.request_queue.append({
            "request_id": str(request_id),
            "prompt": prompt,
            "adapter_id": adapter_id,

            "input_ids": input_ids,
            "seq_len": prompt_len,
            "past_key_values": None,

            "tokens_gen": [],
            "max_new_tokens": int(max_new_tokens),
            "done": False,
        })

    # ---------------------------
    # Scheduler (簡易)
    # ---------------------------
    def _fill_running_batch(self):
        self.running_queue = [r for r in self.running_queue if not r.get("done", False)]

        if self.current_merged_adapter is not None:
            kept = []
            for r in self.running_queue:
                if r["adapter_id"] == self.current_merged_adapter:
                    kept.append(r)
                else:
                    self.request_queue.insert(0, r)
            self.running_queue = kept

        i = 0
        while len(self.running_queue) < self.max_batch_size and i < len(self.request_queue):
            req = self.request_queue[i]
            aid = req["adapter_id"]

            if self.current_merged_adapter is not None and aid != self.current_merged_adapter:
                i += 1
                continue

            if self.current_merged_adapter is None:
                current_aids = {r["adapter_id"] for r in self.running_queue}
                if aid not in current_aids and len(current_aids) >= self.adapter_slots:
                    i += 1
                    continue

            self.running_queue.append(self.request_queue.pop(i))

    # ---------------------------
    # Finalize request
    # ---------------------------
    def _finalize_request(self, req: Dict[str, Any], reason: str):
        rid = req.get("request_id", "")
        self.finished_results.append({
            "request_id": rid,
            "adapter_id": req.get("adapter_id"),
            "tokens": list(req.get("tokens_gen", [])),
            "reason": reason,
        })

        req["past_key_values"] = None
        req["input_ids"] = None
        req["done"] = True

        if self.on_finish is not None and rid:
            try:
                self.on_finish(rid, reason)
            except Exception:
                pass

    # ---------------------------
    # Decode step (KV cache)
    # ---------------------------
    @torch.no_grad()
    def step(self) -> bool:
        self._monitor_resources()

        self._fill_running_batch()
        if not self.running_queue:
            return False

        if self.current_merged_adapter is not None:
            required = [self.current_merged_adapter]
        else:
            required = sorted({str(r["adapter_id"]) for r in self.running_queue})

        self._ensure_adapters_resident(required)

        prefill_reqs = [r for r in self.running_queue if r["past_key_values"] is None and not r.get("done", False)]
        decode_reqs = [r for r in self.running_queue if r["past_key_values"] is not None and not r.get("done", False)]

        did_work = False

        # ---------------------------
        # Prefill
        # ---------------------------
        if prefill_reqs:
            did_work = True

            if self.current_merged_adapter is None:
                mapping = torch.tensor(
                    [self.adapter_to_slot[r["adapter_id"]] for r in prefill_reqs],
                    dtype=torch.long,
                    device=self.device
                )
                LoRAContext.set_mapping(mapping)
            else:
                LoRAContext.set_mapping(None)

            input_list = [r["input_ids"] for r in prefill_reqs]
            max_len = max(x.shape[1] for x in input_list)
            B = len(input_list)

            padded = torch.full((B, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

            for i, ids in enumerate(input_list):
                L = ids.shape[1]
                padded[i, -L:] = ids[0]
                attn[i, -L:] = 1

            out = self.model(input_ids=padded, attention_mask=attn, use_cache=True)
            logits = out.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            past_legacy = _to_legacy_cache(out.past_key_values)

            for i, req in enumerate(prefill_reqs):
                tok = int(next_tokens[i].item())
                req["tokens_gen"].append(tok)

                rid = req.get("request_id", "")
                if self.on_token is not None and rid:
                    try:
                        self.on_token(rid, tok)
                    except Exception:
                        pass

                req_len = int(req["seq_len"])
                req["past_key_values"] = _slice_past_for_sample(past_legacy, i, req_len)

                req["input_ids"] = torch.cat([req["input_ids"], next_tokens[i:i+1].view(1, 1)], dim=-1)
                req["seq_len"] += 1

                if tok == self.tokenizer.eos_token_id:
                    self._finalize_request(req, reason="eos")
                elif len(req["tokens_gen"]) >= int(req["max_new_tokens"]):
                    self._finalize_request(req, reason="max_new_tokens")

            LoRAContext.set_mapping(None)

        # ---------------------------
        # Decode
        # ---------------------------
        if decode_reqs:
            did_work = True

            decode_reqs = [r for r in decode_reqs if not r.get("done", False)]
            if not decode_reqs:
                self.running_queue = [r for r in self.running_queue if not r.get("done", False)]
                return did_work

            if self.current_merged_adapter is None:
                mapping = torch.tensor(
                    [self.adapter_to_slot[r["adapter_id"]] for r in decode_reqs],
                    dtype=torch.long,
                    device=self.device
                )
                LoRAContext.set_mapping(mapping)
            else:
                LoRAContext.set_mapping(None)

            past_list: List[LegacyPast] = [r["past_key_values"] for r in decode_reqs]
            past_lens = [int(p[0][0].shape[2]) for p in past_list]
            target_past_len = max(past_lens)

            batched_past_legacy = _batch_past(past_list, target_past_len)
            batched_past_for_model = _to_model_cache(batched_past_legacy)

            last_tokens = torch.cat([r["input_ids"][:, -1:] for r in decode_reqs], dim=0)

            B = len(decode_reqs)
            total_len = target_past_len + 1
            attn = torch.zeros((B, total_len), dtype=torch.long, device=self.device)
            for i, pl in enumerate(past_lens):
                attn[i, -(pl + 1):] = 1

            out = self.model(
                input_ids=last_tokens,
                attention_mask=attn,
                past_key_values=batched_past_for_model,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            new_past_legacy = _to_legacy_cache(out.past_key_values)

            for i, req in enumerate(decode_reqs):
                tok = int(next_tokens[i].item())
                req["tokens_gen"].append(tok)

                rid = req.get("request_id", "")
                if self.on_token is not None and rid:
                    try:
                        self.on_token(rid, tok)
                    except Exception:
                        pass

                new_len = past_lens[i] + 1
                req["past_key_values"] = _slice_past_for_sample(new_past_legacy, i, new_len)

                req["input_ids"] = torch.cat([req["input_ids"], next_tokens[i:i+1].view(1, 1)], dim=-1)
                req["seq_len"] += 1

                if tok == self.tokenizer.eos_token_id:
                    self._finalize_request(req, reason="eos")
                elif len(req["tokens_gen"]) >= int(req["max_new_tokens"]):
                    self._finalize_request(req, reason="max_new_tokens")

            LoRAContext.set_mapping(None)

        self.running_queue = [r for r in self.running_queue if not r.get("done", False)]
        return did_work

    # ---------------------------
    # Status helpers
    # ---------------------------
    def is_idle(self) -> bool:
        return (len(self.request_queue) == 0) and (len(self.running_queue) == 0)

    def get_system_status(self) -> Dict[str, Any]:
        gpu_status = {f"Slot {k}": v for k, v in sorted(self.gpu_slots.items())}
        queue_status = {"Waiting": len(self.request_queue), "Running": len(self.running_queue), "Finished": len(self.finished_results)}
        merge_status = "Unmerged" if self.current_merged_adapter is None else f"Merged '{self.current_merged_adapter}'"
        return {"Merge Status": merge_status, "GPU Slots": gpu_status, "Queue Status": queue_status}

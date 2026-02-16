import os
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gc
import random
import csv
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. Global Context & LoRA Layer
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
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.adapter_slots = adapter_slots

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        if adapter_slots > 0:
            self.lora_As = nn.Parameter(torch.zeros(adapter_slots, base_layer.in_features, r, device=device, dtype=dtype))
            self.lora_Bs = nn.Parameter(torch.zeros(adapter_slots, r, base_layer.out_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_As, a=5**0.5)
            nn.init.zeros_(self.lora_Bs)
        else:
            self.lora_As = None
            self.lora_Bs = None

        self.is_merged = False
        self.merged_idx = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        
        # optimized path for merged or no-adapter mode
        if self.is_merged or self.adapter_slots == 0:
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
        if self.adapter_slots == 0: return
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
        if self.adapter_slots == 0: return
        if not self.is_merged: return
        W = self.base_layer.weight.data
        A = self.lora_As.data[self.merged_idx]
        B = self.lora_Bs.data[self.merged_idx]
        W.addmm_(B.T, A.T, alpha=-self.scaling)
        self.is_merged = False
        self.merged_idx = -1

# ============================================================
# 2. KV Cache Utilities (Fixed for Transformers v4.36+)
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
# 3. Multi-LoRA Engine
# ============================================================
class MultiLoRAEngine:
    def __init__(self, model_id: str, r: int = 16, alpha: int = 64, adapter_slots: int = 8, device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.adapter_slots = int(adapter_slots)
        self.max_cpu_loras = 50
        
        print(f"⏳ [Engine] Loading base model (FP16/BF16): {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=self.dtype, 
            low_cpu_mem_usage=True
        ).to(self.device).eval()
        
        self._replace_layers(r, alpha)

        self.adapter_paths: Dict[str, str] = {}
        self.cpu_cache: OrderedDict[str, Dict] = OrderedDict()
        self.gpu_slots: Dict[int, str] = {}
        self.adapter_to_slot: Dict[str, int] = {}
        self.slot_lru = OrderedDict((i, 0) for i in range(self.adapter_slots))
        
        self.request_queue: List[Dict] = []
        self.running_queue: List[Dict] = []
        self.current_merged_adapter: Optional[str] = None
        
        self.max_batch_size = 64

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

    def register_adapter(self, adapter_id: str, path: str):
        self.adapter_paths[adapter_id] = path

    def _ensure_cpu_loaded(self, adapter_id: str):
        if adapter_id in self.cpu_cache:
            self.cpu_cache.move_to_end(adapter_id)
            return
        if adapter_id not in self.adapter_paths:
            self.cpu_cache[adapter_id] = {} 
            return

        path = self.adapter_paths[adapter_id]
        while len(self.cpu_cache) >= self.max_cpu_loras:
            self.cpu_cache.popitem(last=False)
        
        try:
            weights = load_file(path, device="cpu")
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
            self.cpu_cache[adapter_id] = adapter_weights
        except Exception as e:
            print(f"❌ Failed to load {adapter_id}: {e}")
            self.cpu_cache[adapter_id] = {}

    def _evict_slot(self, slot_id: int):
        if slot_id in self.gpu_slots:
            old_adapter = self.gpu_slots.pop(slot_id)
            self.adapter_to_slot.pop(old_adapter, None)
        if slot_id in self.slot_lru:
            self.slot_lru.move_to_end(slot_id, last=False)

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        if self.adapter_slots == 0: return
        
        self._ensure_cpu_loaded(adapter_id)
        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id: 
            self._evict_slot(slot_id)
        
        if adapter_id in self.cpu_cache and self.cpu_cache[adapter_id]:
            for n, m in self.model.named_modules():
                if isinstance(m, DynamicLoRALinear) and n in self.cpu_cache[adapter_id]:
                    w = self.cpu_cache[adapter_id][n]
                    m.lora_As.data[slot_id].copy_(w["A"].to(m.lora_As.device, m.lora_As.dtype, non_blocking=True))
                    m.lora_Bs.data[slot_id].copy_(w["B"].to(m.lora_Bs.device, m.lora_Bs.dtype, non_blocking=True))
        
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id
        self.slot_lru.move_to_end(slot_id, last=True)

    def _ensure_adapters_resident(self, required_adapters: List[str]):
        if self.adapter_slots == 0: return
        missing = [aid for aid in required_adapters if aid not in self.adapter_to_slot]
        if not missing:
            for aid in required_adapters: self.slot_lru.move_to_end(self.adapter_to_slot[aid], last=True)
            return
        evictable_slots = [s for s in self.slot_lru if self.gpu_slots.get(s) not in set(required_adapters)]
        for aid in missing:
            if not evictable_slots: 
                evictable_slots = [next(iter(self.slot_lru))]
            self._load_adapter_to_slot(aid, evictable_slots.pop(0))

    @torch.no_grad()
    def merge_adapter(self, adapter_id: str):
        if self.adapter_slots == 0: return
        self._ensure_cpu_loaded(adapter_id)
        if adapter_id not in self.adapter_to_slot:
            self._load_adapter_to_slot(adapter_id, next(iter(self.slot_lru)))
        slot_id = self.adapter_to_slot[adapter_id]
        if self.current_merged_adapter and self.current_merged_adapter != adapter_id: 
            self.unmerge_all()
        if self.current_merged_adapter != adapter_id:
            for m in self.model.modules():
                if isinstance(m, DynamicLoRALinear): m.manual_merge(slot_id)
            self.current_merged_adapter = adapter_id

    @torch.no_grad()
    def unmerge_all(self):
        if self.current_merged_adapter:
            for m in self.model.modules():
                if isinstance(m, DynamicLoRALinear): m.manual_unmerge()
            self.current_merged_adapter = None

    def add_request(self, prompt: str, adapter_id: str, request_id: str, max_new_tokens: int = 128):
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

    @torch.no_grad()
    def step(self) -> bool:
        if not self.running_queue:
            while self.request_queue and len(self.running_queue) < self.max_batch_size:
                self.running_queue.append(self.request_queue.pop(0))
        
        if not self.running_queue: return False
        active_reqs = [r for r in self.running_queue if not r["done"]]
        if not active_reqs: 
            self.running_queue = [] 
            return False

        if self.current_merged_adapter or self.adapter_slots == 0:
            LoRAContext.set_mapping(None)
        else:
            required = sorted(list({r["adapter_id"] for r in active_reqs}))
            self._ensure_adapters_resident(required)

        prefill_reqs = [r for r in active_reqs if r["past_key_values"] is None]
        decode_reqs = [r for r in active_reqs if r["past_key_values"] is not None]

        if prefill_reqs:
            target_group = prefill_reqs
            if not self.current_merged_adapter and self.adapter_slots > 0:
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
            target_group = decode_reqs
            if not self.current_merged_adapter and self.adapter_slots > 0:
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

        logits = out.logits[:, -1, :]
        new_tokens = torch.argmax(logits, dim=-1)
        new_past_legacy = _to_legacy_cache(out.past_key_values)

        for i, req in enumerate(target_group):
            token_id = new_tokens[i].item()
            req["tokens_gen"].append(token_id)
            req["input_ids"] = torch.cat([req["input_ids"], new_tokens[i:i+1].view(1, 1)], dim=-1)
            
            current_len = req["seq_len"] + len(req["tokens_gen"])
            req["past_key_values"] = _slice_past_for_sample(new_past_legacy, i, current_len)
            
            if len(req["tokens_gen"]) >= req["max_new_tokens"]:
                req["done"] = True
        
        LoRAContext.set_mapping(None)
        return True

# ============================================================
# 4. Utilities
# ============================================================

def get_robust_mean(data: List[float]) -> float:
    if len(data) < 3: return np.mean(data)
    trimmed_data = sorted(data)[1:-1]
    return np.mean(trimmed_data)

def save_csv(filename: str, headers: List[str], rows: List[List[Any]]):
    if not os.path.exists("results"):
        os.makedirs("results")
    filepath = os.path.join("results", filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"   💾 Saved data to {filepath}")

def init_engine_for_experiment(model_id, adapter_path, slots=8):
    """Helper to cleanly init engine and load adapters"""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        engine = MultiLoRAEngine(model_id, r=16, alpha=64, adapter_slots=slots)
        adapter_ids = [f"lora_v{i}" for i in range(max(1, slots))]
        
        # Load adapters (or dummy)
        if os.path.exists(adapter_path):
            print("   📥 Loading real adapters...")
            for i, aid in enumerate(adapter_ids):
                engine.register_adapter(aid, adapter_path)
                engine._load_adapter_to_slot(aid, i % slots if slots > 0 else 0)
        else:
            print("   ⚠️ Loading dummy adapters...")
            for i, aid in enumerate(adapter_ids):
                engine._load_adapter_to_slot(aid, i % slots if slots > 0 else 0)
        return engine, adapter_ids
    except Exception as e:
        print(f"❌ Engine Init Failed: {e}")
        exit()

# ============================================================
# 5. Experiment 1: Full Scan (Batch Size 1-64)
# ============================================================
def run_experiment_1_full_scan(model_id, adapter_path) -> Dict[str, List[float]]:
    print("\n" + "="*50)
    print("🔬 EXPERIMENT 1: Overhead vs Batch Size")
    print("="*50)
    
    # Init Engine
    engine, adapter_ids = init_engine_for_experiment(model_id, adapter_path, slots=8)

    batch_sizes = [i for i in range(1, 65)]
    configs = [
        {"name": "Merged (Baseline)", "type": "merged", "n_lora": 1},
        {"name": "Unmerged (1 LoRA)", "type": "unmerged", "n_lora": 1},
    ]
    
    results = {cfg["name"]: [] for cfg in configs}
    TEST_STEPS = 10
    WARMUP_STEPS = 3
    
    csv_rows = []

    for cfg in configs:
        mode_name = cfg["name"]
        print(f"\nTesting Mode: {mode_name}")
        
        if cfg["type"] == "merged":
            engine.merge_adapter(adapter_ids[0])
        else:
            engine.unmerge_all()

        for bs in batch_sizes:
            latencies = []
            for rep in range(3):
                engine.running_queue = []
                engine.request_queue = []
                gc.collect()
                torch.cuda.empty_cache()
                engine.max_batch_size = bs
                
                n_lora_active = cfg["n_lora"]
                for i in range(bs):
                    aid = adapter_ids[0] if cfg["type"] == "merged" else adapter_ids[i % n_lora_active]
                    engine.add_request("Explain LoRA.", aid, f"req_{bs}_{rep}_{i}", TEST_STEPS + WARMUP_STEPS + 5)

                engine.step() 
                for _ in range(WARMUP_STEPS): engine.step()
                    
                t_accum = 0.0
                step_count = 0
                torch.cuda.synchronize()
                
                for _ in range(TEST_STEPS):
                    t0 = time.perf_counter()
                    did_work = engine.step()
                    torch.cuda.synchronize()
                    if did_work:
                        t_accum += (time.perf_counter() - t0) * 1000
                        step_count += 1
                
                if step_count > 0:
                    latencies.append(t_accum / step_count)

            if latencies:
                final_latency = np.median(latencies)
                results[mode_name].append(final_latency)
                csv_rows.append([mode_name, bs, final_latency])
                if bs % 8 == 0:
                    print(f"   Batch Size {bs:2d} | Latency: {final_latency:.2f} ms")
            else:
                results[mode_name].append(0.0)
                csv_rows.append([mode_name, bs, 0.0])

    save_csv("exp1_results.csv", ["Mode", "Batch_Size", "Latency_ms"], csv_rows)

    if not os.path.exists("plots"): os.makedirs("plots")
    plt.figure(figsize=(10, 6))
    for cfg in configs:
        plt.plot(batch_sizes, results[cfg["name"]], label=cfg["name"])
    plt.xlabel("Batch Size")
    plt.ylabel("Latency per Step (ms)")
    plt.title("Exp 1: LoRA Overhead")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/exp1_overhead.png")

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    return {
        "merged": results["Merged (Baseline)"],
        "unmerged_1_lora": results["Unmerged (1 LoRA)"]
    }

# ============================================================
# 6. Experiment 2: Latency Model
# ============================================================
def run_experiment_2_latency_model(model_id, adapter_path, external_decode_data: Optional[Dict] = None):
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 2: Latency Model Parameter Estimation")
    print("="*60)
    
    # Init Engine
    engine, adapter_ids = init_engine_for_experiment(model_id, adapter_path, slots=8)

    print("\n[Part A] Estimating Prefill Linear Model (BS=1)")
    input_lengths = list(range(8, 513, 64)) 
    avg_prefill_times = []
    
    engine.merge_adapter(adapter_ids[0]) 
    engine.max_batch_size = 1

    for L in input_lengths:
        dummy_prompt = "A " * L
        engine.running_queue = []
        engine.request_queue = []
        torch.cuda.empty_cache()
        engine.add_request(dummy_prompt, adapter_ids[0], "prefill_test", 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step() 
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        avg_prefill_times.append(dt)
        print(f"   Input Len {L:4d} | Time: {dt:.2f} ms")

    coeffs = np.polyfit(input_lengths, avg_prefill_times, 1)
    alpha_pre = coeffs[0]
    alpha_base_pre = coeffs[1]
    
    print("\n[Part B] Gamma Calculation")
    if external_decode_data:
        merged = external_decode_data["merged"]
        unmerged = external_decode_data["unmerged_1_lora"]
        gammas = []
        for tm, tu in zip(merged, unmerged):
            if tm > 0: gammas.append((tu / tm) - 1)
        avg_gamma = np.mean(gammas) if gammas else 0.0
    else:
        avg_gamma = 0.26
    
    print(f"   Estimated alpha_pre: {alpha_pre:.4f}")
    print(f"   Estimated gamma: {avg_gamma:.4f}")
    
    save_csv("exp2_parameters.csv", ["Parameter", "Value"], [
        ["alpha_pre", alpha_pre],
        ["alpha_base", alpha_base_pre],
        ["gamma", avg_gamma]
    ])
    
    # Cleanup
    del engine
    torch.cuda.empty_cache()

# ============================================================
# 7. Experiment 3: Max Capacity Search (Slots vs Batch Size)
# ============================================================
def _try_run_batch(engine, adapter_ids, bs, input_len, output_len):
    """
    Helper function: Tries to run a specific batch size.
    Returns: True if successful, False if OOM.
    """
    try:
        # Reset States
        engine.running_queue = []
        engine.request_queue = []
        gc.collect()
        torch.cuda.empty_cache()
        
        # Prepare Prompt
        base_text = "The quick brown fox jumps over the lazy dog. " * (input_len // 5 + 10)
        tokens = engine.tokenizer(base_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokens) < input_len: tokens = torch.cat([tokens, tokens])
        tokens = tokens[:input_len]
        dummy_prompt = engine.tokenizer.decode(tokens)
        
        engine.max_batch_size = bs
        
        # Queue Requests
        for i in range(bs):
            aid = adapter_ids[i % len(adapter_ids)] if adapter_ids else "dummy"
            engine.add_request(dummy_prompt, aid, f"try_{bs}_{i}", output_len)
            
        # Run Loop
        steps = 0
        while True:
            did_work = engine.step()
            # torch.cuda.synchronize() # Optional: sync for accurate timing, but slows down search
            
            if not did_work:
                break
            
            steps += 1
            if steps >= output_len:
                break
                
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False # OOM Detected
        raise e # Other errors should crash
    except Exception as e:
        raise e

def run_experiment_3_max_capacity(model_id, adapter_path):
    print("\n" + "="*60)
    print("🔥 EXPERIMENT 3: Max Batch Size vs. LoRA Slots")
    print("="*60)
    
    # 測試這幾種 Slot 設定
    # 0 = 純 Base Model (最大空間)
    # 64 = 極端 Multi-LoRA (佔用大量 VRAM)
    test_slot_counts = [i for i in range(17)] 
    
    results = []
    
    for slots in test_slot_counts:
        print(f"\n[Configuration] Testing with {slots} LoRA Slots...")
        
        # 1. 初始化 Engine (每次都重新分配記憶體)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            engine = MultiLoRAEngine(model_id, r=16, alpha=64, adapter_slots=slots)
            
            # 準備 Adapter IDs
            if slots > 0:
                adapter_ids = [f"lora_v{i}" for i in range(slots)]
                # Load dummy weights just to fill VRAM
                for i, aid in enumerate(adapter_ids):
                    # We skip actual file loading to save time, 
                    # relying on engine's dummy init (zeros) which takes same VRAM
                    engine._load_adapter_to_slot(aid, i)
            else:
                adapter_ids = []
                
        except Exception as e:
            print(f"❌ Failed to init engine with {slots} slots: {e}")
            continue

        # 2. 二分搜尋最大 Batch Size
        # 範圍：1 ~ 128 (對於 24G 顯卡，通常極限在 64 附近)
        low = 1
        high = 32
        max_safe_bs = 0
        
        INPUT_LEN = 512
        OUTPUT_LEN = 256
        
        print(f"   🔍 Searching Max Batch Size (Binary Search)...")
        
        while low <= high:
            mid = (low + high) // 2
            print(f"      Trying BS={mid}...", end="\r")
            
            success = _try_run_batch(engine, adapter_ids, mid, INPUT_LEN, OUTPUT_LEN)
            
            if success:
                print(f"      Trying BS={mid} -> ✅ Pass")
                max_safe_bs = mid
                low = mid + 1
            else:
                print(f"      Trying BS={mid} -> ❌ OOM ")
                high = mid - 1
                
        # 3. 記錄結果
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   🏆 Result: Slots={slots} | Max BS={max_safe_bs} | Peak VRAM={peak_mem:.2f} GB")
        results.append([slots, max_safe_bs, peak_mem])
        
        # Cleanup
        del engine
        torch.cuda.empty_cache()
        
    # 4. 繪圖與存檔
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Slots':<10} | {'Max Batch Size':<15} | {'Gain':<10}")
    print("-" * 45)
    
    base_bs = 0
    for r in results:
        slots, bs, _ = r
        if slots == test_slot_counts[-1]: base_bs = bs # Compare against max slots
        
    for r in results:
        slots, bs, mem = r
        gain = f"+{bs - base_bs}" if base_bs > 0 else "0"
        print(f"{slots:<10} | {bs:<15} | {gain:<10}")
        
    save_csv("exp3_max_capacity.csv", ["Slots", "Max_BS", "Peak_VRAM_GB"], results)
    
    # Plotting
    slots_x = [r[0] for r in results]
    bs_y = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(slots_x, bs_y, 'o-', linewidth=3, markersize=10, color='purple')
    
    for x, y in zip(slots_x, bs_y):
        plt.text(x, y + 1, f"{y}", ha='center', fontsize=12, fontweight='bold')
        
    plt.xlabel("Reserved LoRA Slots")
    plt.ylabel("Max Supported Batch Size")
    plt.title(f"Trade-off: VRAM Reservation vs Throughput\n(Llama-3.1-8B, SeqLen={INPUT_LEN}+{OUTPUT_LEN})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().invert_xaxis() # 讓 0 在右邊或左邊看習慣，通常希望看 Slots 減少 -> BS 增加
    plt.savefig("plots/exp3_capacity_tradeoff.png")
    print("   💾 Saved plot to plots/exp3_capacity_tradeoff.png")

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check drivers.")
        exit()

    MODEL_ID = "unsloth/Meta-Llama-3.1-8B" 
    ADAPTER_PATH = "./testLoRA/LoRA_1/adapter_model.safetensors" 
    
    # --- RUN EXPERIMENTS ---
    EXPERIMENT_TO_RUN = [3] # Adjust as needed
    
    exp1_data = None
    
    if 1 in EXPERIMENT_TO_RUN:
        exp1_data = run_experiment_1_full_scan(MODEL_ID, ADAPTER_PATH)
        
    if 2 in EXPERIMENT_TO_RUN:
        run_experiment_2_latency_model(MODEL_ID, ADAPTER_PATH, external_decode_data=exp1_data)

    if 3 in EXPERIMENT_TO_RUN:
        # 3a. Merged Mode (Simulate Base Model Only)
        run_experiment_3_max_capacity(MODEL_ID, ADAPTER_PATH)
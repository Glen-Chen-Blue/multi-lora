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

        self.lora_As = nn.Parameter(torch.zeros(adapter_slots, base_layer.in_features, r, device=device, dtype=dtype))
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
# 2. KV Cache Utilities
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
        print(f"🔧 [Engine] Replaced {replaced_count} layers with DynamicLoRALinear.")

    def register_adapter(self, adapter_id: str, path: str):
        self.adapter_paths[adapter_id] = path

    def _ensure_cpu_loaded(self, adapter_id: str):
        if adapter_id in self.cpu_cache:
            self.cpu_cache.move_to_end(adapter_id)
            return
        if adapter_id not in self.adapter_paths:
            raise KeyError(f"Adapter {adapter_id} not registered.")
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
            if adapter_weights:
                self.cpu_cache[adapter_id] = adapter_weights
        except Exception as e:
            print(f"❌ Failed to load {adapter_id}: {e}")
            raise e

    def _evict_slot(self, slot_id: int):
        if slot_id in self.gpu_slots:
            old_adapter = self.gpu_slots.pop(slot_id)
            self.adapter_to_slot.pop(old_adapter, None)
        if slot_id in self.slot_lru:
            self.slot_lru.move_to_end(slot_id, last=False)

    @torch.no_grad()
    def _load_adapter_to_slot(self, adapter_id: str, slot_id: int):
        self._ensure_cpu_loaded(adapter_id)
        if slot_id in self.gpu_slots and self.gpu_slots[slot_id] != adapter_id: 
            self._evict_slot(slot_id)
        for n, m in self.model.named_modules():
            if isinstance(m, DynamicLoRALinear) and n in self.cpu_cache[adapter_id]:
                w = self.cpu_cache[adapter_id][n]
                m.lora_As.data[slot_id].copy_(w["A"].to(m.lora_As.device, m.lora_As.dtype, non_blocking=True))
                m.lora_Bs.data[slot_id].copy_(w["B"].to(m.lora_Bs.device, m.lora_Bs.dtype, non_blocking=True))
        self.gpu_slots[slot_id] = adapter_id
        self.adapter_to_slot[adapter_id] = slot_id
        self.slot_lru.move_to_end(slot_id, last=True)

    def _ensure_adapters_resident(self, required_adapters: List[str]):
        missing = [aid for aid in required_adapters if aid not in self.adapter_to_slot]
        if not missing:
            for aid in required_adapters: self.slot_lru.move_to_end(self.adapter_to_slot[aid], last=True)
            return
        evictable_slots = [s for s in self.slot_lru if self.gpu_slots.get(s) not in set(required_adapters)]
        for aid in missing:
            if not evictable_slots: raise RuntimeError("No available slots!")
            self._load_adapter_to_slot(aid, evictable_slots.pop(0))

    @torch.no_grad()
    def merge_adapter(self, adapter_id: str):
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

        if self.current_merged_adapter:
            LoRAContext.set_mapping(None)
        else:
            required = sorted(list({r["adapter_id"] for r in active_reqs}))
            self._ensure_adapters_resident(required)

        prefill_reqs = [r for r in active_reqs if r["past_key_values"] is None]
        decode_reqs = [r for r in active_reqs if r["past_key_values"] is not None]

        if prefill_reqs:
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
    """Utility to save data to CSV"""
    filepath = os.path.join("results", filename)
    if not os.path.exists("results"):
        os.makedirs("results")
    
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"   💾 Saved data to {filepath}")

# ============================================================
# 5. Experiment 1: Full Scan & Data Generation
# ============================================================
def run_experiment_1_full_scan(engine: MultiLoRAEngine, adapter_ids: List[str]) -> Dict[str, List[float]]:
    """
    Run Exp 1 (1-64 batch size).
    Returns: Dictionary containing smoothed data for 'merged' and 'unmerged_1_lora'.
    """
    print("\n" + "="*50)
    print("🔬 EXPERIMENT 1: Overhead vs Batch Size (Dense Scan 1-64)")
    print("="*50)

    batch_sizes = [i for i in range(1, 65)]
    configs = [
        {"name": "Merged (Baseline)", "type": "merged", "n_lora": 1},
        {"name": "Unmerged (1 LoRA)", "type": "unmerged", "n_lora": 1},
        {"name": "Unmerged (2 LoRAs)", "type": "unmerged", "n_lora": 2},
        {"name": "Unmerged (4 LoRAs)", "type": "unmerged", "n_lora": 4},
        {"name": "Unmerged (8 LoRAs)", "type": "unmerged", "n_lora": 8},
    ]
    
    results = {cfg["name"]: [] for cfg in configs}
    TEST_STEPS = 20
    WARMUP_STEPS = 5
    REPEAT_COUNT = 5 
    
    csv_rows = []

    for cfg in configs:
        mode_name = cfg["name"]
        print(f"\n========================================")
        print(f"🚀 Testing Mode: {mode_name}")
        print(f"========================================")
        
        if cfg["type"] == "merged":
            print("🔄 Merging Adapter...")
            engine.merge_adapter(adapter_ids[0])
        else:
            print("🔄 Unmerging All...")
            engine.unmerge_all()

        for bs in batch_sizes:
            latencies_for_current_bs = []

            for rep in range(REPEAT_COUNT):
                engine.running_queue = []
                engine.request_queue = []
                gc.collect()
                torch.cuda.empty_cache()
                engine.max_batch_size = bs
                
                n_lora_active = cfg["n_lora"]
                for i in range(bs):
                    if cfg["type"] == "merged":
                        target_aid = adapter_ids[0]
                    else:
                        target_aid = adapter_ids[i % n_lora_active]
                    engine.add_request("Explain LoRA.", target_aid, f"req_{bs}_{rep}_{i}", TEST_STEPS + WARMUP_STEPS)

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
                    latencies_for_current_bs.append(t_accum / step_count)

            if latencies_for_current_bs:
                final_latency = np.median(latencies_for_current_bs)
                results[mode_name].append(final_latency)
                csv_rows.append([mode_name, bs, final_latency])
                print(f"   Batch Size {bs:2d} | Median Latency: {final_latency:.2f} ms")
            else:
                results[mode_name].append(0.0)
                csv_rows.append([mode_name, bs, 0.0])

    save_csv("exp1_results.csv", ["Mode", "Batch_Size", "Latency_ms"], csv_rows)

    print("\n📊 Generating Exp 1 Plot...")
    plt.figure(figsize=(12, 7))
    colors = cm.viridis(np.linspace(0, 0.9, len(configs)))
    
    for idx, cfg in enumerate(configs):
        name = cfg["name"]
        data = results[name]
        
        if "Merged" in name:
            style = 's-'
            lw = 3
            color = 'black'
        else:
            style = 'o--'
            lw = 2
            color = colors[idx]
            
        plt.plot(batch_sizes, data, style, label=name, linewidth=lw, color=color)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Latency per Step (ms)")
    plt.title(f"Dynamic LoRA Overhead (Native GEMM FP16)")
    
    sparse_ticks = [1] + [i for i in range(8, 65, 8)]
    valid_ticks = [t for t in sparse_ticks if t <= max(batch_sizes)]
    plt.xticks(valid_ticks)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("lora_overhead_exp1.png")
    print("✅ Saved to lora_overhead_exp1.png")

    # [Important] Return data for Exp 2 to use
    return {
        "merged": results["Merged (Baseline)"],
        "unmerged_1_lora": results["Unmerged (1 LoRA)"]
    }

# ============================================================
# 6. Experiment 2: Latency Model (Using Smooth Data from Exp 1)
# ============================================================
def run_experiment_2_latency_model(engine: MultiLoRAEngine, adapter_ids: List[str], external_decode_data: Optional[Dict] = None):
    """
    Experiment 2: High-Resolution Latency Model Parameter Estimation & Validation
    If external_decode_data is provided, it uses that for Part B instead of re-measuring.
    """
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 2: Latency Model Parameter Estimation")
    print("="*60)
    
    REPEAT_COUNT = 10 
    if not os.path.exists("plots"): os.makedirs("plots")

    # ==========================================
    # Part A: Estimate alpha_pre (Prefill Rate)
    # ==========================================
    print("\n[Part A] Estimating Prefill Linear Model (Batch Size=1)")
    input_lengths = list(range(8, 1025, 8)) 
    avg_prefill_times = []
    
    engine.merge_adapter(adapter_ids[0]) 
    engine.max_batch_size = 1

    for L in input_lengths:
        dummy_prompt = "A " * L 
        inputs = engine.tokenizer(dummy_prompt, return_tensors="pt")
        real_L = inputs.input_ids.shape[1]
        
        times_for_L = []
        for _ in range(REPEAT_COUNT):
            engine.running_queue = []
            engine.request_queue = []
            torch.cuda.empty_cache()
            engine.add_request(dummy_prompt, adapter_ids[0], "prefill_test", 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.step() 
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            times_for_L.append(dt)
        
        avg_t = get_robust_mean(times_for_L)
        avg_prefill_times.append(avg_t)
        
        if L % 64 == 0: 
            print(f"   Input Length {real_L:4d} | Robust Mean Prefill: {avg_t:.2f} ms")

    if len(avg_prefill_times) > 1:
        coeffs = np.polyfit(input_lengths, avg_prefill_times, 1)
        alpha_pre = coeffs[0]
        alpha_base_pre = coeffs[1]
        print(f"   => Estimated alpha_pre: {alpha_pre:.4f} ms/token")
        print(f"   => Estimated alpha_base (intercept): {alpha_base_pre:.4f} ms")

        plt.figure(figsize=(10, 6))
        plt.scatter(input_lengths, avg_prefill_times, alpha=0.5, s=10, label='Measured (Trimmed Mean)', color='blue')
        x_fit = np.array(input_lengths)
        y_fit = np.polyval(coeffs, x_fit)
        plt.plot(x_fit, y_fit, 'r--', label=f'Model: {alpha_pre:.3f}x + {alpha_base_pre:.1f}', linewidth=2)
        plt.xlabel("Input Tokens ($N_{prefill}$)", fontsize=12)
        plt.ylabel("Prefill Latency (ms)", fontsize=12)
        plt.title("Fig 1: High-Res Prefill Latency Linear Fit", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/exp2_fig1_prefill_fit.png")
        print("   ✅ [Fig 1] Saved plots/exp2_fig1_prefill_fit.png")

    # ==========================================
    # Part B: Estimate f_dec(n) (Decoding Scaling)
    # ==========================================
    print("\n[Part B] Estimating Decoding Scaling f_dec(n_active)")
    batch_sizes = list(range(1, 65))
    
    # [KEY MODIFICATION] Use external smooth data if available
    if external_decode_data:
        print("   >>> Using SMOOTH data from Experiment 1 (Median filtered) <<<")
        decode_times_merged = external_decode_data["merged"]
        decode_times_unmerged = external_decode_data["unmerged_1_lora"]
        
        # Ensure lengths match
        if len(decode_times_merged) != len(batch_sizes):
            print("   ⚠️ Warning: Data length mismatch. Truncating/Adjusting...")
            min_len = min(len(decode_times_merged), len(batch_sizes))
            decode_times_merged = decode_times_merged[:min_len]
            decode_times_unmerged = decode_times_unmerged[:min_len]
            batch_sizes = batch_sizes[:min_len]
            
    else:
        # Re-measure if no data passed (Previous robust mean logic)
        print("   >>> Measuring Decoding Latency (Robust Mean) <<<")
        decode_times_merged = []
        decode_times_unmerged = []
        
        # 1. Merged
        engine.merge_adapter(adapter_ids[0])
        for bs in batch_sizes:
            times_for_bs = []
            for _ in range(5): # Reduced for speed if fallback
                engine.running_queue = []; engine.request_queue = []; torch.cuda.empty_cache(); engine.max_batch_size = bs
                for i in range(bs): engine.add_request("Short prompt", adapter_ids[0], f"dec_{bs}_{i}", 20)
                engine.step()
                t_accum = 0; steps = 0; torch.cuda.synchronize()
                for _ in range(10): 
                    t0 = time.perf_counter(); did = engine.step(); torch.cuda.synchronize()
                    if did: t_accum += (time.perf_counter() - t0)*1000; steps += 1
                times_for_bs.append(t_accum/steps if steps>0 else 0)
            decode_times_merged.append(get_robust_mean(times_for_bs))

        # 2. Unmerged
        engine.unmerge_all()
        for bs in batch_sizes:
            times_for_bs = []
            for _ in range(5):
                engine.running_queue = []; engine.request_queue = []; torch.cuda.empty_cache(); engine.max_batch_size = bs
                for i in range(bs): engine.add_request("Short prompt", adapter_ids[0], f"dec_u_{bs}_{i}", 20)
                engine.step()
                t_accum = 0; steps = 0; torch.cuda.synchronize()
                for _ in range(10):
                    t0 = time.perf_counter(); did = engine.step(); torch.cuda.synchronize()
                    if did: t_accum += (time.perf_counter() - t0)*1000; steps += 1
                times_for_bs.append(t_accum/steps if steps>0 else 0)
            decode_times_unmerged.append(get_robust_mean(times_for_bs))

    # --- Plot 2 ---
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, decode_times_merged, '-', color='black', label='Merged (Baseline)')
    plt.plot(batch_sizes, decode_times_unmerged, '--', color='#E24A33', label='Unmerged (Multi-LoRA)')
    # Since data comes from Exp1 (Median), points are already smooth, scatter might be redundant but okay
    plt.scatter(batch_sizes, decode_times_merged, s=10, color='black', alpha=0.5)
    plt.scatter(batch_sizes, decode_times_unmerged, s=10, color='#E24A33', alpha=0.5)
    plt.xlabel("Batch Size ($n_{active}$)")
    plt.ylabel("Decode Latency per Step (ms)")
    plt.title("Fig 2: Dense Decoding Latency Scaling (Using Exp1 Data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/exp2_fig2_decode_scaling.png")
    print("   ✅ [Fig 2] Saved plots/exp2_fig2_decode_scaling.png")

    # ==========================================
    # Part C: Gamma Analysis & Model Validation
    # ==========================================
    print("\n[Part C] Gamma Analysis & Model Validation")
    gammas = []
    for tm, tu in zip(decode_times_merged, decode_times_unmerged):
        if tm > 0:
            g = (tu / tm) - 1
            gammas.append(g)
        else:
            gammas.append(0)
    avg_gamma = np.mean(gammas)
    print(f"   => Average Gamma: {avg_gamma:.4f}")

    # --- Plot 3 ---
    plt.figure(figsize=(10, 5))
    plt.scatter(batch_sizes, gammas, color='skyblue', edgecolor='blue', alpha=0.6, s=30)
    plt.axhline(y=avg_gamma, color='r', linestyle='--', linewidth=2, label=f'Mean $\gamma$ = {avg_gamma:.3f}')
    plt.xlabel("Batch Size")
    plt.ylabel("Overhead Ratio ($\gamma$)")
    plt.title("Fig 3: Stability of LoRA Overhead ($\gamma$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/exp2_fig3_gamma_stability.png")
    print("   ✅ [Fig 3] Saved plots/exp2_fig3_gamma_stability.png")

    # --- Plot 4 ---
    predicted_unmerged = [tm * (1 + avg_gamma) for tm in decode_times_merged]
    plt.figure(figsize=(10, 6))
    plt.scatter(batch_sizes, decode_times_unmerged, color='red', s=15, label='Measured Unmerged (Exp1 Data)', zorder=5)
    plt.plot(batch_sizes, predicted_unmerged, 'b--', label=f'Model Prediction\n($T_{{merge}} \\times {1+avg_gamma:.2f}$)', linewidth=2)
    plt.fill_between(batch_sizes, decode_times_unmerged, predicted_unmerged, color='gray', alpha=0.2, label='Error')
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("Fig 4: Mathematical Model Validation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/exp2_fig4_model_validation.png")
    print("   ✅ [Fig 4] Saved plots/exp2_fig4_model_validation.png")

    # [CSV] Save Model Validation Data (Fig 4)
    validation_rows = []
    for bs, meas, pred in zip(batch_sizes, decode_times_unmerged, predicted_unmerged):
        error = abs(meas - pred) / meas if meas > 0 else 0
        validation_rows.append([bs, meas, pred, error, 1+avg_gamma])
    save_csv("exp2_validation.csv", ["Batch_Size", "Measured_Unmerged_ms", "Predicted_ms", "Error_Ratio", "Theta"], validation_rows)

    # ==========================================
    # Part D: Random Workload Validation (Safe Mode)
    # ==========================================
    print("\n[Part D] End-to-End Random Workload Validation")
    MAX_SAFE_TOKENS = 3000 
    NUM_RANDOM_TESTS = 1000
    test_cases = []
    for _ in range(NUM_RANDOM_TESTS):
        bs = random.randint(1, 64)
        max_allowed_len = MAX_SAFE_TOKENS // bs
        if max_allowed_len < 10:
            bs = random.randint(1, 32)
            max_allowed_len = MAX_SAFE_TOKENS // bs
        upper_bound = min(max_allowed_len, 512)
        seq_len = random.randint(8, upper_bound)
        test_cases.append((bs, seq_len))
    
    measured_latencies = []
    predicted_latencies = []
    random_test_rows = [] # [CSV]
    
    engine.unmerge_all()
    # Lookup using Exp 1 smoothed data
    f_dec_map = {bs: time for bs, time in zip(batch_sizes, decode_times_merged)}

    print(f"   Generated {NUM_RANDOM_TESTS} safe test cases...")

    for i, (bs, seq_len) in enumerate(test_cases):
        if bs not in f_dec_map: continue 
        
        # 1. Prediction
        f_dec_val = f_dec_map[bs]
        N_prefill_total = bs * seq_len
        pred_prefill = alpha_base_pre + alpha_pre * N_prefill_total
        pred_decode = f_dec_val 
        total_pred = (pred_prefill + pred_decode) * (1 + avg_gamma)
        predicted_latencies.append(total_pred)

        # 2. Measurement
        engine.running_queue = []
        engine.request_queue = []
        gc.collect()
        torch.cuda.empty_cache()
        engine.max_batch_size = bs
        dummy_prompt = "A " * seq_len
        for k in range(bs):
            aid = adapter_ids[k % min(len(adapter_ids), 8)] 
            engine.add_request(dummy_prompt, aid, f"rand_{i}_{k}", 2) 
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            engine.step() # Prefill
            engine.step() # Decode
            torch.cuda.synchronize()
            total_measured = (time.perf_counter() - t0) * 1000
            measured_latencies.append(total_measured)
            
            # [CSV] Add row
            error = abs(total_measured - total_pred) / total_measured
            random_test_rows.append([bs, seq_len, bs*seq_len, total_pred, total_measured, error])
            
            print(f"   Test {i:2d}: BS={bs:2d}, L={seq_len:3d} -> Pred: {total_pred:6.1f}ms, Meas: {total_measured:6.1f}ms")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   Test {i:2d}: SKIPPED (OOM)")
                torch.cuda.empty_cache()
                predicted_latencies.pop()
            else:
                raise e

    # [CSV] Save Random Test Results
    save_csv("exp2_random_test.csv", ["Batch_Size", "Seq_Len", "Total_Tokens", "Predicted_ms", "Measured_ms", "Error_Ratio"], random_test_rows)

    # --- Plot 5: Random Validation ---
    if measured_latencies:
        plt.figure(figsize=(8, 8))
        plt.scatter(predicted_latencies, measured_latencies, alpha=0.6, color='purple', s=40, label='Test Cases')
        min_val = min(min(predicted_latencies), min(measured_latencies))
        max_val = max(max(predicted_latencies), max(measured_latencies))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')
        mape = np.mean(np.abs((np.array(measured_latencies) - np.array(predicted_latencies)) / np.array(measured_latencies))) * 100
        plt.xlabel("Predicted Latency (ms)", fontsize=12)
        plt.ylabel("Measured Latency (ms)", fontsize=12)
        plt.title(f"Fig 5: End-to-End Validation (MAPE={mape:.1f}%)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/exp2_fig5_random_validation.png")
        print(f"   ✅ [Fig 5] Saved plots/exp2_fig5_random_validation.png (MAPE={mape:.1f}%)")

    # [CSV] Save Parameters
    param_rows = [
        ["alpha_pre (ms/token)", alpha_pre],
        ["alpha_base (ms)", alpha_base_pre],
        ["gamma", avg_gamma],
        ["theta_unmerged", 1 + avg_gamma]
    ]
    save_csv("exp2_parameters.csv", ["Parameter", "Value"], param_rows)

    print("\n" + "="*60)
    print("📝 FINAL LATENCY MODEL PARAMETERS")
    print("="*60)
    print(f"1. Prefill Parameters:")
    print(f"   α_pre  = {alpha_pre:.6f} ms/token")
    print(f"   α_base = {alpha_base_pre:.4f} ms")
    print("-" * 30)
    print(f"2. LoRA Overhead:")
    print(f"   γ (Gamma) = {avg_gamma:.4f}")
    print("="*60)

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        exit()

    MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
    ADAPTER_PATH = "./testLoRA/LoRA_1/adapter_model.safetensors"
    
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Error: Adapter path not found: {ADAPTER_PATH}")
        exit()

    SLOT_COUNT = 8
    engine = MultiLoRAEngine(MODEL_ID, r=16, alpha=64, adapter_slots=SLOT_COUNT)
    
    adapter_ids = [f"lora_v{i}" for i in range(SLOT_COUNT)]
    print("📥 Loading adapters into slots...")
    for i, aid in enumerate(adapter_ids):
        engine.register_adapter(aid, ADAPTER_PATH)
        engine._load_adapter_to_slot(aid, i)

    # 1 = Overhead vs Batch Size (Original) -> NOW RETURNS DATA
    # 2 = Latency Model Parameters (Accepts External Data)
    EXPERIMENT_TO_RUN = [1, 2]
    
    exp1_data = None
    if 1 in EXPERIMENT_TO_RUN:
        # Run Exp 1 and capture data
        exp1_data = run_experiment_1_full_scan(engine, adapter_ids)
        
    if 2 in EXPERIMENT_TO_RUN:
        # Pass Exp 1 data to Exp 2
        run_experiment_2_latency_model(engine, adapter_ids, external_decode_data=exp1_data)
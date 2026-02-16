import os
import torch
import torch.nn as nn
import time
import gc
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# ============================================================
# Core Engine Components
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
        nn.init.kaiming_uniform_(self.lora_As, a=5**0.5)
        nn.init.zeros_(self.lora_Bs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.is_merged: return base_out

        mapping = LoRAContext.get_mapping()
        if mapping is None: return base_out
        if mapping.device != self.lora_As.device: mapping = mapping.to(self.lora_As.device)
        x_lora = x.to(self.lora_As.dtype)
        A_sel = self.lora_As.index_select(0, mapping)
        B_sel = self.lora_Bs.index_select(0, mapping)
        if x_lora.dim() == 2:
            lora_h = torch.einsum("bi,bir->br", x_lora, A_sel)
            lora_out = torch.einsum("br,bro->bo", lora_h, B_sel)
        else:
            lora_h = torch.einsum("bti,bir->btr", x_lora, A_sel)
            lora_out = torch.einsum("btr,bro->bto", lora_h, B_sel)
        return base_out + (lora_out.to(base_out.dtype) * self.scaling)

def _left_pad_kv(k, v, target_len):
    cur = k.shape[2]
    if cur >= target_len: return k, v
    pad = target_len - cur
    k_pad = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3], device=k.device, dtype=k.dtype)
    v_pad = torch.zeros(v.shape[0], v.shape[1], pad, v.shape[3], device=v.device, dtype=v.dtype)
    return torch.cat([k_pad, k], dim=2), torch.cat([v_pad, v], dim=2)

class MultiLoRAEngine:
    def __init__(self, model_id: str, adapter_slots: int, lora_rank: int = 32, device: str = "cuda"):
        self.device = torch.device(device)
        self.adapter_slots = adapter_slots
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [Engine] Loading Model (Slots={adapter_slots})...", end="\r")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self._replace_layers(r=lora_rank, alpha=lora_rank*2)
        self.request_queue = []
        self.running_queue = []
        self.adapter_to_slot = {}
        for i in range(adapter_slots): self.adapter_to_slot[f"lora_{i}"] = i

    def _replace_layers(self, r, alpha):
        target_suffixes = {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "o_proj"}
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and name.split(".")[-1] in target_suffixes:
                parent = self.model.get_submodule(".".join(name.split(".")[:-1]))
                new_layer = DynamicLoRALinear(module, self.adapter_slots, r, alpha).to(self.device)
                setattr(parent, name.split(".")[-1], new_layer)

    def add_request(self, prompt, adapter_id, req_id, max_new=128, forced_input_len=None):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        if forced_input_len is not None:
            curr_len = input_ids.shape[1]
            if curr_len < forced_input_len:
                pad_len = forced_input_len - curr_len
                pad_token = self.tokenizer.pad_token_id or 0
                padding = torch.full((1, pad_len), pad_token, dtype=input_ids.dtype)
                input_ids = torch.cat([padding, input_ids], dim=1)
            elif curr_len > forced_input_len:
                input_ids = input_ids[:, -forced_input_len:]
        self.request_queue.append({
            "id": req_id, "adapter_id": adapter_id, "input_ids": input_ids.to(self.device),
            "seq_len": input_ids.shape[1], "past_key_values": None, "tokens_gen": [], "max_new": max_new, "done": False
        })

    @torch.no_grad()
    def step(self, batch_limit):
        self.running_queue = [r for r in self.running_queue if not r["done"]]
        while len(self.running_queue) < batch_limit and self.request_queue:
            self.running_queue.append(self.request_queue.pop(0))
        if not self.running_queue: return False
        target = self.running_queue
        
        if target[0]["past_key_values"] is None:
            max_len = max(r["input_ids"].shape[1] for r in target)
            ids = torch.full((len(target), max_len), self.tokenizer.pad_token_id or 0, device=self.device)
            mask = torch.zeros((len(target), max_len), device=self.device)
            for i, r in enumerate(target):
                l = r["input_ids"].shape[1]
                ids[i, -l:] = r["input_ids"]
                mask[i, -l:] = 1
            past = None
        else:
            ids = torch.cat([r["input_ids"][:, -1:] for r in target], dim=0)
            p_list = [r["past_key_values"] for r in target]
            max_p = max(p[0][0].shape[2] for p in p_list)
            batched_p = []
            for layer_idx in range(len(p_list[0])):
                ks = [p[layer_idx][0] for p in p_list]
                vs = [p[layer_idx][1] for p in p_list]
                k_cat = torch.cat([_left_pad_kv(k, v, max_p)[0] for k, v in zip(ks, vs)], dim=0)
                v_cat = torch.cat([_left_pad_kv(k, v, max_p)[1] for k, v in zip(ks, vs)], dim=0)
                batched_p.append((k_cat, v_cat))
            past = DynamicCache.from_legacy_cache(tuple(batched_p))
            mask = torch.ones((len(target), max_p + 1), device=self.device)

        slot_ids = [self.adapter_to_slot[r["adapter_id"]] for r in target]
        LoRAContext.set_mapping(torch.tensor(slot_ids, device=self.device))
        try:
            out = self.model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
            new_toks = torch.argmax(out.logits[:, -1, :], dim=-1)
            new_past = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
            for i, r in enumerate(target):
                r["tokens_gen"].append(new_toks[i].item())
                r["input_ids"] = torch.cat([r["input_ids"], new_toks[i:i+1].view(1, 1)], dim=-1)
                r["past_key_values"] = tuple((layer[0][i:i+1], layer[1][i:i+1]) for layer in new_past)
                if len(r["tokens_gen"]) >= r["max_new"]: r["done"] = True
            return True
        finally:
            LoRAContext.set_mapping(None)

# ============================================================
# Test: 1 LoRA + 19 Requests (Unmerged)
# ============================================================
def test_unmerge_ratio():
    if not torch.cuda.is_available(): return

    MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
    TARGET_BASE_MEM_MB = 16384  # 16 GB Base Lock
    
    # 驗證參數
    NUM_LORAS = 1    # 1 Page
    BATCH_SIZE = 19  # 19 Pages
    TOTAL_PAGES = NUM_LORAS + BATCH_SIZE # 20 Pages
    
    # Pool Limit (假設 Pool 為 25)
    POOL_CAPACITY = 25 
    
    print("\n" + "="*80)
    print(f"🧪 UNMERGED RATIO VERIFICATION")
    print(f"   Config: {NUM_LORAS} LoRA + {BATCH_SIZE} Requests")
    print(f"   Pages Used: {TOTAL_PAGES} / {POOL_CAPACITY}")
    print(f"   Buffer Reserved: {POOL_CAPACITY - TOTAL_PAGES} Pages (Spillover)")
    print("="*80)

    torch.cuda.empty_cache()
    gc.collect()
    
    engine = None
    padding_tensor = None

    try:
        # 1. Initialize
        engine = MultiLoRAEngine(MODEL_ID, adapter_slots=NUM_LORAS, lora_rank=32)
        
        # 2. Lock Base Memory
        torch.cuda.empty_cache()
        curr = torch.cuda.memory_allocated()
        lora_size = sum(m.lora_As.nbytes + m.lora_Bs.nbytes for m in engine.model.modules() if isinstance(m, DynamicLoRALinear))
        pad_bytes = (TARGET_BASE_MEM_MB * 1024**2) - (curr - lora_size)
        if pad_bytes > 0:
            padding_tensor = torch.empty(int(pad_bytes), dtype=torch.uint8, device='cuda')
        
        print(f"   ▶️  Running 19 Concurrent Requests...")
        
        # 3. Add Requests
        prompt = "Hello " * 100
        engine.request_queue = []
        engine.running_queue = []
        torch.cuda.reset_peak_memory_stats()
        
        # 單一 LoRA ID (lora_0)
        lid = "lora_0"
        
        for i in range(BATCH_SIZE):
            engine.add_request(prompt, lid, f"req_{i}", max_new=256, forced_input_len=512)
        
        # 4. Step
        steps = 0
        while True:
            not_empty = engine.step(batch_limit=BATCH_SIZE)
            if not not_empty: break
            steps += 1
            if steps > 5: break 
        
        peak = torch.cuda.max_memory_allocated() / 1024**2
        managed = TARGET_BASE_MEM_MB + (TOTAL_PAGES * 160)
        
        print(f"   ✅ PASS | Peak VRAM: {peak:.0f} MB")
        print(f"           | Managed: {managed:.0f} MB (Base + {TOTAL_PAGES} Pages)")
        print(f"           | Activations: {peak - managed:.0f} MB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ❌ FAIL | OOM Triggered")
        else:
            print(f"   ❌ ERROR| {e}")
    finally:
        if engine: del engine
        if padding_tensor is not None: del padding_tensor
        torch.cuda.empty_cache()

    print("="*80)

if __name__ == "__main__":
    test_unmerge_ratio()
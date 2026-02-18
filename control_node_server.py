import os
import time
import uuid
import asyncio
import httpx
import logging
import copy
from collections import deque, defaultdict
from typing import Dict, List, Optional, Deque, Any, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# ============================================================
# Config & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CONTROL: %(message)s")
logger = logging.getLogger("ControlNode")

LORA_PATH = os.environ.get("LORA_PATH", "./testLoRA")
# 這裡設定允許的 Adapter ID，對應您測試腳本或真實資料夾名稱
ALLOWED_ADAPTERS = ["LoRA_1", "LoRA_2", "LoRA_3"] 
INITIAL_NODES = [x.strip() for x in os.environ.get("COMPUTE_NODES", "http://127.0.0.1:8001").split(",") if x.strip()]

limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
client = httpx.AsyncClient(limits=limits, timeout=60.0)

# ============================================================
# Global State
# ============================================================
class NodeManager:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {} 
        for url in INITIAL_NODES:
            self.register_node(url)

    def register_node(self, url: str):
        if url not in self.nodes:
            # metrics 結構參考 compute_node_server 回傳格式
            self.nodes[url] = {"metrics": None, "last_seen": 0}
            logger.info(f"✅ Registered Node: {url}")

    def update_metrics(self, url: str, metrics: Dict):
        if url in self.nodes:
            self.nodes[url]["metrics"] = metrics
            self.nodes[url]["last_seen"] = time.time()

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "active_nodes": len([n for n, d in self.nodes.items() if d.get("metrics")]),
            "nodes_detail": self.nodes
        }

node_mgr = NodeManager()
request_queues: Dict[str, Deque] = defaultdict(deque)
stream_queues: Dict[str, asyncio.Queue] = {}

# ============================================================
# SP2 Scheduling Logic Helpers & Virtual State
# ============================================================

class VirtualNodeState:
    """
    用於在單次 Scheduling Window 中模擬節點狀態
    避免因 Metrics 回報延遲導致的超發 (Over-commitment)
    """
    def __init__(self, url: str, metrics: Dict):
        self.url = url
        self.mode = metrics.get("mode", "unmerge") # "merge" or "unmerge"
        
        # Load Info
        self.running_batch = metrics["load"]["running_batch"]
        
        # LoRA Info
        self.merged_adapter = metrics["lora_state"].get("merged_adapter")
        # 複製一份 Set，因為我們會在模擬過程中修改它
        self.active_loras = set(metrics["lora_state"].get("running_adapters", []))
        
        # Capacity Config (Hardcoded rules from SP2)
        self.capacity_merged = 15
        self.capacity_unmerged_base = 12 

    def get_free_slots(self, target_lora: str) -> int:
        """
        計算針對特定 LoRA，該節點還能吃多少 Request
        """
        if self.mode == "merge":
            # Merged Mode: 只接受對應的 LoRA
            if self.merged_adapter == target_lora:
                return max(0, self.capacity_merged - self.running_batch)
            else:
                return 0 # 專用節點不接客
        else:
            # Unmerged Mode: Cost = Running Requests + Unique LoRAs
            # 公式: Capacity >= running_batch + len(active_loras)
            
            current_unique_count = len(self.active_loras)
            
            # 判斷這個 LoRA 是否是新的
            is_new_lora = target_lora not in self.active_loras
            # Cost 計算邏輯：
            # 如果是新的，需要 2 Slots (1 Request + 1 Context overhead)
            # 如果是舊的，需要 1 Slot (1 Request)
            
            current_total_cost = self.running_batch + current_unique_count
            margin = self.capacity_unmerged_base - current_total_cost
            
            if is_new_lora:
                if margin >= 2:
                    return margin - 1 # 扣掉 Context 佔用的 1 格
                else:
                    return 0
            else:
                return max(0, margin)

    def commit_request(self, target_lora: str):
        """
        虛擬派發：更新狀態
        """
        self.running_batch += 1
        self.active_loras.add(target_lora)

# ============================================================
# State Transition Helpers
# ============================================================

async def do_merge_node(url: str, adapter_id: str):
    """發送 Merge 指令"""
    try:
        logger.info(f"🔄 [State Change] Triggering MERGE on {url} for {adapter_id}")
        # 注意：Compute Node 的 merge 實作通常會自動 unmerge，但這裡直接呼叫 merge 即可
        resp = await client.post(f"{url}/merge", json={"adapter_id": adapter_id, "force": False})
        if resp.status_code == 200:
            logger.info(f"✅ {url} merged successfully.")
        else:
            logger.warning(f"⚠️ Merge failed on {url}: {resp.text}")
    except Exception as e:
        logger.error(f"❌ Merge request error {url}: {e}")

async def do_unmerge_node(url: str):
    """發送 Unmerge 指令"""
    try:
        logger.info(f"🔄 [State Change] Triggering UNMERGE on {url}")
        resp = await client.post(f"{url}/unmerge", json={"force": False})
        if resp.status_code == 200:
            logger.info(f"✅ {url} unmerged successfully.")
    except Exception as e:
        logger.error(f"❌ Unmerge request error {url}: {e}")

# ============================================================
# Background Tasks
# ============================================================

async def metrics_poller():
    while True:
        for url in list(node_mgr.nodes.keys()):
            try:
                resp = await client.get(f"{url}/metrics", timeout=5.0)
                if resp.status_code == 200:
                    node_mgr.update_metrics(url, resp.json())
            except Exception:
                pass
        await asyncio.sleep(1)

async def dispatch_task(node_url: str, req_data: dict):
    rid = req_data["request_id"]
    user_q = stream_queues.get(rid)
    try:
        payload = {
            "prompt": req_data["prompt"],
            "adapter_id": req_data["adapter_id"],
            "max_new_tokens": req_data["max_new_tokens"]
        }
        async with client.stream("POST", f"{node_url}/add_request", json=payload) as response:
            if response.status_code != 200:
                logger.error(f"❌ Dispatch failed to {node_url}: {response.status_code}")
                if user_q: await user_q.put({"type": "error", "message": f"Node error {response.status_code}"})
                return
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if content and user_q: await user_q.put(content)
    except Exception as e:
        logger.error(f"❌ Error during dispatch to {node_url}: {e}")
        if user_q: await user_q.put({"type": "error", "message": str(e)})
    finally:
        if user_q: await user_q.put(None) 

# ============================================================
# Scheduler Loop (Core SP2 Logic with Anti-Starvation)
# ============================================================

async def scheduler_loop():
    logger.info("⏳ SP2-Aligned Scheduler loop started (1s interval).")
    
    while True:
        # 1. 建立虛擬狀態快照 (Snapshot)
        virtual_nodes: List[VirtualNodeState] = []
        for url, data in node_mgr.nodes.items():
            if data.get("metrics"):
                virtual_nodes.append(VirtualNodeState(url, data["metrics"]))
        
        if not virtual_nodes:
            await asyncio.sleep(1.0)
            continue

        # 計算目前的狀態統計
        total_pending_count = sum(len(q) for q in request_queues.values())
        # [防鎖死] 統計目前 Unmerge 的節點數量
        unmerged_node_count = sum(1 for n in virtual_nodes if n.mode == "unmerge")

        # =========================================================
        # Phase 0: Mode Switching Logic (Auto Merge/Unmerge)
        # =========================================================
        
        for v_node in virtual_nodes:
            # --- Case A: Check for MERGE Trigger ---
            # 條件: Unmerged Mode 且 (Batch >= 10) 且 (Unique LoRA == 1)
            # 意義: 節點已經被單一 LoRA 塞滿，轉換為 Dedicated 可提升容量 (12 -> 15)
            if v_node.mode == "unmerge":
                # [保護機制] 確保 Merge 後至少還有 1 個 Unmerge 節點活著，避免餓死新進 LoRA
                if unmerged_node_count > 1 and v_node.running_batch >= 10 and len(v_node.active_loras) == 1:
                    target_lora = list(v_node.active_loras)[0]
                    # 優化: 只有當該 LoRA 還有待處理請求時，Merge 才有價值
                    if len(request_queues[target_lora]) > 0:
                        logger.info(f"🔥 [Auto-Merge] Node {v_node.url} saturated with {target_lora}. Merging! (Unmerged Left: {unmerged_node_count-1})")
                        asyncio.create_task(do_merge_node(v_node.url, target_lora))
                        v_node.mode = "switching" 
                        unmerged_node_count -= 1 # 扣除計數

            # --- Case B: Check for UNMERGE Trigger ---
            # 條件: Merged Mode 且 (Batch < 10) 且 (本 LoRA 無排隊) 且 (有其他 LoRA 在排隊)
            # 意義: 專用節點負載降低，且外部有需求，釋放資源回歸共享池
            elif v_node.mode == "merge":
                my_lora = v_node.merged_adapter
                my_queue_len = len(request_queues[my_lora])
                others_pending = total_pending_count - my_queue_len
                
                if v_node.running_batch < 10 and my_queue_len == 0 and others_pending > 0:
                    logger.info(f"🧊 [Auto-Unmerge] Node {v_node.url} cooling down. Others waiting ({others_pending}). Unmerging!")
                    asyncio.create_task(do_unmerge_node(v_node.url))
                    v_node.mode = "switching"
                    unmerged_node_count += 1 # 增加計數

        # =========================================================
        # Phase 1: Hot Dispatch (Cost = 1)
        # =========================================================
        # 優先派發給已經載入該 LoRA 的節點
        
        active_adapters = [aid for aid, q in request_queues.items() if len(q) > 0]
        
        for aid in active_adapters:
            queue = request_queues[aid]
            for v_node in virtual_nodes:
                if v_node.mode == "switching": continue 
                
                while len(queue) > 0:
                    # Hit 判斷
                    is_merged_hit = (v_node.mode == "merge" and v_node.merged_adapter == aid)
                    is_unmerged_hit = (v_node.mode == "unmerge" and aid in v_node.active_loras)
                    
                    if not (is_merged_hit or is_unmerged_hit): break
                    
                    if v_node.get_free_slots(aid) > 0:
                        req = queue.popleft()
                        logger.info(f"⚡ [Hot] {req['request_id'][:8]} ({aid}) -> {v_node.url}")
                        v_node.commit_request(aid)
                        asyncio.create_task(dispatch_task(v_node.url, req))
                    else:
                        break

        # =========================================================
        # Phase 2: Cold Dispatch (Cost = 2)
        # =========================================================
        # 處理剩餘請求，這會增加 Unique LoRA
        
        remaining_adapters = [aid for aid, q in request_queues.items() if len(q) > 0]
        
        # [排序策略變更] Oldest Request First (Fairness)
        # 依照每個 Adapter 佇列中 "最老請求" 的等待時間排序
        # 這樣即使是冷門 LoRA，只要等得夠久，也能優先獲得分配新節點的權利
        remaining_adapters.sort(key=lambda aid: request_queues[aid][0]['arrival_time'])
        
        for aid in remaining_adapters:
            queue = request_queues[aid]
            if len(queue) == 0: continue
            
            # 尋找候選節點 (只選 Unmerged)
            candidates = []
            for v_node in virtual_nodes:
                if v_node.mode == "switching": continue
                # 必須是 Unmerged 且還沒載入該 LoRA
                if v_node.mode == "unmerge" and aid not in v_node.active_loras:
                    if v_node.get_free_slots(aid) > 0: # 隱含檢查了是否有 2 slots
                        candidates.append(v_node)
            
            if not candidates: continue
            
            # 排序: 優先選 Shared Node (entropy 高)，反碎片化
            candidates.sort(key=lambda n: (len(n.active_loras), n.get_free_slots(aid)), reverse=True)
            target_node = candidates[0]
            
            # 集中派發 (Bundling)
            # 在這個 Time Window 內，鎖定這個節點為該 LoRA 的服務點
            # 只要節點還有空位，就繼續塞同一個 LoRA 的請求
            dispatched_count = 0
            while len(queue) > 0:
                if target_node.get_free_slots(aid) > 0:
                    req = queue.popleft()
                    logger.info(f"❄️ [Cold] {req['request_id'][:8]} ({aid}) -> {target_node.url} (Waited: {time.time()-req['arrival_time']:.2f}s)")
                    target_node.commit_request(aid)
                    asyncio.create_task(dispatch_task(target_node.url, req))
                    dispatched_count += 1
                else:
                    break # 這個節點滿了，換下一個 Adapter (公平性：不獨佔所有空閒節點)

        await asyncio.sleep(1.0)

# ============================================================
# API Endpoints
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(LORA_PATH, exist_ok=True)
    asyncio.create_task(metrics_poller())
    asyncio.create_task(scheduler_loop())
    yield
    await client.aclose()

app = FastAPI(title="Control Node (SP2 Complete)", lifespan=lifespan)

class AddRequest(BaseModel):
    prompt: str
    adapter_id: str
    max_new_tokens: int = 128

@app.post("/register_node")
async def register_node(request: Request):
    data = await request.json()
    url = data.get("control_node_url") or str(request.base_url)
    node_mgr.register_node(url)
    return {"status": "registered", "url": url}

@app.get("/status")
async def status():
    return node_mgr.get_status_snapshot()

@app.post("/send_request")
async def send_request(req: AddRequest):
    rid = str(uuid.uuid4())
    stream_queues[rid] = asyncio.Queue()
    request_queues[req.adapter_id].append({
        "request_id": rid,
        "prompt": req.prompt,
        "adapter_id": req.adapter_id,
        "max_new_tokens": req.max_new_tokens,
        "arrival_time": time.time() # [新增] 用於防飢餓排序
    })
    logger.info(f"📥 Received Request {rid} for {req.adapter_id}. Queued.")
    return {"request_id": rid}

@app.get("/stream/{request_id}")
async def stream(request_id: str):
    if request_id not in stream_queues: raise HTTPException(404, "Request ID not found")
    q = stream_queues[request_id]
    async def event_generator():
        yield "event: open\ndata: connected\n\n"
        try:
            while True:
                data = await q.get()
                if data is None: 
                    yield "event: end\ndata: [DONE]\n\n"
                    break
                if isinstance(data, dict):
                    import json
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"data: {data}\n\n"
        finally:
            stream_queues.pop(request_id, None)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/fetch_adapter/{adapter_id}")
async def fetch_adapter(adapter_id: str):
    target_path = os.path.join(LORA_PATH, adapter_id, "adapter_model.safetensors")
    if os.path.exists(target_path):
        return FileResponse(target_path, media_type="application/octet-stream", filename="adapter_model.safetensors")
    logger.error(f"❌ Adapter not found: {target_path}")
    raise HTTPException(404, f"Adapter {adapter_id} not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

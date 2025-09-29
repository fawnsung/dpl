import asyncio
import httpx
import uvicorn
import threading
import json
from decouple import config
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import uuid
import time
import platform
import re

# --- 集中化配置 ---
# 在实际应用中，这些配置应该来自 .env 文件或其他配置源。
class Settings:
    GATEWAY_PORT = 8000
    # 使用Tailscale替代ngrok - 更稳定的内网穿透方案
    NODES = [
        {
            "id": 1, "name": "Node 1 (Host 1)",
            "monitor_base_url": "http://127.0.0.1:8001",  # Tailscale IP - 主机1监控API
            "llm_url": "http://127.0.0.1:11434/api/chat",  # Ollama Chat API（替换 LM Studio）
        },
        {
            "id": 2, "name": "Node 2 (Host 2)",
            "monitor_base_url": "http://localhost:51083",  # Tailscale IP - 主机2监控API
            "llm_url": "http://localhost:51067/api/chat",  # Tailscale IP - 主机2 LM Studio
        },
        {
            "id": 3, "name": "Node 3 (Host 3)",
            "monitor_base_url": "http://localhost:57487",  # Tailscale IP - 主机3监控API
            "llm_url": "http://localhost:54365/api/chat",  # Ollama Chat API（替换 LM Studio）
        },
    ]
    MONITOR_INTERVAL = 5 # Monitor polling interval (seconds)

settings = Settings()

# --- 状态管理 ---
# This dictionary will hold the real-time status of all nodes. It is the only source of truth for the entire application.
NODE_STATUS_CACHE = {
    node["id"]: {"id": node["id"], "name": node["name"], "online": False, "busy": False, "metrics": None}
    for node in settings.NODES
}
# Use thread-safe lock for cache operations to handle concurrent access.
CACHE_LOCK = threading.Lock()

# This dictionary will hold the CPU model of all nodes.
CPU_INFO_CACHE = {}
# --- End of new additions ---

# --- New additions: Dataset batch processing task management ---
# This dictionary will hold the status of all dataset processing tasks.
# The key is the task ID, the value is the detailed information of the task (status, progress, results, etc.).
DATASET_JOBS = {}
JOBS_LOCK = threading.Lock()
# --- End of new additions ---

# --- New additions: Alert management ---
# This list will save the current active alerts.
ALERTS_LIST = []
ALERTS_LOCK = threading.Lock()
# --- End of new additions ---

# --- FastAPI application initialization ---
app = FastAPI()

# --- Add CORS support ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you may need to restrict specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a reusable HTTP client
client = httpx.AsyncClient(timeout=10.0)

# --- 数据模型定义 ---
class ChatRequest(BaseModel):
    messages: list
    model: Optional[str] = None # Allow the frontend to specify the model to use
    stream: bool = True

# --- New additions: Data model for task status ---
class JobStatus(BaseModel):
    job_id: str
    status: str
    total_items: int
    processed_items: int
    start_time: float
    end_time: Optional[float] = None
    node_reports: dict # Record the number of items processed and the time taken by each node

# --- Background monitoring task ---
async def fetch_single_node_status(node_config):
    """Asynchronously fetch the status of a single node."""
    node_id = node_config["id"]
    url = f"{node_config['monitor_base_url']}/status"
    print(f"[DEBUG] Trying to connect: {url}")
    try:
        response = await client.get(url)
        print(f"[DEBUG] Node {node_id} response code: {response.status_code}")

        if response.status_code == 200:
            try:
                metrics = response.json()
                with CACHE_LOCK:
                    if not NODE_STATUS_CACHE[node_id]["online"]:
                        print(f"Node {node_id} status restored: online.")
                    NODE_STATUS_CACHE[node_id].update(online=True, metrics=metrics)
                    # Uniformly cache CPU model, compatible with different fields
                    cpu_model = metrics.get("cpu_model") or metrics.get("cpu_info")
                    if cpu_model:
                        CPU_INFO_CACHE[node_id] = cpu_model
                    print(f"[CACHE_UPDATE] Node {node_id} updated, cache status: {NODE_STATUS_CACHE[node_id]}")
            except json.JSONDecodeError:
                print(f"[ERROR] Node {node_id} responded with 200 OK, but returned invalid JSON.")
                print(f"[ERROR] Original response content from {url}: {response.text}")
                with CACHE_LOCK:
                    if NODE_STATUS_CACHE[node_id]["online"]:
                        print(f"Node {node_id} turned offline due to data format error.")
                    NODE_STATUS_CACHE[node_id].update(online=False, metrics=None)
        else:
            with CACHE_LOCK:
                 if NODE_STATUS_CACHE[node_id]["online"]:
                     print(f"Node {node_id} now offline. Reason: HTTP status code {response.status_code}")
                 NODE_STATUS_CACHE[node_id].update(online=False, metrics=None)

    except httpx.RequestError as e:
        with CACHE_LOCK:
            if NODE_STATUS_CACHE[node_id]["online"] or not NODE_STATUS_CACHE[node_id].get("last_error"):
                 error_msg = f"{type(e).__name__}"
                 print(f"Node {node_id} connection failed turned offline. Reason: {error_msg}")
                 NODE_STATUS_CACHE[node_id]["last_error"] = error_msg
            NODE_STATUS_CACHE[node_id].update(online=False, metrics=None)

async def monitor_nodes_periodically():
    """Periodically execute the monitoring task for all nodes."""
    while True:
        await asyncio.gather(*(fetch_single_node_status(node) for node in settings.NODES))
        await asyncio.sleep(settings.MONITOR_INTERVAL)

# --- API 端点 ---
@app.on_event("startup")
async def startup_event():
    """When the server starts, start the background monitoring and alert checking tasks."""
    asyncio.create_task(monitor_nodes_periodically())
    asyncio.create_task(alert_checker_periodically())

@app.get("/api/status/all")
async def get_all_statuses():
    """Get the latest status cache of all nodes, and merge CPU model information."""
    with CACHE_LOCK:
        statuses = list(NODE_STATUS_CACHE.values())
        for status in statuses:
            if status["online"] and status["metrics"]:
                # Get CPU model from cache and add to response
                status["metrics"]["cpu_model"] = CPU_INFO_CACHE.get(status["id"], "未知处理器")
        print(f"[API_READ] /api/status/all data returned to frontend: {statuses}")
        return statuses

@app.get("/api/alerts")
async def get_alerts():
    """Get the current active alert list."""
    with ALERTS_LOCK:
        return list(ALERTS_LIST)

@app.get("/api/models")
async def get_available_models():
    """Get all available model lists, including running and launchable models."""
    running_models = set()
    all_available_models = set()
    
    with CACHE_LOCK:
        for status in NODE_STATUS_CACHE.values():
            metrics = status.get("metrics") or {}
            if status.get("online") and metrics:
                # Get the current running model
                if metrics.get("model_id"):
                    running_models.add(metrics.get("model_id"))
                
                # Get all available models on the node
                available_models = metrics.get("available_models", [])
                if available_models:
                    all_available_models.update(available_models)
    
        # If no available model list is obtained, try to get it from the Ollama API
    if not all_available_models:
        try:
            for node_config in settings.NODES:
                if node_config["id"] in [s["id"] for s in NODE_STATUS_CACHE.values() if s["online"]]:
                    # Try to get the model list from the Ollama API
                    ollama_url = node_config["llm_url"].replace("/api/chat", "/api/tags")
                    try:
                        response = await client.get(ollama_url, timeout=5.0)
                        if response.status_code == 200:
                            data = response.json()
                            models = data.get("models", [])
                            for model in models:
                                model_name = model.get("name", "")
                                if model_name:
                                    all_available_models.add(model_name)
                    except Exception as e:
                        print(f"[WARNING] Unable to get model list from node {node_config['id']}: {e}")
        except Exception as e:
            print(f"[WARNING] Error getting model list: {e}")
    
    # If still no model, return some common model names as fallback
    if not all_available_models:
        all_available_models = {
            "llama3:8b", "llama3:7b", "llama2:7b", "codellama:7b", 
            "mistral:7b", "qwen:7b", "gemma:7b", "phi3:3.8b"
        }
    
    # Return model information, including running status
    models_info = []
    for model in sorted(all_available_models):
        models_info.append({
            "name": model,
            "running": model in running_models,
            "display_name": model.split(":")[0] if ":" in model else model
        })
    
    return models_info

@app.post("/api/models/start/{model_name}")
async def start_model(model_name: str):
    """Start the specified model."""
    # 找到可用的节点
    available_node = None
    with CACHE_LOCK:
        for node_id, status in NODE_STATUS_CACHE.items():
            if status["online"] and not status["metrics"].get("locked", False):
                config = next((n for n in settings.NODES if n["id"] == node_id), None)
                if config:
                    available_node = config
                    break
    
    if not available_node:
        raise HTTPException(status_code=503, detail="No available node to start model")
    
    try:
        upstream_url = available_node["llm_url"]
        provider = detect_upstream_provider_by_url(upstream_url)

        async def warm_up(url: str, provider_key: str):
            if provider_key in ("ollama_chat", "ollama_generate", "unknown"):
                # 对 Ollama 系列，使用 /api/generate 预热最稳妥
                gen_url = url.replace("/api/chat", "/api/generate") if "/api/chat" in url else (
                    url if "/api/generate" in url else url.rstrip("/") + "/api/generate"
                )
                payload = {"model": model_name, "prompt": "warmup", "stream": False}
                return await client.post(gen_url, json=payload, timeout=30.0)
            else:
                # OpenAI 兼容: 直接调用 chat/completions 非流式
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "stream": False
                }
                return await client.post(url, json=payload, timeout=30.0)

        response = await warm_up(upstream_url, provider)
        if response.status_code in (404, 405, 501) and provider == "ollama_chat":
            # 回退到 generate
            fallback_url = upstream_url.replace("/api/chat", "/api/generate")
            response = await client.post(fallback_url, json={
                "model": model_name, "prompt": "warmup", "stream": False
            }, timeout=30.0)

        if response.status_code == 200:
            # After successful startup, actively refresh the node's monitoring status
            try:
                await fetch_single_node_status(available_node)
                print(f"[INFO] Model {model_name} started successfully, node {available_node['id']} status refreshed")
            except Exception as e:
                print(f"[WARNING] Failed to refresh node status after startup: {e}")
            
            return {
                "status": "success", 
                "message": f"Model {model_name} started successfully on node {available_node['id']}",
                "node_id": available_node["id"]
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to start model: {response.text}")
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error when starting model: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start model: {str(e)}")

@app.get("/api/cluster/overview")
async def get_cluster_overview():
    """Aggregate per-node metrics for cluster overview charts."""
    with CACHE_LOCK:
        nodes = list(NODE_STATUS_CACHE.values())

    labels = []
    cpu = []
    memory = []
    nic_total_mbps = []
    disk_total_MBps = []

    for node in nodes:
        labels.append(node.get("name") or f"Host {node.get('id')}")
        metrics = node.get("metrics") or {}
        cpu.append(metrics.get("cpu_usage_percent", 0) or 0)
        mem_obj = metrics.get("memory") or {}
        memory.append(mem_obj.get("percent", 0) or 0)
        net_obj = metrics.get("network") or {}
        nic_total_mbps.append(net_obj.get("total_mbps", 0) or 0)
        disk_obj = metrics.get("disk") or {}
        disk_total_MBps.append(disk_obj.get("total_MBps", 0) or 0)

    return {
        "labels": labels,
        "cpu_percent": cpu,
        "memory_percent": memory,
        "nic_total_mbps": nic_total_mbps,
        "disk_total_MBps": disk_total_MBps,
    }

async def lock_node(node_config):
    """Send lock request to the specified node."""
    try:
        response = await client.post(f"{node_config['monitor_base_url']}/lock")
        return response.status_code == 200
    except httpx.RequestError:
        return False

async def unlock_node(node_config):
    """Send unlock request to the specified node."""
    try:
        response = await client.post(f"{node_config['monitor_base_url']}/unlock")
        if response.status_code == 200:
            print(f"[INFO] Node {node_config['id']} unlocked successfully")
            return True
        else:
            print(f"[WARNING] Node {node_config['id']} unlock failed, HTTP status: {response.status_code}")
            return False
    except httpx.RequestError as e:
        print(f"[WARNING] Failed to send unlock request to node {node_config['id']}: {e}")
        return False


def detect_upstream_provider_by_url(url: str) -> str:
    """Detect upstream provider type by URL path.
    Returns one of: 'openai_chat', 'ollama_chat', 'ollama_generate', 'unknown'.
    """
    try:
        path = url.split('://', 1)[-1]
        path = path[path.find('/'):] if '/' in path else ''
    except Exception:
        path = ''

    if "/v1/chat/completions" in path:
        return "openai_chat"
    if "/api/chat/completions" in path:
        return "ollama_chat"
    if "/api/generate" in path:
        return "ollama_generate"
    return "unknown"


def messages_to_prompt(messages: List[dict]) -> str:
    """Convert OpenAI-style messages to a plain prompt string for Ollama /api/generate fallback."""
    parts = []
    for msg in messages or []:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Simple readable formatting
        if role:
            parts.append(f"{role.capitalize()}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


@app.post("/api/chat/completions")
async def chat_proxy(request: ChatRequest):
    """
    Core proxy function for handling chat requests.
    它会根据负载和用户指定的模型来选择最佳节点，并以流式响应返回结果。
    """
    # If the user specifies a specific model, first try to unlock all nodes to release resources
    if request.model:
        print(f"[INFO] User specified model {request.model}, first try to unlock all nodes to release resources")
        try:
            unlock_tasks = [unlock_node(node) for node in settings.NODES]
            await asyncio.gather(*unlock_tasks, return_exceptions=True)
            # Wait for a short time to update the node status
            await asyncio.sleep(0.5)
            
            # Force refresh the node status
            refresh_tasks = [fetch_single_node_status(node) for node in settings.NODES]
            await asyncio.gather(*refresh_tasks, return_exceptions=True)
        except Exception as e:
            print(f"[WARNING] Error unlocking nodes: {e}")
    
    candidate_configs = []
    with CACHE_LOCK:
        for node_id, status in NODE_STATUS_CACHE.items():
            metrics = status.get("metrics") or {}
            # Filter conditions: node online, not locked, and (if user specified model) model ID matches
            if status.get("online") and not metrics.get("locked", True):
                model_matches = (not request.model) or (metrics.get("model_id") == request.model)
                if model_matches:
                    config = next((n for n in settings.NODES if n["id"] == node_id), None)
                    if config:
                        candidate_configs.append((config, metrics))

    # If no matching node is found, but the user specified a model, try to start the model
    if not candidate_configs and request.model:
        print(f"[INFO] No node found running model {request.model}, try to start the model")
        try:
            # Find the first available node
            available_node = None
            with CACHE_LOCK:
                for node_id, status in NODE_STATUS_CACHE.items():
                    if status["online"]:
                        config = next((n for n in settings.NODES if n["id"] == node_id), None)
                        if config:
                            available_node = config
                            break
            
            if available_node:
                # Try to start the model
                ollama_url = available_node["llm_url"].replace("/api/chat", "/api/generate")
                payload = {
                    "model": request.model,
                    "prompt": "test",
                    "stream": False
                }
                
                response = await client.post(ollama_url, json=payload, timeout=30.0)
                if response.status_code == 200:
                    print(f"[INFO] Successfully started model {request.model} on node {available_node['id']}")
                    # 启动成功后刷新节点状态
                    try:
                        await fetch_single_node_status(available_node)
                        print(f"[INFO] Node {available_node['id']} status refreshed")
                    except Exception as e:
                        print(f"[WARNING] Failed to refresh node status: {e}")
                    
                    # Re-get candidate nodes
                    with CACHE_LOCK:
                        entry = NODE_STATUS_CACHE.get(available_node["id"], {})
                        m = (entry or {}).get("metrics", {})
                        if entry.get("online") and not m.get("locked", True):
                            candidate_configs.append((available_node, m))
                else:
                    print(f"[WARNING] Failed to start model {request.model}: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Error starting model: {e}")
    
    # If still no matching node is found, try to use any available node (no model restriction)
    if not candidate_configs:
        print(f"[INFO] No matching node found, try to use any available node")
        with CACHE_LOCK:
            for node_id, status in NODE_STATUS_CACHE.items():
                metrics = status.get("metrics") or {}
                if status.get("online") and not metrics.get("locked", True):
                    config = next((n for n in settings.NODES if n["id"] == node_id), None)
                    if config:
                        candidate_configs.append((config, metrics))
                        break  # 只取第一个可用节点

    # 按GPU利用率排序，其次是CPU利用率 (升序)
    candidate_configs.sort(key=lambda x: (
        (x[1] or {}).get("gpu", {}).get("utilization_percent", float('inf')),
        (x[1] or {}).get("cpu_usage_percent", float('inf'))
    ))
    
    # Try to lock candidate nodes one by one, until success
    selected_node_config = None
    selected_node_metrics = None
    for config, _ in candidate_configs:
        if await lock_node(config):
            selected_node_config = config
            # Save corresponding metrics, for later supplementing model field
            with CACHE_LOCK:
                cache_entry = NODE_STATUS_CACHE.get(config["id"], {})
                selected_node_metrics = (cache_entry or {}).get("metrics", {})
            break
    
    if not selected_node_config:
        raise HTTPException(status_code=503, detail="All suitable nodes are busy or unavailable.")

    async def stream_generator():
        """
        用于流式传输后端LLM响应的生成器。
        它会首先发送一个自定义的'node_assigned'事件，
        然后在代理LLM响应时主动寻找'[DONE]'信号以提前解锁节点。
        """
        unlocked = False
        
        # 首先发送一个自定义事件，告知前端哪个节点被选中
        try:
            node_name = selected_node_config["name"]
            # 为兼容前端解析，仅通过 data 行传递事件名称和内容
            payload = json.dumps({"event": "node_assigned", "node_name": node_name})
            yield f"data: {payload}\n\n".encode('utf-8')
        except Exception as e:
            print(f"Unable to send node_assigned event: {e}")

        # Enhanced streaming processing: support Ollama /api/chat, Ollama /api/generate (fallback), and OpenAI /v1/chat/completions
        try:
            upstream_url = selected_node_config["llm_url"]
            provider = detect_upstream_provider_by_url(upstream_url)
            model_name = (request.model or (selected_node_metrics or {}).get("model_id"))

            async def open_stream(url: str, provider_key: str):
                if provider_key == "ollama_chat":
                    payload = {
                        "model": model_name,
                        "messages": request.messages,
                        "stream": True if request.stream else False,
                    }
                elif provider_key == "ollama_generate":
                    payload = {
                        "model": model_name,
                        "prompt": messages_to_prompt(request.messages),
                        "stream": True if request.stream else False,
                    }
                else:  # openai_chat or unknown -> use OpenAI-style chat payload
                    payload = {
                        "model": model_name or "",
                        "messages": request.messages,
                        "stream": True if request.stream else False,
                    }
                return client.stream("POST", url, json=payload, timeout=300.0)

            # 首次尝试
            async with await open_stream(upstream_url, provider) as response:
                # 常见错误回退（404/405/501）
                if response.status_code in (404, 405, 501) and provider == "ollama_chat":
                    fallback_url = upstream_url.replace("/api/chat", "/api/generate")
                    provider = "ollama_generate"
                    await response.aclose()
                    async with await open_stream(fallback_url, provider) as response2:
                        buffer = b""
                        async for chunk in response2.aiter_bytes():
                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line.decode("utf-8"))
                                    done = bool(obj.get("done"))
                                    if done and not unlocked:
                                        yield b"data: [DONE]\n\n"
                                        unlock_success = await unlock_node(selected_node_config)
                                        if unlock_success:
                                            unlocked = True
                                        else:
                                            print(f"[WARNING] Node {selected_node_config['id']} unlock failed, will retry later")
                                        break
                                    delta_content = obj.get("response") or ""
                                    if delta_content:
                                        sse_payload = {"choices": [{"delta": {"content": delta_content}}]}
                                        yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                                except Exception:
                                    continue
                        return

                # 正常处理不同 provider 的流
                buffer = b""
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            if provider == "openai_chat":
                                if not line.startswith(b"data:"):
                                    continue
                                data_part = line[len(b"data:"):].strip()
                                if data_part == b"[DONE]":
                                    if not unlocked:
                                        yield b"data: [DONE]\n\n"
                                        unlock_success = await unlock_node(selected_node_config)
                                        if unlock_success:
                                            unlocked = True
                                        else:
                                            print(f"[WARNING] Node {selected_node_config['id']} unlock failed, will retry later")
                                    break
                                obj = json.loads(data_part.decode("utf-8"))
                                for choice in obj.get("choices") or []:
                                    delta = choice.get("delta") or {}
                                    delta_content = delta.get("content") or ""
                                    if delta_content:
                                        sse_payload = {"choices": [{"delta": {"content": delta_content}}]}
                                        yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                            elif provider == "ollama_generate":
                                obj = json.loads(line.decode("utf-8"))
                                done = bool(obj.get("done"))
                                if done and not unlocked:
                                    yield b"data: [DONE]\n\n"
                                    unlock_success = await unlock_node(selected_node_config)
                                    if unlock_success:
                                        unlocked = True
                                    else:
                                        print(f"[WARNING] Node {selected_node_config['id']} unlock failed, will retry later")
                                    break
                                delta_content = obj.get("response") or ""
                                if delta_content:
                                    sse_payload = {"choices": [{"delta": {"content": delta_content}}]}
                                    yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                            else:
                                obj = json.loads(line.decode('utf-8'))
                                done = bool(obj.get("done"))
                                if done and not unlocked:
                                    yield b"data: [DONE]\n\n"
                                    unlock_success = await unlock_node(selected_node_config)
                                    if unlock_success:
                                        unlocked = True
                                    else:
                                        print(f"[WARNING] Node {selected_node_config['id']} unlock failed, will retry later")
                                    break
                                msg = obj.get("message") or {}
                                delta_content = msg.get("content") or ""
                                if delta_content:
                                    sse_payload = {"choices": [{"delta": {"content": delta_content}}]}
                                    yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n".encode('utf-8')
                        except Exception:
                            continue
        except httpx.RequestError as e:
            # 网络错误：发出SSE错误并尝试解锁
            err = {"error": f"Upstream connection failed: {type(e).__name__}: {str(e)}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode('utf-8')
        except Exception as e:
            err = {"error": f"Internal proxy error: {str(e)}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode('utf-8')
        finally:
            # 作为一个健壮的后备方案，如果流结束时仍未解锁，则在这里解锁。
            if not unlocked:
                print(f"Warning: [DONE] signal not received to unlock. Execute fallback unlock in finally block, node {selected_node_config['id']}.")
                unlock_success = await unlock_node(selected_node_config)
                if not unlock_success:
                    print(f"[WARNING] Node {selected_node_config['id']} fallback unlock also failed, may need manual unlock")
            
            # 无论如何，流结束后刷新该节点状态以更新当前运行模型等信息
            try:
                await fetch_single_node_status(selected_node_config)
                print(f"[INFO] Session ended, node {selected_node_config['id']} status refreshed")
            except Exception as e:
                print(f"[WARNING] Session ended refresh node status failed: {e}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/api/unlock/all")
async def unlock_all_nodes():
    """
    Force send unlock command to all configured nodes.
    这是一个应急接口，用于解决节点被意外锁定的问题。
    """
    print("Received request to force unlock all nodes...")
    
    try:
        tasks = [unlock_node(node) for node in settings.NODES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        report = {}
        success_count = 0
        for i, result in enumerate(results):
            node_id = settings.NODES[i]['id']
            if isinstance(result, Exception):
                status = "Failed"
                print(f"Force unlock node {node_id} failed: {result}")
            else:
                status = "Success"
                success_count += 1
            report[node_id] = status

        # After unlocking, trigger status refresh (safe handling of exceptions)
        try:
            refresh_tasks = [fetch_single_node_status(node) for node in settings.NODES]
            asyncio.create_task(asyncio.gather(*refresh_tasks, return_exceptions=True))
        except Exception as e:
            print(f"Exception occurred during status refresh (不影响解锁结果): {e}")
        
        message = f"Sent unlock command to all nodes, successfully unlocked {success_count}/{len(settings.NODES)} nodes"
        return {"status": "success", "detail": message, "report": report}
        
    except Exception as e:
        print(f"Exception occurred when unlocking all nodes: {e}")
        # 仍然尝试解锁每个节点
        report = {}
        for node in settings.NODES:
            try:
                await unlock_node(node)
                report[node['id']] = "Success"
            except Exception as ex:
                report[node['id']] = f"Failed: {ex}"
        
        return {"status": "partial", "detail": "Partial unlock completed", "report": report}

@app.post("/api/dataset/cleanup/all")
async def cleanup_all_jobs():
    """Force clean up all dataset processing tasks. Emergency function, used to clean up stuck tasks."""
    job_ids_to_stop = []
    
    with JOBS_LOCK:
        for job_id, job in DATASET_JOBS.items():
            if job["status"] in ["processing", "pending", "queued"]:
                job_ids_to_stop.append(job_id)
    
    if not job_ids_to_stop:
        return {"status": "success", "message": "No tasks need to be cleaned up"}
    
    print(f"Force clean up tasks: {job_ids_to_stop}")
    
    # 停止所有正在进行的任务
    cleanup_tasks = [stop_job_processing(job_id) for job_id in job_ids_to_stop]
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
        # Force unlock all nodes
    try:
        unlock_result = await unlock_all_nodes()
        unlock_msg = unlock_result.get("detail", "Unlock completed")
    except Exception as e:
        print(f"Failed to unlock nodes during cleanup: {e}")
        unlock_msg = "Unlock partially failed"
    
    return {
        "status": "success", 
        "message": f"Force cleaned up {len(job_ids_to_stop)} tasks, {unlock_msg}"
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get system overall status, including node status and task status"""
    # 获取节点状态
    with CACHE_LOCK:
        nodes_status = list(NODE_STATUS_CACHE.values())
    
    # Get task status
    with JOBS_LOCK:
        jobs_status = {}
        for job_id, job in DATASET_JOBS.items():
            jobs_status[job_id] = {
                "status": job["status"],
                "total_items": job["total_items"],
                "processed_items": job["processed_items"],
                "start_time": job["start_time"],
                "end_time": job.get("end_time"),
                "active_tasks": len(job.get("processing_tasks", [])),
                "locked_nodes": len(job.get("locked_nodes", []))
            }
    
    # Statistics - safe handling of metrics
    online_nodes = sum(1 for node in nodes_status if node["online"])
    busy_nodes = 0
    for node in nodes_status:
        if node["online"]:
            metrics = node.get("metrics") or {}
            if metrics.get("locked", False):
                busy_nodes += 1
    processing_jobs = sum(1 for job in jobs_status.values() if job["status"] == "processing")
    
    return {
        "nodes": {
            "total": len(nodes_status),
            "online": online_nodes,
            "busy": busy_nodes,
            "details": nodes_status
        },
        "jobs": {
            "total": len(jobs_status),
            "processing": processing_jobs,
            "details": jobs_status
        },
        "timestamp": time.time()
    }


# --- 数据集处理相关API ---

# --- 核心改造：数据集处理端点 ---
# 这是新的异步任务上传入口
@app.post("/api/dataset/upload")
async def process_dataset_endpoint(file: UploadFile = File(...), data_count: str = Form(None)):
    """
    Receive dataset file, create a background processing task, and immediately return task ID.
    """
    job_id = str(uuid.uuid4())
    content = await file.read()
    try:
        dataset = json.loads(content)
        if not isinstance(dataset, list):
            raise ValueError("Dataset must be in JSON array format.")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"File format error: {e}")

    # 添加调试信息
    print(f"[DEBUG] Task {job_id}: Received data_count parameter: {repr(data_count)}")
    print(f"[DEBUG] Task {job_id}: data_count type: {type(data_count)}")
    print(f"[DEBUG] Task {job_id}: Original dataset size: {len(dataset)}")

    # 如果用户通过 data_count 指定了处理数量，则截取数据集
    if data_count and str(data_count).strip().isdigit():
        count = int(str(data_count).strip())
        print(f"[INFO] Task {job_id}: User specified processing {count} items, original dataset size {len(dataset)}")
        if count > 0 and count <= len(dataset):
            dataset = dataset[:count]
            print(f"[DEBUG] Task {job_id}: After truncation, dataset size: {len(dataset)}")
        else:
            print(f"[WARNING] Task {job_id}: data_count out of range ({count}), using full dataset")
    else:
        print(f"[INFO] Task {job_id}: Using full dataset, total {len(dataset)} items")
        print(f"[DEBUG] Task {job_id}: data_count invalid - value: {repr(data_count)}, is digit: {str(data_count).strip().isdigit() if data_count else False}")

    total_items = len(dataset)
    
    if total_items == 0:
        raise HTTPException(status_code=400, detail="Dataset has no items to process.")

    # 初始化任务状态
    with JOBS_LOCK:
        DATASET_JOBS[job_id] = {
            "status": "queued",
            "total_items": total_items,
            "processed_items": 0,
            "start_time": time.time(),
            "end_time": None,
            "results": [],
            "original_full_dataset": json.loads(content),  # 保存完整原始数据集
            "selected_dataset": dataset,  # 保存用户选择的数据集
            "user_data_count": total_items,  # 用户实际选择的数据条数
            "node_reports": {node["id"]: {"processed_count": 0, "total_time": 0.0} for node in settings.NODES}
        }

    # 在后台启动真正的处理任务
    asyncio.create_task(run_dataset_processing(job_id, dataset))

    return {"job_id": job_id, "message": f"Task created, total {total_items} items."}


async def run_dataset_processing(job_id: str, dataset: list):
    """
    重构版：简化的数据处理函数
    1. 获取可用节点列表
    2. 平均分配数据给各节点
    3. 并行处理，每条数据处理完立即保存
    4. 处理异常和节点断线重分配
    """
    print(f"[INFO] Starting task {job_id}, dataset size: {len(dataset)}")
    
    with JOBS_LOCK:
        job_info = DATASET_JOBS[job_id]
        job_info["status"] = "processing"
        job_info["processing_tasks"] = []
        job_info["remaining_data"] = list(enumerate(dataset))  # 未处理的数据（索引，数据）
        job_info["results"] = [None] * len(dataset)  # 结果数组，按原始索引存储
        
    print(f"[INFO] Task {job_id}: Starting to distribute data to available nodes")
    


    # --- Simplified main processing logic ---
    try:
        # 步骤1: 获取可用节点
        available_nodes = []
        with CACHE_LOCK:
            for node_id, status in NODE_STATUS_CACHE.items():
                if status["online"]:
                    is_locked = status.get("metrics", {}).get("locked", False) if status.get("metrics") else False
                    if not is_locked:
                        config = next((n for n in settings.NODES if n["id"] == node_id), None)
                        if config:
                            available_nodes.append(config)
        
        if not available_nodes:
            print(f"[ERROR] Task {job_id}: No available nodes, task failed")
            with JOBS_LOCK:
                DATASET_JOBS[job_id]["status"] = "failed"
                DATASET_JOBS[job_id]["end_time"] = time.time()
            return
        
        print(f"[INFO] Task {job_id}: Found {len(available_nodes)} available nodes")
        
        # 步骤2: 平均分配数据
        indexed_data = list(enumerate(dataset))
        node_assignments = distribute_data_to_nodes(indexed_data, available_nodes)
        
        # 步骤3: 启动所有节点并行处理
        tasks = []
        for i, (node_config, assigned_data) in enumerate(node_assignments):
            print(f"[INFO] Distributing {len(assigned_data)} items to node {node_config['id']}")
            task = asyncio.create_task(process_node_data_with_job_id(job_id, node_config, assigned_data))
            tasks.append(task)
            
            with JOBS_LOCK:
                DATASET_JOBS[job_id]["processing_tasks"].append(task)
        
        # 步骤4: 等待所有节点完成处理
        print(f"[INFO] Task {job_id}: Waiting for all nodes to complete processing...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 步骤5: 检查最终状态
        with JOBS_LOCK:
            job_info = DATASET_JOBS[job_id]
            if job_info["status"] == "processing":
                job_info["status"] = "completed"
                print(f"[INFO] Task {job_id} completed")
            else:
                print(f"[INFO] Task {job_id} ended with status {job_info['status']}")
            job_info["end_time"] = time.time()
            
    except asyncio.CancelledError:
        print(f"[INFO] Task {job_id} cancelled")
        with JOBS_LOCK:
            DATASET_JOBS[job_id]["status"] = "cancelled"
            DATASET_JOBS[job_id]["end_time"] = time.time()
    except Exception as e:
        print(f"[ERROR] Task {job_id} processing exception: {e}")
        with JOBS_LOCK:
            DATASET_JOBS[job_id]["status"] = "failed"
            DATASET_JOBS[job_id]["end_time"] = time.time()
    finally:
        # 关键修复：确保任务结束后所有节点都被解锁
        await ensure_all_nodes_unlocked(job_id)


async def ensure_all_nodes_unlocked(job_id: str, max_retries: int = 3):
    """
    确保任务结束后所有节点都被正确解锁
    This is a critical cleanup function to prevent nodes from remaining locked
    """
    print(f"[INFO] Task {job_id}: Starting to ensure all nodes are unlocked")
    
    # 获取所有在线节点，不管是否锁定
    online_nodes = []
    with CACHE_LOCK:
        for node_id, status in NODE_STATUS_CACHE.items():
            if status["online"]:
                config = next((n for n in settings.NODES if n["id"] == node_id), None)
                if config:
                    is_locked = status.get("metrics", {}).get("locked", False) if status.get("metrics") else False
                    online_nodes.append((config, is_locked))
    
    if not online_nodes:
        print(f"[INFO] Task {job_id}: No online nodes need to be unlocked")
        return
    
    # 对所有在线节点尝试解锁（包括已经解锁的，确保状态一致）
    unlock_results = {}
    
    for retry_attempt in range(max_retries):
        nodes_to_unlock = []
        
        # 重新检查哪些节点需要解锁
        with CACHE_LOCK:
            for node_config, was_locked in online_nodes:
                node_id = node_config["id"]
                current_status = NODE_STATUS_CACHE.get(node_id, {})
                current_metrics = current_status.get("metrics", {})
                is_still_locked = current_metrics.get("locked", False) if current_metrics else False
                
                # 如果节点仍然锁定，或者这是第一次尝试，都要解锁
                if is_still_locked or retry_attempt == 0:
                    nodes_to_unlock.append(node_config)
        
        if not nodes_to_unlock:
            print(f"[INFO] Task {job_id}: All nodes are unlocked")
            break
        
        print(f"[INFO] Task {job_id}: Attempting to unlock {len(nodes_to_unlock)} nodes (第 {retry_attempt + 1}/{max_retries} 次)")
        
        # 并行解锁所有需要解锁的节点
        unlock_tasks = [unlock_node(node_config) for node_config in nodes_to_unlock]
        results = await asyncio.gather(*unlock_tasks, return_exceptions=True)
        
        # 记录结果
        for i, result in enumerate(results):
            node_config = nodes_to_unlock[i]
            node_id = node_config["id"]
            
            if isinstance(result, Exception):
                unlock_results[node_id] = f"Exception: {result}"
                print(f"[ERROR] Node {node_id} unlock exception: {result}")
            elif result:
                unlock_results[node_id] = "Success"
                print(f"[INFO] Node {node_id} unlock successfully")
            else:
                unlock_results[node_id] = "Failed"
                print(f"[WARNING] Node {node_id} unlock failed")
        
        # 等待一点时间让节点状态更新
        if retry_attempt < max_retries - 1:
            await asyncio.sleep(1)
    
    # 最终状态检查
    await asyncio.sleep(2)  # Wait for status update
    
    still_locked_nodes = []
    with CACHE_LOCK:
        for node_config, _ in online_nodes:
            node_id = node_config["id"]
            current_status = NODE_STATUS_CACHE.get(node_id, {})
            current_metrics = current_status.get("metrics", {})
            is_locked = current_metrics.get("locked", False) if current_metrics else False
            
            if is_locked:
                still_locked_nodes.append(node_id)
    
    if still_locked_nodes:
        print(f"[WARNING] Task {job_id}: The following nodes are still locked: {still_locked_nodes}")
        print(f"[INFO] You can use the /api/unlock/all interface to manually unlock all nodes")
    else:
        print(f"[INFO] Task {job_id}: All nodes are unlocked")
    
    return unlock_results


def distribute_data_to_nodes(indexed_data: list, available_nodes: list) -> list:
    """
    Average distribute data to available nodes
    返回: [(node_config, assigned_data), ...]
    """
    num_nodes = len(available_nodes)
    num_items = len(indexed_data)
    items_per_node = num_items // num_nodes
    remainder = num_items % num_nodes
    
    assignments = []
    start_idx = 0
    
    for i, node_config in enumerate(available_nodes):
        # The first few nodes get one extra item (to handle the remainder)
        node_items = items_per_node + (1 if i < remainder else 0)
        end_idx = start_idx + node_items
        
        assigned_data = indexed_data[start_idx:end_idx]
        assignments.append((node_config, assigned_data))
        
        start_idx = end_idx
    
    return assignments


async def handle_node_failure_and_redistribute(job_id: str, failed_node_id: int):
    """
    Handle node failure and redistribute unfinished data to other online nodes
    """
    print(f"[WARNING] Detected node {failed_node_id} failure, starting to redistribute data")
    
    with JOBS_LOCK:
        job_info = DATASET_JOBS.get(job_id)
        if not job_info or job_info["status"] != "processing":
            return
        
        # 找出未完成的数据
        unfinished_data = []
        selected_dataset = job_info.get("selected_dataset", [])
        for i, result in enumerate(job_info["results"]):
            if result is None and i < len(selected_dataset):
                # 获取用户选择的数据
                original_data = selected_dataset[i]
                unfinished_data.append((i, original_data))
        
        if not unfinished_data:
            print(f"[INFO] Node {failed_node_id} failure, but no unfinished data")
            return
    
    # Get current online nodes
    available_nodes = []
    with CACHE_LOCK:
        for node_id, status in NODE_STATUS_CACHE.items():
            if status["online"] and node_id != failed_node_id:
                is_locked = status.get("metrics", {}).get("locked", False) if status.get("metrics") else False
                if not is_locked:
                    config = next((n for n in settings.NODES if n["id"] == node_id), None)
                    if config:
                        available_nodes.append(config)
    
    if not available_nodes:
        print(f"[ERROR] Node {failed_node_id} failure, but no other available nodes, task failed")
        with JOBS_LOCK:
            DATASET_JOBS[job_id]["status"] = "failed"
            DATASET_JOBS[job_id]["end_time"] = time.time()
        return
    
    print(f"[INFO] Redistributing {len(unfinished_data)} unfinished data to {len(available_nodes)} nodes")
    
    # Redistribute data
    node_assignments = distribute_data_to_nodes(unfinished_data, available_nodes)
    
    # Start new processing tasks
    new_tasks = []
    for node_config, assigned_data in node_assignments:
        print(f"[INFO] Redistributing {len(assigned_data)} items to node {node_config['id']}")
        task = asyncio.create_task(process_node_data_with_job_id(job_id, node_config, assigned_data))
        new_tasks.append(task)
        
        with JOBS_LOCK:
            DATASET_JOBS[job_id]["processing_tasks"].extend(new_tasks)
    
    return new_tasks


async def process_node_data_with_job_id(job_id: str, node_config: dict, assigned_data: list):
    """
    Wrapped node processing function, used when redistributing
    """
    # 这是原来process_node_data函数的逻辑，但需要job_id参数
    node_id = node_config["id"]
    print(f"[INFO] Node {node_id} starting to reprocess {len(assigned_data)} items")
    
    # 锁定节点
    if not await lock_node(node_config):
        print(f"[ERROR] Node {node_id} lock failed")
        return
    
    node_start_time = time.time()
    actual_processing_time = 0.0
    
    try:
        # 获取当前节点的 metrics 以便选择模型
        with CACHE_LOCK:
            cache_entry = NODE_STATUS_CACHE.get(node_id, {})
            selected_node_metrics = (cache_entry or {}).get("metrics", {})

        upstream_url = node_config["llm_url"]
        provider = detect_upstream_provider_by_url(upstream_url)

        async def nonstream_call(url: str, provider_key: str, model_name: str, messages: list):
            if provider_key == "ollama_chat":
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                }
                return await client.post(url, json=payload, timeout=120.0)
            elif provider_key == "ollama_generate":
                payload = {
                    "model": model_name,
                    "prompt": messages_to_prompt(messages),
                    "stream": False,
                }
                return await client.post(url, json=payload, timeout=120.0)
            else:  # openai_chat or unknown
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                }
                return await client.post(url, json=payload, timeout=120.0)

        for original_index, item_data in assigned_data:
            # 检查任务是否被停止（支持优雅停止）
            with JOBS_LOCK:
                current_status = DATASET_JOBS[job_id]["status"]
                if current_status == "stopping":
                    print(f"[INFO] Received stop signal, node {node_id} will exit gracefully after completing the current item")
                    break
                elif current_status != "processing":
                    print(f"[INFO] Task status changed to {current_status}, node {node_id} will exit")
                    break
            
            # 构造prompt
            instruction = item_data.get("instruction", "")
            input_text = item_data.get("input", "")
            prompt = f"Instruction: {instruction}\nInput: {input_text}"
            
            # 选择模型名：优先节点当前模型，其次兜底
            model_name = selected_node_metrics.get("model_id") or "llama3:8b"
            
            # 处理单条数据
            item_start_time = time.time()
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await nonstream_call(upstream_url, provider, model_name, messages)
                if response.status_code in (404, 405, 501) and provider == "ollama_chat":
                    # 回退到 generate
                    provider = "ollama_generate"
                    upstream_url = upstream_url.replace("/api/chat", "/api/generate")
                    response = await nonstream_call(upstream_url, provider, model_name, messages)
                response.raise_for_status()
                result = response.json()
                
                item_end_time = time.time()
                actual_processing_time += (item_end_time - item_start_time)
                
                print(f"[DEBUG] Node {node_id} reprocessed item {original_index}")
                
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                item_end_time = time.time()
                actual_processing_time += (item_end_time - item_start_time)
                
                # 失败时记录错误信息
                result = {"error": f"Processing failed: {str(e)}"}
                print(f"[ERROR] Node {node_id} reprocessed item {original_index} failed: {e}")
            
            # Immediately save the result
            with JOBS_LOCK:
                if job_id in DATASET_JOBS:
                    job_info = DATASET_JOBS[job_id]
                    job_info["results"][original_index] = result
                    
                    # Update node report
                    report = job_info["node_reports"][node_id]
                    report["processed_count"] += 1
                    report["total_time"] = actual_processing_time
                    
                    # 更新全局进度
                    selected_dataset = job_info.get("selected_dataset", [])
                    completed_count = sum(1 for r in job_info["results"] if r is not None)
                    job_info["processed_items"] = completed_count
                    
                    print(f"[DEBUG] Node {node_id} saved processing results, global progress: {completed_count}/{len(selected_dataset)}")
            
    finally:
        # 解锁节点（带重试机制）
        unlock_success = await unlock_node(node_config)
        if not unlock_success:
            print(f"[WARNING] Node {node_id} initial unlock failed, will retry when task ends")
        
        node_end_time = time.time()
        print(f"[INFO] Node {node_id} reprocessed completed, actual time: {actual_processing_time:.2f}s, total time: {node_end_time - node_start_time:.2f}s")


@app.get("/api/dataset/status/{job_id}", response_model=JobStatus)
async def get_dataset_status(job_id: str):
    """Get the current status and progress of the task by task ID."""
    with JOBS_LOCK:
        job = DATASET_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Task ID not found")
        
        # To conform to the Pydantic model, data needs to be extracted from the job dictionary
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            total_items=job["total_items"],
            processed_items=job["processed_items"],
            start_time=job["start_time"],
            end_time=job["end_time"],
            node_reports=job["node_reports"]
        )

@app.get("/api/dataset/detailed-status/{job_id}")
async def get_detailed_dataset_status(job_id: str):
    """Get detailed status information of the task, including error statistics, node status, etc."""
    with JOBS_LOCK:
        job = DATASET_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Task ID not found")
        
        # 分析结果统计
        results_array = job.get("results", [])
        total_items = job["total_items"]
        processed_items = job["processed_items"]
        
        # Count different types of results
        successful_count = 0
        error_count = 0
        not_processed_count = 0
        error_types = {}
        
        for i, result in enumerate(results_array):
            if result is None:
                not_processed_count += 1
            elif isinstance(result, dict) and "error" in result:
                error_count += 1
                error_msg = result["error"]
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
            else:
                successful_count += 1
        
        # Calculate processing rate
        completion_rate = (processed_items / total_items * 100) if total_items > 0 else 0
        success_rate = (successful_count / total_items * 100) if total_items > 0 else 0
        error_rate = (error_count / total_items * 100) if total_items > 0 else 0
        
        # Node status analysis
        node_status_summary = {}
        node_reports = job.get("node_reports", {})
        
        # Check current online nodes
        with CACHE_LOCK:
            for node_id, report in node_reports.items():
                node_cache = NODE_STATUS_CACHE.get(int(node_id), {})
                node_status_summary[node_id] = {
                    "processed_count": report.get("processed_count", 0),
                    "total_time": report.get("total_time", 0.0),
                    "is_online": node_cache.get("online", False),
                    "is_locked": node_cache.get("metrics", {}).get("locked", False) if node_cache.get("metrics") else False,
                    "node_name": next((n["name"] for n in settings.NODES if n["id"] == int(node_id)), f"Node {node_id}")
                }
        
        # 运行时长
        start_time = job["start_time"]
        end_time = job.get("end_time")
        runtime = (end_time or time.time()) - start_time
        
        # 状态描述
        status_description = {
            "queued": "Task created, waiting to start processing",
            "processing": "Processing",
            "stopping": "Gracefully stopping, waiting for nodes to complete the current item",
            "stopped": "Task stopped",
            "completed": "Task completed",
            "failed": "Task processing failed",
            "cancelled": "Task cancelled"
        }.get(job["status"], "Unknown status")
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "status_description": status_description,
            "progress": {
                "total_items": total_items,
                "processed_items": processed_items,
                "successful_count": successful_count,
                "error_count": error_count,
                "not_processed_count": not_processed_count,
                "completion_rate": round(completion_rate, 1),
                "success_rate": round(success_rate, 1),
                "error_rate": round(error_rate, 1)
            },
            "errors": {
                "error_types": error_types,
                "total_errors": error_count
            },
            "nodes": node_status_summary,
            "timing": {
                "start_time": start_time,
                "end_time": end_time,
                "runtime_seconds": round(runtime, 2),
                "runtime_formatted": f"{int(runtime//60)}分{int(runtime%60)}秒" if runtime >= 60 else f"{runtime:.1f}秒"
            },
            "is_terminal": job["status"] in ["completed", "stopped", "failed", "cancelled"]
        }

@app.get("/api/dataset/result/{job_id}")
async def get_dataset_result(job_id: str):
    """获取任务处理结果，支持已完成和已停止的任务。"""
    with JOBS_LOCK:
        job = DATASET_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Task ID not found")
        
        # --- 修复: 允许下载stopped任务的部分结果 ---
        valid_statuses = ["completed", "stopped"]
        if job["status"] not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Task status does not allow downloading results, current status: {job['status']}")
        
        # 重构版：直接使用结果数组，合并用户选择的数据
        final_results = []
        selected_dataset = job.get("selected_dataset", [])  # 使用用户选择的数据集
        results_array = job.get("results", [])

        for i, original_item in enumerate(selected_dataset):
            final_item = original_item.copy()
            
            # 检查是否有对应的处理结果
            if i < len(results_array) and results_array[i] is not None:
                final_item['model_output'] = results_array[i]
            else:
                # 未处理或处理失败的条目
                if job["status"] == "stopped":
                    final_item['model_output'] = {"error": "Processing stopped"}
                else:
                    final_item['model_output'] = {"error": "Not processed"}
            
            final_results.append(final_item)
            
    return JSONResponse(content={"job_id": job_id, "results": final_results, "status": job["status"]})

async def stop_job_processing(job_id: str):
    """
    Refactored version: Gracefully stop task
    1. Set status to "stopping"
    2. Wait for nodes to complete the current item and exit naturally
    3. Set final status to "stopped"
    """
    with JOBS_LOCK:
        job = DATASET_JOBS.get(job_id)
        if not job:
            print(f"[ERROR] Attempt to stop task {job_id}, but task does not exist")
            return False
        
        if job["status"] not in ["processing"]:
            print(f"[INFO] Task {job_id} status is {job['status']}, no need to stop")
            return False
        
        print(f"[INFO] Starting to gracefully stop task {job_id}")
        
        # Set to stopping status, so nodes can exit gracefully
        job["status"] = "stopping"
        
        # Get running tasks
        running_tasks = job.get("processing_tasks", [])
    
    # Wait for all nodes to complete the current item and exit
    if running_tasks:
        print(f"[INFO] Task {job_id}: Waiting for {len(running_tasks)} nodes to complete the current item...")
        try:
            # 等待所有任务自然完成（不取消）
            await asyncio.gather(*running_tasks, return_exceptions=True)
            print(f"[INFO] Task {job_id}: All nodes have exited gracefully")
        except Exception as e:
            print(f"[WARNING] Task {job_id}: Exception occurred while waiting for nodes to exit: {e}")
    
    # Set final status
    with JOBS_LOCK:
        if job_id in DATASET_JOBS:
            job_info = DATASET_JOBS[job_id]
            job_info["status"] = "stopped"
            job_info["end_time"] = time.time()
            print(f"[INFO] Task {job_id} has been gracefully stopped")
    
    return True

@app.post("/api/dataset/stop/{job_id}")
async def stop_dataset_processing_endpoint(job_id: str):
    """API endpoint, used to manually stop a task."""
    try:
        await stop_job_processing(job_id)
        return {"status": "success", "message": f"Task {job_id} has been marked as stopped."}
    except Exception as e:
        print(f"Error stopping task {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")

@app.get("/api/debug/info")
async def get_debug_info():
    """Debug endpoint, providing server status information."""
    with JOBS_LOCK:
        job_summary = {
            job_id: {"status": job["status"], "total_items": job["total_items"], "processed_items": job["processed_items"]}
            for job_id, job in DATASET_JOBS.items()
        }
    
    with CACHE_LOCK:
        node_summary = dict(NODE_STATUS_CACHE)
    
    return {
        "server_time": time.time(),
        "active_jobs": len(DATASET_JOBS),
        "jobs": job_summary,
        "nodes": node_summary,
        "cors_enabled": True
    }

@app.get("/api/debug/diagnosis")
async def diagnostic_info():
    """Diagnostic API: Check node status, monitor status, task status, etc."""
    import sys
    import traceback
    
    diagnosis = {
        "timestamp": time.time(),
        "gateway_status": "running",
        "nodes_config": settings.NODES,
        "node_status_cache": {},
        "active_jobs": {},
        "monitor_test": {}
    }
    
    # Check node status cache
    with CACHE_LOCK:
        diagnosis["node_status_cache"] = dict(NODE_STATUS_CACHE)
    
    # Check active tasks
    with JOBS_LOCK:
        for job_id, job in DATASET_JOBS.items():
            diagnosis["active_jobs"][job_id] = {
                "status": job["status"],
                "total_items": job["total_items"],
                "processed_items": job["processed_items"],
                "start_time": job["start_time"],
                "has_processing_tasks": len(job.get("processing_tasks", [])),
                "node_reports": job["node_reports"]
            }
    
    # Test node connectivity
    for node in settings.NODES:
        node_id = node["id"]
        test_result = {
            "node_id": node_id,
            "monitor_url": node["monitor_base_url"],
            "llm_url": node["llm_url"],
            "connection_test": "testing..."
        }
        
        try:
                # Test monitor connection
            response = await client.get(f"{node['monitor_base_url']}/status", timeout=5.0)
            if response.status_code == 200:
                metrics = response.json()
                test_result["connection_test"] = "success"
                test_result["monitor_response"] = metrics
            else:
                test_result["connection_test"] = f"monitor_error_{response.status_code}"
        except Exception as e:
            test_result["connection_test"] = f"monitor_failed: {str(e)}"
        
        diagnosis["monitor_test"][node_id] = test_result
    
    return diagnosis

# --- 静态前端文件服务 ---
# 必须在所有API路由定义之后
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

# --- 新增: Prometheus查询代理 ---
PROMETHEUS_URL = "http://localhost:9090"

@app.get("/api/prometheus/query_range")
async def prometheus_query_proxy(query: str, start: str, end: str, step: str):
    """
    A safe Prometheus range query proxy.
    Forward the frontend's request to Prometheus and return the original response directly.
    """
    params = {
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            
            # 直接返回Prometheus的原始JSON数据
            return response.json()

    except httpx.HTTPStatusError as e:
        error_detail = f"Prometheus query failed: {e.response.text}"
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        error_detail = f"Internal error occurred while forwarding request: {str(e)}"
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
# --- 结束新增 ---

# --- 新增: 告警检查逻辑 ---
def check_for_alerts():
    """
    检查所有节点的状态，并根据预设规则生成或清除告警。
    """
    with CACHE_LOCK, ALERTS_LOCK:
        global ALERTS_LIST
        active_alerts = []
        
        # 预设的告警规则
        ALERT_RULES = {
            "gpu_temp_severe": {"threshold": 70, "level": "severe", "message": "GPU temperature is too high"},
            "gpu_temp_warning": {"threshold": 50, "level": "warning", "message": "GPU temperature is high"},
            "mem_usage_severe": {"threshold": 50, "level": "severe", "message": "Memory usage is too high"},
            "mem_usage_warning": {"threshold": 30, "level": "warning", "message": "Memory usage is high"},
            "gpu_util_severe": {"threshold": 80, "level": "severe", "message": "GPU load is too high"},
            "gpu_util_warning": {"threshold": 60, "level": "warning", "message": "GPU load is high"},
        }
        
        for node_id, status in NODE_STATUS_CACHE.items():
            if not status["online"] or not status["metrics"]:
                continue

            node_name = next((n["name"] for n in settings.NODES if n["id"] == node_id), f"Host {node_id}")

            # 1. GPU 温度告警
            gpu_temp = status["metrics"].get("gpu", {}).get("temperature_celsius")
            if gpu_temp is not None:
                if gpu_temp >= ALERT_RULES["gpu_temp_severe"]["threshold"]:
                    active_alerts.append({
                        "id": f"gpu_temp_severe_{node_id}",
                        "level": ALERT_RULES["gpu_temp_severe"]["level"],
                        "message": ALERT_RULES["gpu_temp_severe"]["message"],
                        "details": f"GPU temperature reached {gpu_temp}°C",
                        "host": node_name,
                        "timestamp": time.time()
                    })
                elif gpu_temp >= ALERT_RULES["gpu_temp_warning"]["threshold"]:
                     active_alerts.append({
                        "id": f"gpu_temp_warning_{node_id}",
                        "level": ALERT_RULES["gpu_temp_warning"]["level"],
                        "message": ALERT_RULES["gpu_temp_warning"]["message"],
                        "details": f"GPU temperature reached {gpu_temp}°C",
                        "host": node_name,
                        "timestamp": time.time()
                    })

            # 2. 内存使用率告警
            mem_percent = status["metrics"].get("memory", {}).get("percent")
            if mem_percent is not None:
                if mem_percent >= ALERT_RULES["mem_usage_severe"]["threshold"]:
                    active_alerts.append({
                        "id": f"mem_usage_severe_{node_id}",
                        "level": ALERT_RULES["mem_usage_severe"]["level"],
                        "message": ALERT_RULES["mem_usage_severe"]["message"],
                        "details": f"Memory usage reached {mem_percent}%",
                        "host": node_name,
                        "timestamp": time.time()
                    })
                elif mem_percent >= ALERT_RULES["mem_usage_warning"]["threshold"]:
                    active_alerts.append({
                        "id": f"mem_usage_warning_{node_id}",
                        "level": ALERT_RULES["mem_usage_warning"]["level"],
                        "message": ALERT_RULES["mem_usage_warning"]["message"],
                        "details": f"Memory usage reached {mem_percent}%",
                        "host": node_name,
                        "timestamp": time.time()
                    })
            
            # 3. GPU 利用率告警
            gpu_util = status["metrics"].get("gpu", {}).get("utilization_percent")
            if gpu_util is not None:
                if gpu_util >= ALERT_RULES["gpu_util_severe"]["threshold"]:
                    active_alerts.append({
                        "id": f"gpu_util_severe_{node_id}",
                        "level": ALERT_RULES["gpu_util_severe"]["level"],
                        "message": ALERT_RULES["gpu_util_severe"]["message"],
                        "details": f"GPU load reached {gpu_util}%",
                        "host": node_name,
                        "timestamp": time.time()
                    })
                elif gpu_util >= ALERT_RULES["gpu_util_warning"]["threshold"]:
                    active_alerts.append({
                        "id": f"gpu_util_warning_{node_id}",
                        "level": ALERT_RULES["gpu_util_warning"]["level"],
                        "message": ALERT_RULES["gpu_util_warning"]["message"],
                        "details": f"GPU load reached {gpu_util}%",
                        "host": node_name,
                        "timestamp": time.time()
                    })

        # 更新告警列表，保留已有告警的首次触发时间
        existing_alerts = {alert["id"]: alert for alert in ALERTS_LIST}
        for alert in active_alerts:
            if alert["id"] in existing_alerts:
                alert["timestamp"] = existing_alerts[alert["id"]]["timestamp"] # 保留初次时间戳
        
        ALERTS_LIST = active_alerts

async def alert_checker_periodically():
    """Periodically check alerts."""
    while True:
        await asyncio.sleep(settings.MONITOR_INTERVAL * 2) # Check frequency can be slower than status update
        try:
            check_for_alerts()
        except Exception as e:
            print(f"[ERROR] Alert check failed: {e}")

# --- 结束新增 ---

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.GATEWAY_PORT
    ) 
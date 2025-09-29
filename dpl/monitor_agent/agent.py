import psutil
import uvicorn
import threading
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pynvml import *
from decouple import config
import platform # 新增
import subprocess
import re
import time
import os

app = FastAPI()

# --- Lock Status ---
# Global lock to prevent multiple requests from using the same node simultaneously
NODE_LOCKED = False
# Thread-safe lock for lock operations
lock = threading.Lock()
# --- End Lock Status Definition ---

# --- Sampling Cache (for rate calculation) ---
_NET_SNAPSHOT = {"ts": None, "pernic": {}}  # pernic: {name: {bytes_sent, bytes_recv}}
_DISK_SNAPSHOT = {"ts": None, "perdisk": {}}  # perdisk: {name: {read_bytes, write_bytes}}

# For simplicity, allow cross-origin requests from all sources. In production, you may need to restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if NVIDIA GPU is available
GPU_AVAILABLE = False
try:
    nvmlInit()
    GPU_AVAILABLE = True
except NVMLError as e:
    print(f"NVIDIA GPU not found or driver error: {e}")

"""Ollama Integration: Model ID Retrieval"""
# --- Model ID Retrieval (changed to Ollama) ---
def get_current_model_id():
    """Get current model ID, prioritizing OLLAMA_MODEL or selecting a suitable model from Ollama /api/tags."""
    # 1) Allow explicit specification
    override = os.getenv('OLLAMA_MODEL') or config('OLLAMA_MODEL', default=None)
    if override:
        return override

    # 2) Query Ollama local tags (available models)
    api_host = os.getenv('OLLAMA_HOST') or config('OLLAMA_HOST', default='http://127.0.0.1:11434')
    os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1,::1')
    os.environ.setdefault('no_proxy', 'localhost,127.0.0.1,::1')
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{api_host}/api/tags")
            resp.raise_for_status()
            data = resp.json() or {}
            models = data.get('models', []) or []
            # Prioritize matching llama3.1 or llama-3.1-8b series
            for m in models:
                mid = (m.get('name') or '').strip()
                if mid and ('llama-3.1-8b-instruct' in mid or 'llama3.1' in mid or 'llama3' in mid):
                    return mid
            # Fallback: return first available name
            for m in models:
                mid = (m.get('name') or '').strip()
                if mid:
                    return mid
    except Exception:
        pass
    return None
# --- End Model ID Retrieval (Ollama) ---

# --- New: CPU Model Retrieval ---
def get_cpu_info():
    """Get CPU model information"""
    # On Windows, platform.processor() usually provides complete information
    return platform.processor()
# --- End New ---


@app.get("/status")
def get_system_status():
    """
    Get current node's hardware status, lock status, and model ID.
    """
    # CPU information
    cpu_usage = psutil.cpu_percent(interval=1)

    # Memory information
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    status = {
        "locked": NODE_LOCKED,
        "model_id": get_current_model_id(),
        "cpu_usage_percent": cpu_usage,
        "cpu_model": get_cpu_info(),  # Use cached CPU model
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        },
        "memory_usage_percent": memory_usage, # Keep old field for compatibility
        "gpu": None,
        "memory_details": { # New detailed memory information
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2)
        }
    }

    # --- Network Rate (Maximum Network Interface) ---
    try:
        now = time.time()
        pernic = psutil.net_io_counters(pernic=True)
        prev_ts = _NET_SNAPSHOT["ts"]
        max_if = None
        max_total_mbps = 0.0
        rx_mbps = tx_mbps = 0.0

        if prev_ts is not None and now > prev_ts:
            dt = now - prev_ts
            for name, counters in pernic.items():
                prev = _NET_SNAPSHOT["pernic"].get(name)
                if not prev:
                    continue
                d_rx = max(0, counters.bytes_recv - prev.get("bytes_recv", 0))
                d_tx = max(0, counters.bytes_sent - prev.get("bytes_sent", 0))
                cur_rx_mbps = (d_rx * 8) / dt / 1e6
                cur_tx_mbps = (d_tx * 8) / dt / 1e6
                cur_total = cur_rx_mbps + cur_tx_mbps
                if cur_total > max_total_mbps:
                    max_total_mbps = cur_total
                    max_if = name
                    rx_mbps = cur_rx_mbps
                    tx_mbps = cur_tx_mbps

        # Update snapshot
        _NET_SNAPSHOT["ts"] = now
        _NET_SNAPSHOT["pernic"] = {
            name: {"bytes_sent": c.bytes_sent, "bytes_recv": c.bytes_recv}
            for name, c in pernic.items()
        }

        status["network"] = {
            "max_interface": max_if,
            "rx_mbps": round(rx_mbps, 2),
            "tx_mbps": round(tx_mbps, 2),
            "total_mbps": round(max_total_mbps, 2)
        }
    except Exception as e:
        status["network"] = {"error": str(e)}

    # --- Disk Rate (Maximum Device) ---
    try:
        now = time.time()
        perdisk = psutil.disk_io_counters(perdisk=True)
        prev_ts = _DISK_SNAPSHOT["ts"]
        max_dev = None
        max_total_MBps = 0.0
        read_MBps = write_MBps = 0.0

        if prev_ts is not None and now > prev_ts:
            dt = now - prev_ts
            for name, counters in perdisk.items():
                prev = _DISK_SNAPSHOT["perdisk"].get(name)
                if not prev:
                    continue
                d_r = max(0, counters.read_bytes - prev.get("read_bytes", 0))
                d_w = max(0, counters.write_bytes - prev.get("write_bytes", 0))
                cur_r_MBps = d_r / dt / (1024**2)
                cur_w_MBps = d_w / dt / (1024**2)
                cur_total = cur_r_MBps + cur_w_MBps
                if cur_total > max_total_MBps:
                    max_total_MBps = cur_total
                    max_dev = name
                    read_MBps = cur_r_MBps
                    write_MBps = cur_w_MBps

        _DISK_SNAPSHOT["ts"] = now
        _DISK_SNAPSHOT["perdisk"] = {
            name: {"read_bytes": c.read_bytes, "write_bytes": c.write_bytes}
            for name, c in perdisk.items()
        }

        status["disk"] = {
            "max_device": max_dev,
            "read_MBps": round(read_MBps, 2),
            "write_MBps": round(write_MBps, 2),
            "total_MBps": round(max_total_MBps, 2)
        }
    except Exception as e:
        status["disk"] = {"error": str(e)}

    if GPU_AVAILABLE:
        try:
            # Assume only one GPU, index 0
            handle = nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            utilization = nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            gpu_memory_utilization = utilization.memory

            # GPU memory
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total_gb = round(mem_info.total / (1024**3), 2)
            gpu_memory_used_gb = round(mem_info.used / (1024**3), 2)
            gpu_memory_free_gb = round(mem_info.free / (1024**3), 2)
            gpu_memory_percent = round((mem_info.used / mem_info.total) * 100, 2)

            # GPU temperature
            temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

            # GPU power
            power_watts = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W

            status["gpu"] = {
                "available": True,
                "utilization_percent": gpu_utilization,
                "memory_utilization_percent": gpu_memory_utilization,
                "memory_total_gb": gpu_memory_total_gb,
                "memory_used_gb": gpu_memory_used_gb,
                "memory_free_gb": gpu_memory_free_gb,
                "memory_usage_percent": gpu_memory_percent,
                "temperature_celsius": temperature,
                "power_watts": round(power_watts, 2),
                "type": "nvidia",
            }
        except NVMLError as e:
             status["gpu"] = {
                "available": False,
                "error": str(e)
            }
    else:
        # Apple Silicon (M series) monitoring: try to detect MPS and powermetrics
        is_macos = platform.system() == "Darwin"
        mps_available = False
        try:
            import torch  # Optional
            mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False

        gpu_info = {
            "available": False,
            "type": "mps" if is_macos else "unknown",
        }

        # Only try powermetrics on macOS (usually requires root privileges)
        if is_macos:
            try:
                # Sample once, return quickly
                proc = subprocess.run(
                    ["/usr/bin/powermetrics", "-n", "1", "--samplers", "gpu"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                stdout = proc.stdout or ""

                # Parse GPU Power (mW)
                power_match = re.search(r"GPU Power:\s*(\d+)\s*mW", stdout)
                if power_match:
                    power_mw = float(power_match.group(1))
                    gpu_info["power_watts"] = round(power_mw / 1000.0, 3)

                # Parse activity (may be represented as active residency or Busy, regex is as loose as possible)
                # Example: "GPU HW active residency: 23%"
                util_match = re.search(r"GPU .*active.  *?:\s*(\d+)%", stdout, re.IGNORECASE)
                if util_match:
                    gpu_info["utilization_percent"] = int(util_match.group(1))

                # If powermetrics outputs any GPU fields, consider GPU available
                if "power_watts" in gpu_info or "utilization_percent" in gpu_info:
                    gpu_info["available"] = True
            except subprocess.TimeoutExpired:
                gpu_info["error"] = "powermetrics timeout"
            except FileNotFoundError:
                gpu_info["error"] = "powermetrics not found (macOS system tool)"
            except PermissionError:
                gpu_info["error"] = "powermetrics needs root permission"
            except Exception as e:
                gpu_info["error"] = f"Unable to collect Apple GPU metrics: {e}"

        # If PyTorch MPS is available, mark as available (even without power/usage)
        if mps_available:
            gpu_info["available"] = True

        # Keep output structure consistent with NVIDIA, Apple GPU cannot directly read memory and temperature
        status["gpu"] = gpu_info
        
    return status

# --- Lock Control API ---
@app.post("/lock")
def lock_node():
    """Lock this node to prevent other requests from using it."""
    global NODE_LOCKED
    with lock:
        if NODE_LOCKED:
            # If node is already locked, return 409 Conflict error
            raise HTTPException(status_code=409, detail="Node is already locked")
        NODE_LOCKED = True
    return {"status": "success", "message": "Node locked"}

@app.post("/unlock")
def unlock_node():
    """Unlock this node to make it available for other requests."""
    global NODE_LOCKED
    with lock:
        NODE_LOCKED = False
    return {"status": "success", "message": "Node unlocked"}
# --- End Lock Control API ---


if __name__ == "__main__":
    # The proxy service will be accessible on the local network
    # You need to run this script on all 3 backend computers
    port = config('PORT', default=8001, cast=int)
    uvicorn.run(app, host="0.0.0.0", port=port) 
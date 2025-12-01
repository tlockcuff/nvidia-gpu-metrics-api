from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


# Docker Configuration Note:
# This service is designed to run with docker-compose.yml configured with:
#   - pid: "host" - Shares the host's PID namespace, allowing the container to see
#                   all host processes. The container's /proc IS the host's /proc.
#   - /proc:/host_proc:ro - Mounts host /proc as read-only (redundant with pid: host
#                            but provides an alternative access path if needed)
#
# With pid: "host", psutil automatically sees all host processes without any special
# configuration. Process PIDs from pynvml (GPU processes) are host PIDs, which
# psutil can resolve directly since they share the same PID namespace.
#
# This enables:
#   - Host process monitoring: _get_top_memory_processes() sees all host processes
#   - GPU process resolution: _get_gpu_processes() resolves GPU PIDs to process names


class GPUResponseModel(BaseModel):
    timestamp: str
    system_info: Dict[str, Any]
    gpu_count: int
    gpus: List[Dict[str, Any]]
    summary: Dict[str, Any]
    status: str


app = FastAPI(title="GPU Metrics Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get_valid_api_keys() -> List[str]:
    """Get valid API keys from environment variables."""
    api_keys = []
    # Support single API_KEY
    api_key = os.getenv("API_KEY")
    if api_key:
        api_keys.append(api_key)
    # Support comma-separated API_KEYS
    api_keys_env = os.getenv("API_KEYS")
    if api_keys_env:
        api_keys.extend([key.strip() for key in api_keys_env.split(",") if key.strip()])
    return api_keys


def _validate_api_key(api_key: Optional[str] = None, x_api_key: Optional[str] = None) -> bool:
    """Validate API key from query parameter or header."""
    valid_keys = _get_valid_api_keys()
    # If no API keys are configured, allow access (backward compatibility)
    if not valid_keys:
        return True
    
    # Check query parameter first, then header
    provided_key = api_key or x_api_key
    if not provided_key:
        return False
    
    return provided_key in valid_keys


def _get_cuda_version() -> str:
    # Prefer nvidia-smi; fallback to empty string if unavailable
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # If multiple GPUs, they might report same version; take the first non-empty
        for line in out.splitlines():
            if line and line != "N/A":
                return line
    except Exception:
        pass
    return ""


def _pcie_max_bandwidth_mbps(gen: int, width: int) -> float:
    # Approximate effective per-lane GB/s (unidirectional), accounting for encoding overhead
    # Gen1: 2.5 GT/s ~ 0.250 GB/s, Gen2: 5.0 GT/s ~ 0.500 GB/s
    # Gen3: 8.0 GT/s ~ 0.985 GB/s, Gen4: 16.0 GT/s ~ 1.969 GB/s, Gen5: 32.0 GT/s ~ 3.938 GB/s
    # Convert to MB/s and multiply by width
    per_lane_gbps = {
        1: 0.250,
        2: 0.500,
        3: 0.985,
        4: 1.969,
        5: 3.938,
        6: 7.877,  # Gen6 estimate
    }.get(gen, 0.0)
    per_lane_MBps = per_lane_gbps * 1000
    return per_lane_MBps * float(width)


def _get_cpu_stats() -> Dict[str, Any]:
    """Gather CPU statistics."""
    if psutil is None:
        return {
            "cpu_count_physical": 0,
            "cpu_count_logical": 0,
            "cpu_percent": 0.0,
            "cpu_per_core_percent": [],
            "cpu_freq_current_mhz": 0.0,
            "cpu_freq_min_mhz": 0.0,
            "cpu_freq_max_mhz": 0.0,
        }
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            "cpu_count_physical": psutil.cpu_count(logical=False) or 0,
            "cpu_count_logical": psutil.cpu_count(logical=True) or 0,
            "cpu_percent": round(cpu_percent, 1),
            "cpu_per_core_percent": [round(p, 1) for p in cpu_per_core],
            "cpu_freq_current_mhz": round(cpu_freq.current, 1) if cpu_freq else 0.0,
            "cpu_freq_min_mhz": round(cpu_freq.min, 1) if cpu_freq else 0.0,
            "cpu_freq_max_mhz": round(cpu_freq.max, 1) if cpu_freq else 0.0,
        }
    except Exception:
        return {
            "cpu_count_physical": 0,
            "cpu_count_logical": 0,
            "cpu_percent": 0.0,
            "cpu_per_core_percent": [],
            "cpu_freq_current_mhz": 0.0,
            "cpu_freq_min_mhz": 0.0,
            "cpu_freq_max_mhz": 0.0,
        }


def _get_memory_stats() -> Dict[str, Any]:
    """Gather system memory statistics."""
    if psutil is None:
        return {
            "total_mb": 0,
            "available_mb": 0,
            "used_mb": 0,
            "percent": 0.0,
            "cached_mb": 0,
            "buffers_mb": 0,
            "top_processes": [],
        }
    
    try:
        mem = psutil.virtual_memory()
        total_memory_mb = mem.total / (1024 * 1024)
        top_processes = _get_top_memory_processes(limit=5, total_memory_mb=total_memory_mb)
        return {
            "total_mb": int(round(mem.total / (1024 * 1024))),
            "available_mb": int(round(mem.available / (1024 * 1024))),
            "used_mb": int(round(mem.used / (1024 * 1024))),
            "percent": round(mem.percent, 1),
            "cached_mb": int(round(getattr(mem, 'cached', 0) / (1024 * 1024))),
            "buffers_mb": int(round(getattr(mem, 'buffers', 0) / (1024 * 1024))),
            "top_processes": top_processes,
        }
    except Exception:
        return {
            "total_mb": 0,
            "available_mb": 0,
            "used_mb": 0,
            "percent": 0.0,
            "cached_mb": 0,
            "buffers_mb": 0,
            "top_processes": [],
        }


# Module-level storage for disk I/O rate calculation
_disk_io_cache: Dict[str, Tuple[float, int, int, int, int]] = {}  # device -> (timestamp, read_bytes, write_bytes, read_count, write_count)

# Module-level storage for GPU peak values and rate calculations
_gpu_peak_cache: Dict[int, Dict[str, float]] = {}  # gpu_id -> {metric_name: peak_value}
_gpu_rate_cache: Dict[int, Tuple[float, float, float]] = {}  # gpu_id -> (timestamp, prev_temp, prev_power)


def _get_disk_stats() -> Dict[str, Any]:
    """Gather disk usage statistics and I/O activity - returns used/available space and I/O metrics."""
    if psutil is None:
        return {
            "used_gb": 0.0,
            "available_gb": 0.0,
            "io": {
                "read_bytes_per_sec": 0.0,
                "write_bytes_per_sec": 0.0,
                "read_ops_per_sec": 0.0,
                "write_ops_per_sec": 0.0,
                "total_read_bytes": 0,
                "total_write_bytes": 0,
                "total_read_ops": 0,
                "total_write_ops": 0,
            }
        }
    
    total_used_gb = 0.0
    total_available_gb = 0.0
    
    # Disk I/O statistics
    total_read_bytes = 0
    total_write_bytes = 0
    total_read_count = 0
    total_write_count = 0
    
    # Initialize rate variables
    read_bytes_per_sec = 0.0
    write_bytes_per_sec = 0.0
    read_ops_per_sec = 0.0
    write_ops_per_sec = 0.0
    
    current_time = time.time()
    
    try:
        # Get disk I/O counters (aggregated across all disks)
        io_counters = psutil.disk_io_counters()
        if io_counters:
            # Handle None values that can occur on some systems (e.g., macOS)
            # psutil may return None for read_bytes/read_count on some platforms
            total_read_bytes = getattr(io_counters, 'read_bytes', None)
            total_write_bytes = getattr(io_counters, 'write_bytes', None)
            total_read_count = getattr(io_counters, 'read_count', None)
            total_write_count = getattr(io_counters, 'write_count', None)
            
            # Convert None to 0 for calculations and display
            total_read_bytes = total_read_bytes if total_read_bytes is not None else 0
            total_write_bytes = total_write_bytes if total_write_bytes is not None else 0
            total_read_count = total_read_count if total_read_count is not None else 0
            total_write_count = total_write_count if total_write_count is not None else 0
            
            # Calculate rates using cached previous values
            read_bytes_per_sec = 0.0
            write_bytes_per_sec = 0.0
            read_ops_per_sec = 0.0
            write_ops_per_sec = 0.0
            
            cache_key = "system"
            if cache_key in _disk_io_cache:
                prev_time, prev_read_bytes, prev_write_bytes, prev_read_count, prev_write_count = _disk_io_cache[cache_key]
                time_delta = current_time - prev_time
                
                if time_delta > 0:
                    # Calculate rates - handle potential counter wraparound by checking for negative deltas
                    read_bytes_delta = total_read_bytes - prev_read_bytes
                    write_bytes_delta = total_write_bytes - prev_write_bytes
                    read_count_delta = total_read_count - prev_read_count
                    write_count_delta = total_write_count - prev_write_count
                    
                    # Only calculate positive rates (negative indicates counter reset/wraparound)
                    if read_bytes_delta >= 0:
                        read_bytes_per_sec = read_bytes_delta / time_delta
                    if write_bytes_delta >= 0:
                        write_bytes_per_sec = write_bytes_delta / time_delta
                    if read_count_delta >= 0:
                        read_ops_per_sec = read_count_delta / time_delta
                    if write_count_delta >= 0:
                        write_ops_per_sec = write_count_delta / time_delta
            
            # Update cache
            _disk_io_cache[cache_key] = (current_time, total_read_bytes, total_write_bytes, total_read_count, total_write_count)
        else:
            read_bytes_per_sec = 0.0
            write_bytes_per_sec = 0.0
            read_ops_per_sec = 0.0
            write_ops_per_sec = 0.0
        
        # Get disk usage statistics
        partitions = psutil.disk_partitions()
        seen_devices = set()
        
        for partition in partitions:
            try:
                # Avoid counting the same physical device multiple times
                # Extract base device name (e.g., '/dev/sda1' -> 'sda')
                device_name = partition.device.split('/')[-1] if partition.device else ""
                if device_name and device_name[-1].isdigit():
                    base_device = device_name.rstrip('0123456789')
                else:
                    base_device = device_name
                
                # Only count each physical device once
                if base_device and base_device not in seen_devices:
                    usage = psutil.disk_usage(partition.mountpoint)
                    total_used_gb += usage.used / (1024 ** 3)
                    total_available_gb += usage.free / (1024 ** 3)
                    seen_devices.add(base_device)
            except (PermissionError, OSError):
                # Skip partitions we can't access
                continue
    except Exception:
        pass
    
    return {
        "used_gb": round(total_used_gb, 2),
        "available_gb": round(total_available_gb, 2),
        "io": {
            "read_bytes_per_sec": round(read_bytes_per_sec, 2),
            "write_bytes_per_sec": round(write_bytes_per_sec, 2),
            "read_ops_per_sec": round(read_ops_per_sec, 2),
            "write_ops_per_sec": round(write_ops_per_sec, 2),
            "total_read_bytes": total_read_bytes,
            "total_write_bytes": total_write_bytes,
            "total_read_ops": total_read_count,
            "total_write_ops": total_write_count,
        }
    }


def _get_temperature_stats() -> Dict[str, Any]:
    """Gather system temperature statistics - returns highest temperature."""
    if psutil is None:
        return {
            "temperature_celsius": 0.0,
        }
    
    max_temp = 0.0
    try:
        # psutil.sensors_temperatures() returns a dict of sensor labels to temperature readings
        sensors = psutil.sensors_temperatures()
        if sensors:
            for sensor_name, sensor_readings in sensors.items():
                for reading in sensor_readings:
                    if reading.current > max_temp:
                        max_temp = reading.current
        
        return {
            "temperature_celsius": round(max_temp, 1),
        }
    except Exception:
        return {
            "temperature_celsius": 0.0,
        }


def _get_top_memory_processes(limit: int = 5, total_memory_mb: float = 0.0) -> List[Dict[str, Any]]:
    """
    Get top N processes by memory consumption on the host system.
    
    With docker-compose.yml configured with `pid: "host"`, the container shares
    the host's PID namespace. This means psutil.process_iter() automatically sees
    all host processes without any special configuration. The container's /proc
    filesystem IS the host's /proc, so process information is directly accessible.
    
    Returns a list of dictionaries containing:
    - pid: Process ID (host PID)
    - name: Process name
    - memory_mb: Resident Set Size (RSS) in MB
    - memory_percent: Percentage of total system memory used by this process
    - username: Process owner username
    
    Args:
        limit: Maximum number of processes to return (default: 5)
        total_memory_mb: Total system memory in MB for percentage calculation (default: 0.0)
        
    Returns:
        List of process dictionaries, sorted by memory usage (descending)
    """
    if psutil is None:
        return []
    
    processes = []
    try:
        # With pid: "host", this iterates over all host processes
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'username']):
            try:
                pinfo = proc.info
                mem_info = pinfo.get('memory_info')
                if mem_info:
                    memory_mb = round(mem_info.rss / (1024 * 1024), 1)
                    # Calculate memory percentage if total memory is provided
                    memory_percent = round((memory_mb / total_memory_mb * 100.0), 2) if total_memory_mb > 0 else 0.0
                    
                    processes.append({
                        "pid": pinfo['pid'],
                        "name": pinfo.get('name', ''),
                        "memory_mb": memory_mb,
                        "memory_percent": memory_percent,
                        "username": pinfo.get('username', ''),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process disappeared, permission denied, or is a zombie - skip it
                continue
            except Exception:
                # Unexpected error for this process - skip and continue
                continue
    except Exception:
        # If process_iter() itself fails, return empty list
        return []
    
    # Sort by memory usage (descending) and return top N
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    return processes[:limit]


def _get_gpu_processes(handle: Any, limit: int = 5, total_gpu_memory_mb: float = 0.0) -> List[Dict[str, Any]]:
    """
    Get top N processes using GPU memory for a specific GPU device.
    
    This function queries NVML for processes using the GPU, then resolves process
    names and usernames using psutil. With docker-compose.yml configured with
    `pid: "host"`, the PIDs returned by pynvml are host PIDs, which psutil can
    resolve directly since it shares the same PID namespace.
    
    Process resolution flow:
    1. pynvml returns host PIDs from GPU processes
    2. psutil.Process(pid) resolves these PIDs to process names/usernames
    3. Both operate in the same PID namespace (host), so resolution is seamless
    
    Returns a list of dictionaries containing:
    - pid: Process ID (host PID from pynvml)
    - name: Process name (resolved via psutil)
    - memory_mb: GPU memory usage in MB
    - memory_percent: Percentage of total GPU memory used by this process
    - username: Process owner username (resolved via psutil)
    - cpu_percent: CPU usage percentage for this process
    
    Args:
        handle: NVML device handle for the GPU
        limit: Maximum number of processes to return (default: 5)
        total_gpu_memory_mb: Total GPU memory in MB for percentage calculation (default: 0.0)
        
    Returns:
        List of GPU process dictionaries, sorted by GPU memory usage (descending)
    """
    if pynvml is None:
        return []
    
    processes = []
    try:
        # Try to get compute processes (CUDA compute workloads)
        try:
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in compute_procs:
                try:
                    # Get process name, username, and CPU usage using psutil
                    # With pid: "host", proc.pid is a host PID, so psutil resolves it directly
                    proc_name = ""
                    username = ""
                    cpu_percent = 0.0
                    if psutil:
                        try:
                            p = psutil.Process(proc.pid)
                            proc_name = p.name()
                            username = p.username()
                            # Get CPU usage (non-blocking, uses cached value or 0.0)
                            cpu_percent = p.cpu_percent(interval=None)
                        except psutil.NoSuchProcess:
                            # Process disappeared between pynvml query and psutil lookup
                            # This can happen due to race conditions - continue with PID only
                            pass
                        except psutil.AccessDenied:
                            # Permission denied - continue without name/username
                            pass
                        except psutil.ZombieProcess:
                            # Process is a zombie - continue without name/username
                            pass
                    
                    memory_mb = round(proc.usedGpuMemory / (1024 * 1024), 1)
                    # Calculate GPU memory percentage if total GPU memory is provided
                    memory_percent = round((memory_mb / total_gpu_memory_mb * 100.0), 2) if total_gpu_memory_mb > 0 else 0.0
                    
                    processes.append({
                        "pid": proc.pid,
                        "name": proc_name,
                        "memory_mb": memory_mb,
                        "memory_percent": memory_percent,
                        "username": username,
                        "cpu_percent": round(cpu_percent, 1),
                    })
                except Exception:
                    # Unexpected error for this process - skip and continue
                    continue
        except pynvml.NVMLError:
            # GPU doesn't support compute process queries or error occurred
            pass
        
        # Try to get graphics processes (OpenGL/Vulkan graphics workloads)
        try:
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            for proc in graphics_procs:
                try:
                    # Check if we already have this PID from compute processes
                    # (a process can appear in both lists)
                    existing_pid = any(p['pid'] == proc.pid for p in processes)
                    if not existing_pid:
                        # Get process name, username, and CPU usage using psutil
                        proc_name = ""
                        username = ""
                        cpu_percent = 0.0
                        if psutil:
                            try:
                                p = psutil.Process(proc.pid)
                                proc_name = p.name()
                                username = p.username()
                                # Get CPU usage (non-blocking, uses cached value or 0.0)
                                cpu_percent = p.cpu_percent(interval=None)
                            except psutil.NoSuchProcess:
                                # Process disappeared between pynvml query and psutil lookup
                                pass
                            except psutil.AccessDenied:
                                # Permission denied - continue without name/username
                                pass
                            except psutil.ZombieProcess:
                                # Process is a zombie - continue without name/username
                                pass
                        
                        memory_mb = round(proc.usedGpuMemory / (1024 * 1024), 1)
                        # Calculate GPU memory percentage if total GPU memory is provided
                        memory_percent = round((memory_mb / total_gpu_memory_mb * 100.0), 2) if total_gpu_memory_mb > 0 else 0.0
                        
                        processes.append({
                            "pid": proc.pid,
                            "name": proc_name,
                            "memory_mb": memory_mb,
                            "memory_percent": memory_percent,
                            "username": username,
                            "cpu_percent": round(cpu_percent, 1),
                        })
                except Exception:
                    # Unexpected error for this process - skip and continue
                    continue
        except pynvml.NVMLError:
            # GPU doesn't support graphics process queries or error occurred
            pass
        
        # Sort by GPU memory usage (descending) and return top N
        processes.sort(key=lambda x: x['memory_mb'], reverse=True)
        return processes[:limit]
    except Exception:
        # If anything unexpected happens, return empty list
        return []


def _get_system_info() -> Dict[str, Any]:
    """Gather system information like OS, architecture, hostname, etc."""
    try:
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
    except Exception:
        return {
            "os": "",
            "os_release": "",
            "os_version": "",
            "architecture": "",
            "processor": "",
            "hostname": "",
            "python_version": "",
            "platform": "",
        }


def _gather_metrics() -> GPUResponseModel:
    # Gather system metrics (CPU, memory, disk, temperature)
    cpu_stats = _get_cpu_stats()
    memory_stats = _get_memory_stats()
    disk_stats = _get_disk_stats()
    temperature_stats = _get_temperature_stats()
    system_info_data = _get_system_info()
    
    if pynvml is None:
        return GPUResponseModel(
            timestamp=_now_iso(),
            system_info={
                **system_info_data,
                "driver_version": "",
                "cuda_version": "",
                "nvml_version": "",
                "cpu": cpu_stats,
                "memory": memory_stats,
                "disk_used_gb": disk_stats["used_gb"],
                "disk_available_gb": disk_stats["available_gb"],
                "disk_io": disk_stats["io"],
                "temperature_celsius": temperature_stats["temperature_celsius"],
            },
            gpu_count=0,
            gpus=[],
            summary={
                "total_memory_mb": 0,
                "total_used_memory_mb": 0,
                "memory_utilization_percent": 0,
                "average_gpu_utilization_percent": 0,
                "average_temperature_celsius": 0,
                "total_power_usage_watts": 0,
            },
            status="error",
        )

    pynvml.nvmlInit()
    try:
        # Handle both string and bytes return types for compatibility
        driver_version_raw = pynvml.nvmlSystemGetDriverVersion()
        driver_version = driver_version_raw.decode() if isinstance(driver_version_raw, bytes) else driver_version_raw
        
        nvml_version_raw = pynvml.nvmlSystemGetNVMLVersion()
        nvml_version = nvml_version_raw.decode() if isinstance(nvml_version_raw, bytes) else nvml_version_raw
        
        cuda_version = _get_cuda_version()

        device_count = pynvml.nvmlDeviceGetCount()
        gpus: List[Dict[str, Any]] = []

        total_memory_mb = 0.0
        total_used_memory_mb = 0.0
        total_power_watts = 0.0
        sum_gpu_util = 0.0
        sum_temp_c = 0.0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw

            # Performance State (P-State)
            try:
                perf_state = int(pynvml.nvmlDeviceGetPerformanceState(handle))
            except pynvml.NVMLError:
                perf_state = -1  # Unknown/unsupported

            # Compute Capability
            try:
                compute_major, compute_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{compute_major}.{compute_minor}"
            except pynvml.NVMLError:
                compute_capability = ""

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            free_mb = mem.free / (1024 * 1024)
            mem_util = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
            mem_util_controller = float(util.memory)

            temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

            # Power (mW to W)
            try:
                power_mw = float(pynvml.nvmlDeviceGetPowerUsage(handle))
                power_w = power_mw / 1000.0
            except pynvml.NVMLError:
                power_w = 0.0
            try:
                power_limit_mw = float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
                power_limit_w = power_limit_mw / 1000.0
            except pynvml.NVMLError:
                power_limit_mw = 0.0
                power_limit_w = 0.0
            power_eff = (power_w / power_limit_w * 100.0) if power_limit_w > 0 else 0.0
            power_budget_w = max(0.0, power_limit_w - power_w) if power_limit_w > 0 else 0.0

            # Rate of change calculations and peak tracking
            current_time = time.time()
            temp_delta_per_sec = 0.0
            power_delta_per_sec = 0.0
            
            if i in _gpu_rate_cache:
                prev_time, prev_temp, prev_power = _gpu_rate_cache[i]
                time_delta = current_time - prev_time
                if time_delta > 0:
                    temp_delta = temp_c - prev_temp
                    power_delta = power_w - prev_power
                    temp_delta_per_sec = temp_delta / time_delta
                    power_delta_per_sec = power_delta / time_delta
            
            _gpu_rate_cache[i] = (current_time, temp_c, power_w)
            
            # Peak value tracking
            if i not in _gpu_peak_cache:
                _gpu_peak_cache[i] = {
                    "gpu_utilization_percent": gpu_util,
                    "power_watts": power_w,
                    "temperature_celsius": temp_c,
                }
            else:
                _gpu_peak_cache[i]["gpu_utilization_percent"] = max(_gpu_peak_cache[i]["gpu_utilization_percent"], gpu_util)
                _gpu_peak_cache[i]["power_watts"] = max(_gpu_peak_cache[i]["power_watts"], power_w)
                _gpu_peak_cache[i]["temperature_celsius"] = max(_gpu_peak_cache[i]["temperature_celsius"], temp_c)
            
            peak_gpu_util = _gpu_peak_cache[i]["gpu_utilization_percent"]
            peak_power = _gpu_peak_cache[i]["power_watts"]
            peak_temp = _gpu_peak_cache[i]["temperature_celsius"]

            # Clocks
            try:
                gfx_clock = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                mem_clock = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            except pynvml.NVMLError:
                gfx_clock = 0.0
                mem_clock = 0.0
            try:
                max_gfx_clock = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                max_mem_clock = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            except pynvml.NVMLError:
                max_gfx_clock = 0.0
                max_mem_clock = 0.0
            clock_eff = (gfx_clock / max_gfx_clock * 100.0) if max_gfx_clock > 0 else 0.0

            # PCIe
            try:
                pcie_gen = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
                pcie_width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
            except pynvml.NVMLError:
                pcie_gen = 0
                pcie_width = 0

            # Throughput (KB/s); NVML may return per-direction throughput
            try:
                tx_kbs = float(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
                rx_kbs = float(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))
                # Convert to MB/s
                tx_mbps = tx_kbs / 1024.0
                rx_mbps = rx_kbs / 1024.0
            except pynvml.NVMLError:
                tx_mbps = 0.0
                rx_mbps = 0.0
            total_mbps = tx_mbps + rx_mbps
            max_bw_mbps = _pcie_max_bandwidth_mbps(pcie_gen, pcie_width)
            pcie_util = (total_mbps / max_bw_mbps * 100.0) if max_bw_mbps > 0 else 0.0

            # Throttling / Bottlenecks (heuristic mapping)
            bottlenecks: List[str] = []
            try:
                reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            except pynvml.NVMLError:
                reasons = 0
            is_throttled = reasons != 0

            # Map NVML throttle reasons into our labels
            if reasons & getattr(pynvml, "nvmlClocksThrottleReasonThermal", 0):
                bottlenecks.append("thermal_throttling")
            if reasons & getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0):
                # Often due to power or temperature; categorize as power throttling if power close to limit
                if power_limit_w > 0 and power_w >= 0.95 * power_limit_w:
                    bottlenecks.append("power_throttling")
                else:
                    bottlenecks.append("board_power_throttling")
            if mem_util >= 80.0:
                bottlenecks.append("high_memory_usage")
            # memory-bound workload heuristic: high mem util vs low gpu util
            if mem_util >= 60.0 and gpu_util <= 40.0:
                bottlenecks.append("memory_bound_workload")
            if pcie_util >= 80.0:
                bottlenecks.append("pcie_bandwidth_saturated")

            # Fan
            try:
                fan_rpm = float(pynvml.nvmlDeviceGetFanSpeed_v2(handle).speed)
            except Exception:
                # Some GPUs expose only percent, not RPM
                try:
                    fan_percent = float(pynvml.nvmlDeviceGetFanSpeed(handle))
                    # Keep as percent in RPM field if RPM unavailable
                    fan_rpm = fan_percent
                except Exception:
                    fan_rpm = 0.0

            # More nuanced status detection with lower thresholds and multiple indicators
            is_active = (
                gpu_util >= 5.0 or  # Lower GPU utilization threshold
                power_w > 10.0 or   # Lower power threshold (some GPUs idle at ~15W)
                mem_util >= 5.0 or  # Memory utilization indicates activity
                gfx_clock > max_gfx_clock * 0.3 or  # Graphics clock above base/idle levels
                temp_c > 40.0       # Temperature above typical idle (usually 30-35C)
            )
            status = "active" if is_active else "idle"

            # Get top GPU processes
            gpu_processes = _get_gpu_processes(handle, limit=5, total_gpu_memory_mb=total_mb)

            gpus.append(
                {
                    "id": i,
                    "timestamp": _now_iso(),
                    "name": name,
                    "compute_capability": compute_capability,
                    "performance_state": perf_state,
                    "memory": {
                        "total_mb": round(total_mb, 1),
                        "used_mb": round(used_mb, 1),
                        "free_mb": round(free_mb, 1),
                        "utilization_percent": round(mem_util, 1),
                        "bandwidth_utilization_percent": round(pcie_util, 1),
                    },
                    "utilization": {
                        "gpu_percent": int(round(gpu_util)),
                        "memory_percent": int(round(mem_util_controller)),
                        "sm_percent": int(round(gpu_util)),  # Approximation
                    },
                    "temperature_celsius": int(round(temp_c)),
                    "power": {
                        "usage_watts": round(power_w, 1),
                        "limit_watts": int(round(power_limit_mw)),  # UI divides by 1000
                        "budget_watts": round(power_budget_w, 1),
                        "efficiency_percent": int(round(power_eff)),
                    },
                    "rate_of_change": {
                        "temperature_delta_per_sec": round(temp_delta_per_sec, 2),
                        "power_delta_per_sec": round(power_delta_per_sec, 2),
                    },
                    "peak_values": {
                        "gpu_utilization_percent": round(peak_gpu_util, 1),
                        "power_watts": round(peak_power, 1),
                        "temperature_celsius": round(peak_temp, 1),
                    },
                    "clocks": {
                        "graphics_mhz": int(round(gfx_clock)),
                        "memory_mhz": int(round(mem_clock)),
                        "max_graphics_mhz": int(round(max_gfx_clock)),
                        "max_memory_mhz": int(round(max_mem_clock)),
                        "efficiency_percent": int(round(clock_eff)),
                    },
                    "pcie": {
                        "generation": pcie_gen,
                        "width": pcie_width,
                        "tx_throughput_mbps": round(tx_mbps, 1),
                        "rx_throughput_mbps": round(rx_mbps, 1),
                        "total_throughput_mbps": round(total_mbps, 1),
                        "max_bandwidth_mbps": round(max_bw_mbps, 1),
                        "utilization_percent": int(round(pcie_util)),
                    },
                    "throttling": {
                        "reasons": int(reasons),
                        "is_throttled": bool(is_throttled),
                        "bottlenecks": bottlenecks,
                    },
                    "fan_speed_rpm": int(round(fan_rpm)),
                    "status": status,
                    "top_processes": gpu_processes,
                    "debug_info": {
                        "gpu_util_threshold": f"{gpu_util:.1f}% >= 5.0%",
                        "power_threshold": f"{power_w:.1f}W > 10.0W",
                        "memory_util_threshold": f"{mem_util:.1f}% >= 5.0%",
                        "clock_threshold": f"{gfx_clock:.0f}MHz > {max_gfx_clock * 0.3:.0f}MHz",
                        "temp_threshold": f"{temp_c:.1f}°C > 40.0°C",
                        "is_active": is_active,
                    },
                }
            )

            total_memory_mb += total_mb
            total_used_memory_mb += used_mb
            total_power_watts += power_w
            sum_gpu_util += gpu_util
            sum_temp_c += temp_c

        avg_gpu_util = (sum_gpu_util / device_count) if device_count > 0 else 0.0
        avg_temp_c = (sum_temp_c / device_count) if device_count > 0 else 0.0
        mem_util_total = (
            (total_used_memory_mb / total_memory_mb * 100.0) if total_memory_mb > 0 else 0.0
        )

        return GPUResponseModel(
            timestamp=_now_iso(),
            system_info={
                **system_info_data,
                "driver_version": driver_version,
                "cuda_version": cuda_version,
                "nvml_version": nvml_version,
                "cpu": cpu_stats,
                "memory": memory_stats,
                "disk_used_gb": disk_stats["used_gb"],
                "disk_available_gb": disk_stats["available_gb"],
                "disk_io": disk_stats["io"],
                "temperature_celsius": temperature_stats["temperature_celsius"],
            },
            gpu_count=device_count,
            gpus=gpus,
            summary={
                "total_memory_mb": int(round(total_memory_mb)),
                "total_used_memory_mb": int(round(total_used_memory_mb)),
                "memory_utilization_percent": round(mem_util_total, 1),
                "average_gpu_utilization_percent": round(avg_gpu_util, 1),
                "average_temperature_celsius": int(round(avg_temp_c)),
                "total_power_usage_watts": round(total_power_watts, 1),
            },
            status="ok",
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _generate_typescript_types() -> str:
    """Generate TypeScript type definitions for the GPU metrics API response."""
    return """// TypeScript type definitions for NVIDIA GPU Metrics API
// Generated automatically - do not edit manually

export interface GPUProcess {
  pid: number;
  name: string;
  memory_mb: number;
  memory_percent: number;
  username: string;
  cpu_percent: number;
}

export interface GPUMemory {
  total_mb: number;
  used_mb: number;
  free_mb: number;
  utilization_percent: number;
  bandwidth_utilization_percent: number;
}

export interface GPUUtilization {
  gpu_percent: number;
  memory_percent: number;
  sm_percent: number;
}

export interface GPUPower {
  usage_watts: number;
  limit_watts: number;
  budget_watts: number;
  efficiency_percent: number;
}

export interface GPURateOfChange {
  temperature_delta_per_sec: number;
  power_delta_per_sec: number;
}

export interface GPUPeakValues {
  gpu_utilization_percent: number;
  power_watts: number;
  temperature_celsius: number;
}

export interface GPUClocks {
  graphics_mhz: number;
  memory_mhz: number;
  max_graphics_mhz: number;
  max_memory_mhz: number;
  efficiency_percent: number;
}

export interface GPUPCIe {
  generation: number;
  width: number;
  tx_throughput_mbps: number;
  rx_throughput_mbps: number;
  total_throughput_mbps: number;
  max_bandwidth_mbps: number;
  utilization_percent: number;
}

export interface GPUThrottling {
  reasons: number;
  is_throttled: boolean;
  bottlenecks: string[];
}

export interface GPUDebugInfo {
  gpu_util_threshold: string;
  power_threshold: string;
  memory_util_threshold: string;
  clock_threshold: string;
  temp_threshold: string;
  is_active: boolean;
}

export interface GPU {
  id: number;
  timestamp: string;
  name: string;
  compute_capability: string;
  performance_state: number;
  memory: GPUMemory;
  utilization: GPUUtilization;
  temperature_celsius: number;
  power: GPUPower;
  rate_of_change: GPURateOfChange;
  peak_values: GPUPeakValues;
  clocks: GPUClocks;
  pcie: GPUPCIe;
  throttling: GPUThrottling;
  fan_speed_rpm: number;
  status: "active" | "idle";
  top_processes: GPUProcess[];
  debug_info: GPUDebugInfo;
}

export interface CPUStats {
  cpu_count_physical: number;
  cpu_count_logical: number;
  cpu_percent: number;
  cpu_per_core_percent: number[];
  cpu_freq_current_mhz: number;
  cpu_freq_min_mhz: number;
  cpu_freq_max_mhz: number;
}

export interface MemoryStats {
  total_mb: number;
  available_mb: number;
  used_mb: number;
  percent: number;
  cached_mb: number;
  buffers_mb: number;
  top_processes: Array<{
    pid: number;
    name: string;
    memory_mb: number;
    memory_percent: number;
    username: string;
  }>;
}

export interface DiskIO {
  read_bytes_per_sec: number;
  write_bytes_per_sec: number;
  read_ops_per_sec: number;
  write_ops_per_sec: number;
  total_read_bytes: number;
  total_write_bytes: number;
  total_read_ops: number;
  total_write_ops: number;
}

export interface SystemInfo {
  os: string;
  os_release: string;
  os_version: string;
  architecture: string;
  processor: string;
  hostname: string;
  python_version: string;
  platform: string;
  driver_version: string;
  cuda_version: string;
  nvml_version: string;
  cpu: CPUStats;
  memory: MemoryStats;
  disk_used_gb: number;
  disk_available_gb: number;
  disk_io: DiskIO;
  temperature_celsius: number;
}

export interface GPUSummary {
  total_memory_mb: number;
  total_used_memory_mb: number;
  memory_utilization_percent: number;
  average_gpu_utilization_percent: number;
  average_temperature_celsius: number;
  total_power_usage_watts: number;
}

export interface GPUResponse {
  timestamp: string;
  system_info: SystemInfo;
  gpu_count: number;
  gpus: GPU[];
  summary: GPUSummary;
  status: "ok" | "error";
}

// Helper type for SSE stream events
export interface GPUStreamEvent {
  data: GPUResponse;
  event?: string;
  id?: string;
}
"""


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/types", response_class=Response)
def get_typescript_types():
    """
    Returns TypeScript type definitions for the GPU metrics API response.
    
    This endpoint provides complete TypeScript type definitions that match
    the structure of the `/gpu` endpoint response. Use these types in your
    TypeScript/JavaScript projects for type safety and better IDE support.
    
    **Usage:**
    ```typescript
    // Download types
    // curl http://localhost:8008/types > gpu-metrics-types.ts
    
    // Import in your code
    import { GPUResponse, GPU } from './gpu-metrics-types';
    
    const response: GPUResponse = await fetch('/gpu').then(r => r.json());
    ```
    """
    return Response(
        content=_generate_typescript_types(),
        media_type="application/typescript",
        headers={
            "Content-Disposition": 'attachment; filename="gpu-metrics-types.ts"',
        },
    )


@app.get("/gpu", response_model=GPUResponseModel)
def get_gpu() -> GPUResponseModel:
    return _gather_metrics()


async def _generate_sse_stream(interval: float):
    """Generate SSE stream of GPU metrics."""
    while True:
        try:
            # Gather metrics
            metrics = _gather_metrics()
            
            # Convert to JSON
            json_data = metrics.model_dump_json()
            
            # Format as SSE message
            # SSE format: "data: {json}\n\n"
            message = f"data: {json_data}\n\n"
            
            yield message
            
            # Wait for the specified interval
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            # Client disconnected, clean up
            break
        except Exception as e:
            # Send error event and continue streaming
            error_data = {
                "error": str(e),
                "timestamp": _now_iso(),
                "status": "error"
            }
            error_message = f"data: {json.dumps(error_data)}\n\n"
            yield error_message
            # Wait before retrying
            await asyncio.sleep(interval)


@app.get("/gpu/stream")
async def stream_gpu(
    api_key: Optional[str] = Query(None, description="API key for authentication"),
    interval: float = Query(2.0, ge=0.5, le=60.0, description="Update interval in seconds (0.5-60)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="API key in header")
):
    """
    Stream GPU metrics in real-time using Server-Sent Events (SSE).
    
    This endpoint provides a continuous stream of GPU metrics, eliminating the need
    for client-side polling. Metrics are sent at the specified interval.
    
    **Authentication:**
    - Provide API key via query parameter: `?api_key=your_key`
    - Or via header: `X-API-Key: your_key`
    - If no API keys are configured in environment, authentication is optional
    
    **Parameters:**
    - `interval`: Update frequency in seconds (default: 2.0, min: 0.5, max: 60.0)
    - `api_key`: API key for authentication (query parameter)
    
    **Response:**
    - Content-Type: `text/event-stream`
    - Each message is a JSON object following the same structure as `/gpu` endpoint
    - Messages are formatted as SSE events: `data: {json}\n\n`
    
    **Example Client (JavaScript):**
    ```javascript
    const eventSource = new EventSource('http://localhost:8008/gpu/stream?api_key=your_key&interval=1');
    eventSource.onmessage = (event) => {
        const metrics = JSON.parse(event.data);
        console.log(metrics);
    };
    ```
    """
    # Validate API key
    if not _validate_api_key(api_key, x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide API key via 'api_key' query parameter or 'X-API-Key' header."
        )
    
    # Use environment variable for default interval if available and user didn't specify custom interval
    env_interval = os.getenv("DEFAULT_STREAM_INTERVAL")
    if env_interval and interval == 2.0:  # Only override if using default value
        try:
            interval = float(env_interval)
            # Ensure it's within valid range
            interval = max(0.5, min(60.0, interval))
        except (ValueError, TypeError):
            pass  # Keep default interval if env var is invalid
    
    # Return SSE stream
    return StreamingResponse(
        _generate_sse_stream(interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )



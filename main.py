from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


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
        }
    
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": int(round(mem.total / (1024 * 1024))),
            "available_mb": int(round(mem.available / (1024 * 1024))),
            "used_mb": int(round(mem.used / (1024 * 1024))),
            "percent": round(mem.percent, 1),
            "cached_mb": int(round(getattr(mem, 'cached', 0) / (1024 * 1024))),
            "buffers_mb": int(round(getattr(mem, 'buffers', 0) / (1024 * 1024))),
        }
    except Exception:
        return {
            "total_mb": 0,
            "available_mb": 0,
            "used_mb": 0,
            "percent": 0.0,
            "cached_mb": 0,
            "buffers_mb": 0,
        }


def _get_disk_stats() -> Dict[str, Any]:
    """Gather disk usage statistics - returns used and available space."""
    if psutil is None:
        return {
            "used_gb": 0.0,
            "available_gb": 0.0,
        }
    
    total_used_gb = 0.0
    total_available_gb = 0.0
    try:
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

            gpus.append(
                {
                    "id": i,
                    "timestamp": _now_iso(),
                    "name": name,
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
                        "efficiency_percent": int(round(power_eff)),
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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/gpu", response_model=GPUResponseModel)
def get_gpu() -> GPUResponseModel:
    return _gather_metrics()



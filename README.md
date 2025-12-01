# NVIDIA GPU Metrics API

A FastAPI-based service that provides real-time GPU metrics using NVIDIA's NVML (NVIDIA Management Library). This service runs in a Docker container and exposes RESTful endpoints to monitor GPU performance, utilization, temperature, power consumption, and more.

## Features

- **Real-time GPU monitoring** with comprehensive metrics
- **Multi-GPU support** for systems with multiple NVIDIA GPUs
- **Power and thermal monitoring** including throttling detection
- **Memory utilization tracking** (GPU memory, not system RAM)
- **PCIe bandwidth monitoring** with utilization calculations
- **Clock frequency monitoring** (graphics and memory clocks)
- **Disk I/O activity monitoring** with read/write throughput and operations per second
- **Intelligent status detection** (active/idle based on multiple criteria)
- **Bottleneck detection** with heuristic analysis
- **RESTful API** with JSON responses
- **Server-Sent Events (SSE) streaming** for real-time metrics without polling
- **API key authentication** for secure access to streaming endpoint
- **Docker containerized** for easy deployment
- **Health check endpoint** for monitoring service availability

## Prerequisites

- **NVIDIA GPU** with compatible drivers
- **NVIDIA Container Toolkit** installed on the host
- **Docker** and **Docker Compose**
- **NVIDIA drivers** version 450.80.02 or later


## Quick Start

1. **Start the service:**
   ```bash
   docker-compose up -d
   ```

2. **Get GPU metrics:**
   ```bash
   curl http://localhost:8008/gpu
   ```

## API Endpoints


### GPU Metrics
```http
GET /gpu
```

Returns comprehensive GPU metrics for all detected GPUs.

**Response Structure:**
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "system_info": {
    "driver_version": "535.86.10",
    "cuda_version": "12.2",
    "nvml_version": "12.535.86",
    "cpu": {
      "cpu_count_physical": 8,
      "cpu_count_logical": 16,
      "cpu_percent": 25.5,
      "cpu_per_core_percent": [20.1, 30.2, 15.3, 25.4, 22.1, 28.3, 18.9, 27.6],
      "cpu_freq_current_mhz": 3200.0,
      "cpu_freq_min_mhz": 800.0,
      "cpu_freq_max_mhz": 4200.0
    },
    "memory": {
      "total_mb": 32768,
      "available_mb": 16384,
      "used_mb": 16384,
      "percent": 50.0,
      "cached_mb": 4096,
      "buffers_mb": 512,
      "top_processes": []
    },
    "disk_used_gb": 500.5,
    "disk_available_gb": 199.5,
    "disk_io": {
      "read_bytes_per_sec": 1048576.0,
      "write_bytes_per_sec": 524288.0,
      "read_ops_per_sec": 150.5,
      "write_ops_per_sec": 75.2,
      "total_read_bytes": 1099511627776,
      "total_write_bytes": 549755813888,
      "total_read_ops": 15728640,
      "total_write_ops": 7864320
    },
    "temperature_celsius": 45.0
  },
  "gpu_count": 1,
  "gpus": [
    {
      "id": 0,
      "timestamp": "2024-01-15T10:30:45Z",
      "name": "NVIDIA GeForce RTX 4090",
      "memory": {
        "total_mb": 24564.0,
        "used_mb": 2048.5,
        "free_mb": 22515.5,
        "utilization_percent": 8.3,
        "bandwidth_utilization_percent": 15.2
      },
      "utilization": {
        "gpu_percent": 45,
        "memory_percent": 12,
        "sm_percent": 45
      },
      "temperature_celsius": 58,
      "power": {
        "usage_watts": 185.2,
        "limit_watts": 450000,
        "efficiency_percent": 41
      },
      "clocks": {
        "graphics_mhz": 2520,
        "memory_mhz": 10501,
        "max_graphics_mhz": 2520,
        "max_memory_mhz": 10501,
        "efficiency_percent": 100
      },
      "pcie": {
        "generation": 4,
        "width": 16,
        "tx_throughput_mbps": 1024.5,
        "rx_throughput_mbps": 512.3,
        "total_throughput_mbps": 1536.8,
        "max_bandwidth_mbps": 31504.0,
        "utilization_percent": 5
      },
      "throttling": {
        "reasons": 0,
        "is_throttled": false,
        "bottlenecks": []
      },
      "fan_speed_rpm": 1200,
      "status": "active",
      "debug_info": {
        "gpu_util_threshold": "45.0% >= 5.0%",
        "power_threshold": "185.2W > 10.0W",
        "memory_util_threshold": "8.3% >= 5.0%",
        "clock_threshold": "2520MHz > 756MHz",
        "temp_threshold": "58.0°C > 40.0°C",
        "is_active": true
      }
    }
  ],
  "summary": {
    "total_memory_mb": 24564,
    "total_used_memory_mb": 2049,
    "memory_utilization_percent": 8.3,
    "average_gpu_utilization_percent": 45.0,
    "average_temperature_celsius": 58,
    "total_power_usage_watts": 185.2
  },
  "status": "ok"
}
```

### GPU Metrics Stream (SSE)

```http
GET /gpu/stream
```

Streams GPU metrics in real-time using Server-Sent Events (SSE). This endpoint eliminates the need for client-side polling by pushing updates to connected clients automatically.

**Query Parameters:**
- `api_key` (optional): API key for authentication (required if API keys are configured)
- `interval` (optional): Update interval in seconds (default: 2.0, min: 0.5, max: 60.0)

**Headers:**
- `X-API-Key` (optional): Alternative way to provide API key

**Response:**
- Content-Type: `text/event-stream`
- Each message follows the SSE format: `data: {json}\n\n`
- JSON structure matches the `/gpu` endpoint response

**Example Client (JavaScript):**
```javascript
// Connect to SSE stream
const apiKey = 'your-api-key-here';
const interval = 1.0; // Update every 1 second
const eventSource = new EventSource(
  `http://localhost:8008/gpu/stream?api_key=${apiKey}&interval=${interval}`
);

// Handle incoming metrics
eventSource.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  console.log('GPU Metrics:', metrics);
  // Update your UI with metrics.gpus, metrics.summary, etc.
};

// Handle errors
eventSource.onerror = (error) => {
  console.error('SSE Error:', error);
  // EventSource will automatically reconnect
};

// Close connection when done
// eventSource.close();
```

**Example Client (Python):**
```python
import requests
import json

api_key = 'your-api-key-here'
interval = 1.0
url = f'http://localhost:8008/gpu/stream?api_key={api_key}&interval={interval}'

with requests.get(url, stream=True, headers={'Accept': 'text/event-stream'}) as response:
    for line in response.iter_lines():
        if line:
            # SSE format: "data: {json}\n"
            if line.startswith(b'data: '):
                json_str = line[6:].decode('utf-8')
                metrics = json.loads(json_str)
                print('GPU Metrics:', metrics)
```

**Example Client (cURL):**
```bash
curl -N -H "X-API-Key: your-api-key-here" \
  "http://localhost:8008/gpu/stream?interval=1"
```

**Authentication:**
- If `API_KEY` or `API_KEYS` environment variables are set, authentication is required
- API key can be provided via query parameter (`?api_key=...`) or header (`X-API-Key: ...`)
- If no API keys are configured, the endpoint is accessible without authentication (backward compatible)
- Returns `401 Unauthorized` if API key is missing or invalid

**Error Handling:**
- If metric collection fails, an error event is sent but streaming continues
- Client disconnections are handled gracefully
- EventSource (JavaScript) automatically reconnects on connection loss

## Configuration

### Environment Variables

- `NVIDIA_VISIBLE_DEVICES`: Controls which GPUs are visible (default: "all")
- `NVIDIA_DRIVER_CAPABILITIES`: Required capabilities (default: "compute,utility")
- `API_KEY`: Single API key for SSE streaming endpoint authentication (optional)
- `API_KEYS`: Comma-separated list of API keys for SSE streaming endpoint authentication (optional)
- `DEFAULT_STREAM_INTERVAL`: Default update interval for SSE stream in seconds (default: 2.0)

**Note:** If neither `API_KEY` nor `API_KEYS` is set, the SSE endpoint is accessible without authentication for backward compatibility. It's recommended to set an API key in production environments.


## Status Detection

The service determines GPU status (active/idle) based on multiple criteria:

- **GPU Utilization**: ≥ 5%
- **Power Usage**: > 10W
- **Memory Utilization**: ≥ 5%
- **Graphics Clock**: > 30% of maximum
- **Temperature**: > 40°C

A GPU is marked as "active" if ANY of these conditions are met.

## Bottleneck Detection

The service can detect common GPU bottlenecks:

- **thermal_throttling**: GPU reducing performance due to heat
- **power_throttling**: GPU limited by power consumption
- **board_power_throttling**: Board-level power limiting
- **high_memory_usage**: Memory utilization > 80%
- **memory_bound_workload**: High memory usage with low GPU usage
- **pcie_bandwidth_saturated**: PCIe bandwidth > 80%

## Troubleshooting

### Service Won't Start

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Verify NVIDIA Container Toolkit:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Check Docker logs:**
   ```bash
   docker-compose logs gpu-metrics
   ```

### GPU Not Detected

- Ensure NVIDIA drivers are properly installed
- Verify `NVIDIA_VISIBLE_DEVICES=all` in docker-compose.yml
- Check that the container has GPU access

### Permission Errors

The service requires access to NVIDIA device files. Ensure the Docker daemon has proper permissions and the NVIDIA Container Toolkit is correctly installed.

### Common Errors

**"No module named pynvml"**: The container failed to install dependencies. Rebuild with:
```bash
docker-compose build --no-cache
```

**"NVML not initialized"**: NVIDIA drivers or NVML library not accessible. Check driver installation and container GPU access.

**"AttributeError: 'str' object has no attribute 'decode'"**: Fixed in recent versions. Update to the latest container.

## Development

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8008 --reload
   ```

### Building the Container

```bash
docker build -t gpu-metrics-api .
```

## Dependencies

- **FastAPI**: Web framework for the API
- **uvicorn**: ASGI server
- **pynvml**: Python bindings for NVML
- **pydantic**: Data validation and serialization


## Support

For issues and questions:
- Check the troubleshooting section above
- Review Docker logs: `docker-compose logs gpu-metrics`
- Ensure all prerequisites are met
- Verify GPU access with `nvidia-smi`

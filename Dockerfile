# GPU Metrics FastAPI service with NVIDIA DCGM and NVML
# Base on CUDA runtime (Ubuntu 22.04) so DCGM apt repo is available
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# OS deps and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ca-certificates \
      curl \
      gnupg \
      pciutils && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA DCGM from the CUDA repository (Ubuntu 22.04)
# Note: DCGM requires NVIDIA drivers on the host and NVIDIA Container Toolkit at runtime.
RUN set -eux; \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg -o /usr/share/keyrings/cuda-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends datacenter-gpu-manager; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# App
COPY main.py /app/main.py

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8008/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]



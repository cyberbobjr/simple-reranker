FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TORCHVISION=1       

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git git-lfs build-essential ca-certificates \
    libglib2.0-0 libgl1 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8 (nightly pour RTX 5090 / SM_120)
RUN pip uninstall -y torch torchvision torchaudio triton || true \
 && pip install --pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu128 \
      torch

# Dépendances app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# App
COPY rerank_service.py rerank_service.py
COPY rerank_config.yaml rerank_config.yaml

EXPOSE 8000
CMD ["python", "rerank_service.py", "--config", "rerank_config.yaml", "--serve"]

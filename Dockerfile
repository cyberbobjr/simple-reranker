FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Configurer les variables d'environnement pour éviter les interactions
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Installation de Python 3.12 et des dépendances système
RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Configurer Python 3.12 comme version par défaut
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Installer pip pour Python 3.12
RUN wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    && pip install --upgrade pip setuptools wheel
# PyTorch CUDA (adapte la version CUDA si besoin)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch
# Libs modèles
RUN pip install "transformers>=4.44" "huggingface-hub>=0.24" "accelerate>=0.33" fastapi uvicorn[standard] pyyaml setuptools wheel hf_transfer
# sentence-transformers requis pour le service complet
RUN pip install sentence-transformers
# flash-attn (optionnel). Nécessite NVCC pour la compilation.
RUN pip install flash-attn --no-build-isolation || echo "flash-attn installation failed, continuing without it"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TOKENIZERS_PARALLELISM=false
ENV TORCH_DTYPE=bfloat16

WORKDIR /app
COPY . /app
EXPOSE 8000

CMD ["python", "rerank_service.py", "--config", "rerank_config.yaml", "--serve"]
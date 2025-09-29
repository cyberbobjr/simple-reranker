"""System routes for the reranker service."""

import os
import sys
import logging
from fastapi import APIRouter, Depends
from core.config import Config
from version import get_version


def create_system_routes(config: Config, require_api_key, loaded_models: dict, warmup_models_func) -> APIRouter:
    """Create system routes with dependency injection."""
    router = APIRouter()

    @router.get("/healthz")
    def healthz():
        """Health check endpoint."""
        return {"status": "ok"}

    @router.get("/v1/models")
    def v1_models(_auth_ok: bool = Depends(require_api_key)):
        """Get model configuration and loaded models."""
        m = config.get("model", {}); hf = config.get("huggingface", {})
        return {
            "mode": m.get("mode", "cross"),
            "cross_name": m.get("cross_name"),
            "bi_name": m.get("bi_name"),
            "embedding_name": m.get("embedding_name"),
            "batch_size_cross": m.get("batch_size_cross", 32),
            "batch_size_bi": m.get("batch_size_bi", 64),
            "normalize_embeddings": bool(m.get("normalize_embeddings", True)),
            "trust_remote_code": bool(m.get("trust_remote_code", True)),
            "huggingface": {"cache_dir": hf.get("cache_dir"), "model_dir": hf.get("model_dir"), "prefetch": hf.get("prefetch") or []},
            "loaded_models": list(loaded_models.keys())
        }

    @router.post("/v1/models/reload")
    def v1_models_reload(_auth_ok: bool = Depends(require_api_key)):
        """Reload all models."""
        import gc
        n_before = len(loaded_models)
        loaded_models.clear()
        gc.collect()
        config.ensure_hf_auth_and_cache()
        warmup_models_func(config)
        logging.getLogger("rerank").info("models_reloaded", extra={"extra":{"evicted": n_before}})
        return {"evicted": n_before, "loaded_after": list(loaded_models.keys())}

    @router.get("/v1/config")
    def v1_config(_auth_ok: bool = Depends(require_api_key)):
        """Get service configuration."""
        m = config.get("model", {}); hf = config.get("huggingface", {}); s = config.get("server", {})
        return {
            "version": get_version(),
            "model": {
                "mode": m.get("mode", "cross"),
                "cross_name": m.get("cross_name"),
                "bi_name": m.get("bi_name"),
                "embedding_name": m.get("embedding_name"),
                "batch_size_cross": m.get("batch_size_cross", 32),
                "batch_size_bi": m.get("batch_size_bi", 64),
                "normalize_embeddings": bool(m.get("normalize_embeddings", True)),
                "trust_remote_code": bool(m.get("trust_remote_code", True)),
            },
            "huggingface": {
                "cache_dir": hf.get("cache_dir"),
                "model_dir": hf.get("model_dir"),
                "prefetch": hf.get("prefetch") or [],
                "token": "REDACTED" if (hf.get("token") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) else None
            },
            "server": {
                "host": s.get("host", "0.0.0.0"),
                "port": int(s.get("port", 8000)),
                "cors_origins": s.get("cors_origins", ["*"]),
                "api_keys_configured": bool(s.get("api_key") or s.get("api_keys")) or bool(os.environ.get("RERANK_API_KEYS"))
            }
        }

    @router.get("/v1/diagnostics")
    def v1_diagnostics(_auth_ok: bool = Depends(require_api_key)):
        """Get system diagnostics."""
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        hf = config.get("huggingface", {})
        return {
            "version": get_version(),
            "python": sys.version,
            "torch_cuda_available": cuda_ok,
            "huggingface": {
                "cache_dir": hf.get("cache_dir"),
                "model_dir": hf.get("model_dir"),
                "prefetch": hf.get("prefetch") or []
            },
            "models_loaded": list(loaded_models.keys())
        }

    return router
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reranking & Embedding Service (FastAPI, Cohere-compatible)
- YAML config with explicit huggingface.model_dir
- HF auth + prefetch
- API key auth
- Logging (json/text) + request_id middleware
- Endpoints: /healthz, /v1/rerank, /v1/encode, /v1/models, /v1/models/reload, /v1/config, /v1/diagnostics
- CLI rerank
"""

import os, sys, argparse, uuid, yaml, json, time, logging
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Depends, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download, login as hf_login
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder

import uuid as _uuid

_GLOBAL_CONFIG: Dict[str, Any] = {}

DEFAULT_CONFIG = {
    "model": {
        "mode": "cross",
        "cross_name": "BAAI/bge-reranker-v2-m3",
        "bi_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_name": "intfloat/multilingual-e5-base",
        "batch_size_cross": 32,
        "batch_size_bi": 64,
        "normalize_embeddings": True,
        "trust_remote_code": True
    },
    "huggingface": {
        "token": None,
        "cache_dir": None,
        "model_dir": "/app/models",
        "prefetch": [
            "BAAI/bge-reranker-v2-m3",
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/multilingual-e5-base"
        ]
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        # "api_keys": ["change-me-123"]
        # "api_key": "change-me-123"
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "file": None
    }
}

def _setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt_mode = str(log_cfg.get("format", "json")).lower()
    file_path = log_cfg.get("file")
    logger = logging.getLogger("rerank")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler = logging.FileHandler(file_path) if file_path else logging.StreamHandler()
    if fmt_mode == "json":
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                data = {
                    "ts": int(time.time()*1000),
                    "level": record.levelname,
                    "msg": record.getMessage(),
                    "name": record.name,
                }
                if hasattr(record, "request_id"):
                    data["request_id"] = getattr(record, "request_id", None)
                extra = record.__dict__.get("extra")
                if extra:
                    data.update(extra)
                return json.dumps(data, ensure_ascii=False)
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(cfg, user_cfg)
    return cfg

def ensure_hf_auth_and_cache(cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger("rerank")
    hf_cfg = cfg.get("huggingface", {})
    token = hf_cfg.get("token") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache_dir = hf_cfg.get("cache_dir")
    model_dir = hf_cfg.get("model_dir")

    token_source = None
    if hf_cfg.get("token"):
        token_source = "config"
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token_source = "env"

    summary: Dict[str, Any] = {
        "has_token": bool(token),
        "token_source": token_source,
        "cache_dir": cache_dir,
        "cache_dir_resolved": os.path.expanduser(cache_dir) if cache_dir else None,
        "model_dir": model_dir,
        "model_dir_resolved": None,
        "model_dir_ready": False,
        "hf_home": os.environ.get("HF_HOME"),
        "prefetch_ok": [],
        "prefetch_failed": []
    }

    logger.info("boot_hf_config", extra={"extra": {
        "has_token": summary["has_token"],
        "token_source": token_source,
        "cache_dir": cache_dir,
        "model_dir": model_dir,
        "prefetch": hf_cfg.get("prefetch") or []
    }})

    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        try:
            hf_login(token=token, add_to_git_credential=False)
            logger.info("boot_hf_login_ok")
        except Exception as e:
            logger.warning("boot_hf_login_failed", extra={"extra":{"error": str(e)}})
    if cache_dir:
        os.environ["HF_HOME"] = os.path.expanduser(cache_dir)
        summary["hf_home"] = os.environ.get("HF_HOME")
        logger.info("boot_hf_home_set", extra={"extra":{"HF_HOME": summary["hf_home"]}})
    elif model_dir:
        # Si model_dir est configuré mais pas cache_dir, utiliser model_dir comme HF_HOME
        os.environ["HF_HOME"] = os.path.expanduser(str(model_dir))
        summary["hf_home"] = os.environ.get("HF_HOME")
        logger.info("boot_hf_home_set_from_model_dir", extra={"extra":{"HF_HOME": summary["hf_home"]}})

    local_dir = None
    if model_dir:
        md = os.path.expanduser(str(model_dir))
        try:
            os.makedirs(md, exist_ok=True)
            test_path = os.path.join(md, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            logger.info("boot_model_dir_ready", extra={"extra":{"model_dir": md}})
            local_dir = md
            summary["model_dir_resolved"] = md
            summary["model_dir_ready"] = True
        except Exception as e:
            logger.error("boot_model_dir_not_writable", extra={"extra":{"model_dir": md, "error": str(e)}})
            summary["model_dir_resolved"] = md
            summary["model_dir_ready"] = False

    downloaded = []
    for repo in hf_cfg.get("prefetch") or []:
        try:
            # Utiliser le système de cache standard de HuggingFace
            # HF_HOME a été configuré plus haut, donc HF utilisera automatiquement le bon répertoire
            path = snapshot_download(
                repo_id=repo, 
                token=token
                # Pas besoin de spécifier cache_dir car HF_HOME est configuré
            )
            downloaded.append((repo, path))
        except Exception as e:
            logger.warning("boot_prefetch_failed", extra={"extra":{"repo": repo, "error": str(e)}})
            summary["prefetch_failed"].append({"repo": repo, "error": str(e)})
    if downloaded:
        logger.info("boot_prefetched", extra={"extra":{"count": len(downloaded)}})
        for repo, path in downloaded:
            logger.info("boot_prefetch_item", extra={"extra":{"repo": repo, "path": path}})
            summary["prefetch_ok"].append({"repo": repo, "path": path})
    return summary

def warmup_models(config: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger("rerank")
    m = config.get("model", {})
    s = config.get("server", {})
    w = (s.get("warmup") or {})
    summary: Dict[str, Any] = {
        "enabled": bool(w.get("enabled", False)),
        "texts": [],
        "stages": [],
        "errors": [],
        "loaded_models": list(_LOADED_MODELS.keys())
    }
    if not summary["enabled"]:
        return summary

    texts = w.get("texts") or ["warmup", "échauffement"]
    trust = bool(m.get("trust_remote_code", True))
    bs_cross = int(m.get("batch_size_cross", 32))
    bs_bi = int(m.get("batch_size_bi", 64))
    norm = bool(m.get("normalize_embeddings", True))
    summary["texts"] = texts
    errors: List[Dict[str, Any]] = []

    def record_stage(stage: str, status: str, model: Optional[str], reason: Optional[str] = None, error: Optional[str] = None) -> None:
        entry = {"stage": stage, "status": status, "model": model}
        if reason:
            entry["reason"] = reason
        if error:
            entry["error"] = error
            errors.append({"stage": stage, "error": error})
        summary["stages"].append(entry)

    def warm_embedding() -> None:
        model_name = m.get("embedding_name") or m.get("bi_name")
        if not model_name:
            record_stage("embedding", "skipped", None, reason="no model configured")
            return
        if not w.get("embedding", True):
            record_stage("embedding", "skipped", model_name, reason="disabled")
            return
        try:
            enc = get_embedding_encoder(model_name, trust_remote_code=trust)
            _ = enc.encode(texts, batch_size=bs_bi, convert_to_numpy=True, normalize_embeddings=norm, show_progress_bar=False)
            logger.info("warmup_embedding_ok", extra={"extra":{"model": model_name}})
            record_stage("embedding", "ok", model_name)
        except Exception as e:
            msg = str(e)
            logger.warning("warmup_embedding_failed", extra={"extra":{"model": model_name, "error": msg}})
            record_stage("embedding", "error", model_name, error=msg)

    def warm_cross() -> None:
        model_name = m.get("cross_name")
        if not w.get("cross", True):
            record_stage("cross", "skipped", model_name, reason="disabled")
            return
        if not model_name:
            record_stage("cross", "skipped", None, reason="no model configured")
            return
        try:
            ce = get_cross_encoder(model_name, trust_remote_code=trust)
            pairs = [(texts[0], t) for t in texts]
            _ = ce.predict(pairs, batch_size=min(len(pairs), bs_cross), convert_to_numpy=True, show_progress_bar=False)
            logger.info("warmup_cross_ok", extra={"extra":{"model": model_name}})
            record_stage("cross", "ok", model_name)
        except Exception as e:
            msg = str(e)
            logger.warning("warmup_cross_failed", extra={"extra":{"model": model_name, "error": msg}})
            record_stage("cross", "error", model_name, error=msg)

    def warm_bi() -> None:
        model_name = m.get("bi_name")
        if not w.get("bi", False):
            record_stage("bi", "skipped", model_name, reason="disabled")
            return
        if not model_name:
            record_stage("bi", "skipped", None, reason="no model configured")
            return
        try:
            bi = get_bi_encoder(model_name, trust_remote_code=trust)
            _ = bi.encode(texts, batch_size=bs_bi, convert_to_numpy=True, normalize_embeddings=norm, show_progress_bar=False)
            logger.info("warmup_bi_ok", extra={"extra":{"model": model_name}})
            record_stage("bi", "ok", model_name)
        except Exception as e:
            msg = str(e)
            logger.warning("warmup_bi_failed", extra={"extra":{"model": model_name, "error": msg}})
            record_stage("bi", "error", model_name, error=msg)

    warm_embedding()
    warm_cross()
    warm_bi()

    if errors:
        summary["errors"] = errors
        logger.warning("warmup_failed", extra={"extra":{"errors": errors}})

    summary["loaded_models"] = list(_LOADED_MODELS.keys())
    return summary


def print_boot_banner(config: Dict[str, Any], hf_summary: Dict[str, Any], warmup_summary: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """Affiche un banner élégant avec les informations de configuration au démarrage."""
    import datetime
    import platform
    
    width = 82
    
    def pad(label: str, value: Any, color: str = "\033[0m") -> str:
        """Format une ligne avec label et valeur, avec couleur optionnelle."""
        return f"  {color}{label:<28}\033[0m: {value}"
    
    def section(title: str, icon: str = "") -> str:
        """Crée un séparateur de section avec icône."""
        bar = "─" * width
        title_with_icon = f"{icon} {title}" if icon else title
        return f"\n\033[1;36m{bar}\033[0m\n\033[1;37m{title_with_icon.upper()}\033[0m\n\033[1;36m{bar}\033[0m"
    
    def fmt_bool(value: Optional[bool], true_color: str = "\033[1;32m", false_color: str = "\033[1;31m") -> str:
        """Format booléen avec couleurs."""
        if value is None:
            return "\033[33munknown\033[0m"
        return f"{true_color}✓ yes\033[0m" if value else f"{false_color}✗ no\033[0m"
    
    def fmt_value(value: Any, empty: str = "\033[90m-\033[0m", max_length: int = 50) -> str:
        """Format une valeur avec gestion de la longueur."""
        if value is None:
            return empty
        if isinstance(value, bool):
            return fmt_bool(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return empty
            result = ", ".join(str(v) for v in value)
            if len(result) > max_length:
                return result[:max_length-3] + "..."
            return result
        result = str(value)
        if len(result) > max_length:
            return result[:max_length-3] + "..."
        return result
    
    def fmt_size(size_bytes: float) -> str:
        """Format la taille en bytes de façon lisible."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    lines: List[str] = []
    
    # Header élégant
    header = "═" * width
    lines.append(f"\033[1;35m{header}\033[0m")
    lines.append(f"\033[1;37m{'🚀 RERANK SERVICE BOOT SUMMARY':^{width}}\033[0m")
    lines.append(f"\033[1;35m{header}\033[0m")
    
    # Informations générales
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(pad("Boot time", f"\033[1;33m{now}\033[0m"))
    lines.append(pad("Platform", f"\033[1;34m{platform.system()} {platform.release()}\033[0m"))
    lines.append(pad("Python version", f"\033[1;34m{sys.version.split()[0]}\033[0m"))
    
    if config_path is not None:
        color = "\033[1;32m" if "missing" not in config_path else "\033[1;31m"
        lines.append(pad("Config file", f"{color}{config_path}\033[0m"))

    # Configuration des modèles
    model_cfg = config.get("model", {})
    lines.append(section("MODEL CONFIGURATION", "🤖"))
    lines.append(pad("Active mode", f"\033[1;36m{fmt_value(model_cfg.get('mode', 'cross'))}\033[0m"))
    lines.append(pad("Cross encoder", f"\033[1;32m{fmt_value(model_cfg.get('cross_name'))}\033[0m"))
    lines.append(pad("Bi encoder", f"\033[1;32m{fmt_value(model_cfg.get('bi_name'))}\033[0m"))
    lines.append(pad("Embedding model", f"\033[1;32m{fmt_value(model_cfg.get('embedding_name'))}\033[0m"))
    lines.append(pad("Batch size (cross)", f"\033[1;33m{model_cfg.get('batch_size_cross', 32)}\033[0m"))
    lines.append(pad("Batch size (bi)", f"\033[1;33m{model_cfg.get('batch_size_bi', 64)}\033[0m"))
    lines.append(pad("Normalize embeddings", fmt_bool(model_cfg.get("normalize_embeddings", True))))
    lines.append(pad("Trust remote code", fmt_bool(model_cfg.get("trust_remote_code", True))))

    # Configuration HuggingFace
    lines.append(section("HUGGINGFACE CONFIGURATION", "🤗"))
    lines.append(pad("Authentication token", fmt_bool(hf_summary.get("has_token"))))
    
    token_source = hf_summary.get("token_source")
    if token_source:
        source_color = "\033[1;32m" if token_source == "config" else "\033[1;33m"
        lines.append(pad("Token source", f"{source_color}{token_source}\033[0m"))
    
    cache_dir = hf_summary.get("cache_dir_resolved") or hf_summary.get("cache_dir")
    if cache_dir:
        lines.append(pad("Cache directory", f"\033[1;34m{fmt_value(cache_dir)}\033[0m"))
        
    model_dir = hf_summary.get("model_dir_resolved") or hf_summary.get("model_dir")
    if model_dir:
        ready_status = fmt_bool(hf_summary.get("model_dir_ready"))
        lines.append(pad("Model directory", f"\033[1;34m{fmt_value(model_dir)}\033[0m ({ready_status})"))
        
    hf_home = hf_summary.get("hf_home")
    if hf_home:
        lines.append(pad("HF_HOME", f"\033[1;34m{fmt_value(hf_home)}\033[0m"))

    # Préchargement des modèles
    prefetch_ok = hf_summary.get("prefetch_ok") or []
    prefetch_failed = hf_summary.get("prefetch_failed") or []
    
    if prefetch_ok or prefetch_failed:
        lines.append(section("MODEL PREFETCHING", "⬇️"))
        
        if prefetch_ok:
            lines.append(pad("Successfully prefetched", f"\033[1;32m{len(prefetch_ok)} model(s)\033[0m"))
            for item in prefetch_ok:
                repo = item.get("repo", "unknown")
                path = item.get("path", "unknown")
                lines.append(f"    \033[1;32m✓\033[0m \033[1;37m{repo}\033[0m")
                lines.append(f"      → \033[90m{path}\033[0m")
        
        if prefetch_failed:
            lines.append(pad("Failed to prefetch", f"\033[1;31m{len(prefetch_failed)} model(s)\033[0m"))
            for item in prefetch_failed:
                repo = item.get("repo", "unknown")
                error = item.get("error", "unknown error")
                lines.append(f"    \033[1;31m✗\033[0m \033[1;37m{repo}\033[0m")
                lines.append(f"      → \033[1;31m{error[:60]}{'...' if len(error) > 60 else ''}\033[0m")

    # Configuration du serveur
    server_cfg = config.get("server", {})
    lines.append(section("SERVER CONFIGURATION", "🌐"))
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    lines.append(pad("Listen address", f"\033[1;36m{host}:{port}\033[0m"))
    
    cors_origins = server_cfg.get("cors_origins", ["*"])
    lines.append(pad("CORS origins", f"\033[1;33m{fmt_value(cors_origins)}\033[0m"))
    
    # Vérification des clés API
    has_api_key = bool(server_cfg.get("api_key") or server_cfg.get("api_keys") or os.environ.get("RERANK_API_KEYS"))
    lines.append(pad("API key protection", fmt_bool(has_api_key)))

    # Warmup et modèles chargés
    lines.append(section("MODEL WARMUP & LOADING", "🔥"))
    lines.append(pad("Warmup enabled", fmt_bool(warmup_summary.get("enabled"))))
    
    texts = warmup_summary.get("texts") or []
    if texts:
        lines.append(pad("Warmup texts", f"\033[1;33m{len(texts)} sample(s)\033[0m"))
    
    stages = warmup_summary.get("stages") or []
    if stages:
        lines.append(pad("Warmup stages", f"\033[1;36m{len(stages)} processed\033[0m"))
        for stage in stages:
            stage_name = stage.get("stage", "unknown")
            status = stage.get("status", "unknown")
            model = stage.get("model") or "no model"
            extra = stage.get("reason") or stage.get("error")
            
            if status == "ok":
                status_icon = "\033[1;32m✓\033[0m"
                status_text = "\033[1;32mOK\033[0m"
            elif status == "error":
                status_icon = "\033[1;31m✗\033[0m"
                status_text = "\033[1;31mERROR\033[0m"
            elif status == "skipped":
                status_icon = "\033[1;33m◦\033[0m"
                status_text = "\033[1;33mSKIPPED\033[0m"
            else:
                status_icon = "\033[90m?\033[0m"
                status_text = f"\033[90m{status.upper()}\033[0m"
            
            lines.append(f"    {status_icon} \033[1;37m{stage_name}\033[0m: {status_text}")
            lines.append(f"      → Model: \033[36m{model}\033[0m")
            if extra:
                lines.append(f"      → \033[90m{extra}\033[0m")

    loaded_models = warmup_summary.get("loaded_models") or []
    if loaded_models:
        lines.append(pad("Models in memory", f"\033[1;32m{len(loaded_models)} active\033[0m"))
        for model_key in loaded_models:
            model_display = model_key.replace("::", " → ")
            lines.append(f"    \033[1;32m●\033[0m \033[1;37m{model_display}\033[0m")

    # Logging configuration
    log_cfg = config.get("logging", {})
    if log_cfg:
        lines.append(section("LOGGING CONFIGURATION", "📝"))
        lines.append(pad("Log level", f"\033[1;36m{log_cfg.get('level', 'INFO')}\033[0m"))
        lines.append(pad("Log format", f"\033[1;36m{log_cfg.get('format', 'json')}\033[0m"))
        log_file = log_cfg.get("file")
        if log_file:
            lines.append(pad("Log file", f"\033[1;34m{log_file}\033[0m"))

    # Footer
    lines.append(f"\033[1;35m{header}\033[0m")
    lines.append(f"\033[1;32m{'🎉 SERVICE READY TO SERVE REQUESTS':^{width}}\033[0m")
    lines.append(f"\033[1;35m{header}\033[0m")
    
    print("\n".join(lines))

_LOADED_MODELS: Dict[str, Any] = {}

def get_cross_encoder(name: str, trust_remote_code: bool = True) -> CrossEncoder:
    key = f"cross::{name}"
    if key not in _LOADED_MODELS:
        # Utiliser cache_folder pour diriger HuggingFace vers notre répertoire de modèles
        cache_folder = _GLOBAL_CONFIG.get("huggingface", {}).get("model_dir")
        _LOADED_MODELS[key] = CrossEncoder(
            name,
            trust_remote_code=trust_remote_code,
            cache_folder=cache_folder
        )
    return _LOADED_MODELS[key]

def get_bi_encoder(name: str, trust_remote_code: bool = True) -> SentenceTransformer:
    key = f"bi::{name}"
    if key not in _LOADED_MODELS:
        # Utiliser cache_folder pour diriger HuggingFace vers notre répertoire de modèles
        cache_folder = _GLOBAL_CONFIG.get("huggingface", {}).get("model_dir")
        _LOADED_MODELS[key] = SentenceTransformer(
            name,
            trust_remote_code=trust_remote_code,
            cache_folder=cache_folder
        )
    return _LOADED_MODELS[key]

def get_embedding_encoder(name: str, trust_remote_code: bool = True) -> SentenceTransformer:
    key = f"embedding::{name}"
    if key not in _LOADED_MODELS:
        # Utiliser cache_folder pour diriger HuggingFace vers notre répertoire de modèles
        cache_folder = _GLOBAL_CONFIG.get("huggingface", {}).get("model_dir")
        _LOADED_MODELS[key] = SentenceTransformer(
            name,
            trust_remote_code=trust_remote_code,
            cache_folder=cache_folder
        )
    return _LOADED_MODELS[key]

def rerank_cross(query: str, docs: List[str], model_name: str, batch_size: int, trust_remote_code: bool = True) -> List[Tuple[int, float]]:
    ce = get_cross_encoder(model_name, trust_remote_code=trust_remote_code)
    scores = ce.predict([(query, d) for d in docs], batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return sorted(list(enumerate(scores.tolist())), key=lambda x: x[1], reverse=True)

def rerank_bi(query: str, docs: List[str], model_name: str, batch_size: int, normalize: bool = True, trust_remote_code: bool = True) -> List[Tuple[int, float]]:
    bi = get_bi_encoder(model_name, trust_remote_code=trust_remote_code)
    q = bi.encode(query, batch_size=1, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False)
    D = bi.encode(docs, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False)
    scores = util.cos_sim(q, D).squeeze(0)
    return sorted(list(enumerate([float(s) for s in scores])), key=lambda x: x[1], reverse=True)

def cohere_like_response(query: str, docs: List[str], ranked: List[Tuple[int, float]], top_n: int, return_documents: bool, model_used: str) -> Dict[str, Any]:
    top = ranked[:top_n] if top_n > 0 else ranked
    results = []
    for idx, score in top:
        item = {"index": idx, "relevance_score": float(score)}
        if return_documents:
            item["document"] = docs[idx]
        results.append(item)
    return {"id": f"rerank-{uuid.uuid4()}", "model": model_used, "results": results}

class EncodeRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    input: List[Union[str, Dict[str, Any]]] = Field(...)
    normalize: Optional[bool] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)

class RerankRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    query: str = Field(...)
    documents: List[Union[str, Dict[str, Any]]] = Field(...)
    top_n: int = Field(default=10)
    return_documents: bool = Field(default=False)

def extract_texts(items: List[Union[str, Dict[str, Any]]]) -> List[str]:
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            for key in ("text", "document", "passage", "content"):
                if key in it and isinstance(it[key], str):
                    out.append(it[key]); break
            else:
                out.append(json.dumps(it, ensure_ascii=False))
        else:
            out.append(str(it))
    return out

def _load_allowed_keys(config) -> set[str]:
    allowed = set()
    srv = config.get("server", {})
    if isinstance(srv.get("api_keys"), list):
        allowed.update([str(x).strip() for x in srv.get("api_keys") if str(x).strip()])
    if srv.get("api_key"):
        allowed.add(str(srv.get("api_key")).strip())
    env_keys = os.environ.get("RERANK_API_KEYS")
    if env_keys:
        for k in env_keys.split(","):
            k = k.strip()
            if k: allowed.add(k)
    return allowed

def make_api_key_dependency(config):
    allowed_keys = _load_allowed_keys(config)
    async def require_api_key(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ):
        if not allowed_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
        provided = None
        if x_api_key and x_api_key.strip():
            provided = x_api_key.strip()
        elif authorization and authorization.strip():
            import re
            m = re.match(r"(?i)Bearer\s+(.+)", authorization.strip())
            if m: provided = m.group(1).strip()
        if not provided or provided not in allowed_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
        request.state.api_key_id = provided
        return True
    return require_api_key

def build_app(config: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Rerank & Embedding Service (Cohere-compatible)")
    require_api_key = make_api_key_dependency(config)
    cors = config.get("server", {}).get("cors_origins", ["*"])
    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=cors, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/healthz")
    def healthz(): return {"status": "ok"}

    @app.middleware("http")
    async def _log_requests(request, call_next):
        logger = logging.getLogger("rerank")
        rid = request.headers.get("x-request-id") or str(_uuid.uuid4())
        start = time.time()
        try:
            response = await call_next(request)
            duration_ms = int((time.time()-start)*1000)
            logger.info("http_request", extra={"request_id": rid, "extra": {
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": duration_ms
            }})
            response.headers["x-request-id"] = rid
            return response
        except Exception as e:
            duration_ms = int((time.time()-start)*1000)
            logger.error("http_exception", extra={"request_id": rid, "extra": {
                "path": request.url.path,
                "method": request.method,
                "error": str(e),
                "duration_ms": duration_ms
            }})
            raise

    @app.post("/v1/rerank")
    def v1_rerank(req: RerankRequest, _auth_ok: bool = Depends(require_api_key)):
        mcfg = config.get("model", {})
        mode = mcfg.get("mode", "cross")
        chosen = req.model or (mcfg.get("cross_name") if mode == "cross" else mcfg.get("bi_name"))
        trc = bool(mcfg.get("trust_remote_code", True))
        bs_cross = int(mcfg.get("batch_size_cross", 32))
        bs_bi = int(mcfg.get("batch_size_bi", 64))
        norm = bool(mcfg.get("normalize_embeddings", True))
        docs = extract_texts(req.documents)
        ranked = rerank_cross(req.query, docs, chosen, bs_cross, trc) if mode=="cross" else rerank_bi(req.query, docs, chosen, bs_bi, norm, trc)
        return cohere_like_response(req.query, docs, ranked, req.top_n, req.return_documents, chosen)

    @app.post("/v1/encode")
    def v1_encode(req: EncodeRequest, _auth_ok: bool = Depends(require_api_key)):
        mcfg = config.get("model", {})
        trc = bool(mcfg.get("trust_remote_code", True))
        chosen = req.model or mcfg.get("embedding_name") or mcfg.get("bi_name")
        if not chosen:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No embedding model configured")
        norm_default = bool(mcfg.get("normalize_embeddings", True))
        norm = norm_default if req.normalize is None else bool(req.normalize)
        bs = int(req.batch_size) if req.batch_size else int(mcfg.get("batch_size_bi", 64))
        texts = extract_texts(req.input)
        enc = get_embedding_encoder(chosen, trust_remote_code=trc)
        embs = enc.encode(texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=norm, show_progress_bar=False)
        return {"model": chosen, "dimension": int(embs.shape[1]) if hasattr(embs, "shape") else (len(embs[0]) if embs else 0), "embeddings": [e.tolist() for e in embs]}

    @app.get("/v1/models")
    def v1_models(_auth_ok: bool = Depends(require_api_key)):
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
            "loaded_models": list(_LOADED_MODELS.keys())
        }

    @app.post("/v1/models/reload")
    def v1_models_reload(_auth_ok: bool = Depends(require_api_key)):
        import gc
        n_before = len(_LOADED_MODELS)
        _LOADED_MODELS.clear()
        gc.collect()
        ensure_hf_auth_and_cache(config)
        warmup_models(config)   # 👈 charge en mémoire + premier forward
        logging.getLogger("rerank").info("models_reloaded", extra={"extra":{"evicted": n_before}})
        return {"evicted": n_before, "loaded_after": list(_LOADED_MODELS.keys())}

    @app.get("/v1/config")
    def v1_config(_auth_ok: bool = Depends(require_api_key)):
        m = config.get("model", {}); hf = config.get("huggingface", {}); s = config.get("server", {})
        return {
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

    @app.get("/v1/diagnostics")
    def v1_diagnostics(_auth_ok: bool = Depends(require_api_key)):
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        hf = config.get("huggingface", {})
        return {
            "python": sys.version,
            "torch_cuda_available": cuda_ok,
            "huggingface": {
                "cache_dir": hf.get("cache_dir"),
                "model_dir": hf.get("model_dir"),
                "prefetch": hf.get("prefetch") or []
            },
            "models_loaded": list(_LOADED_MODELS.keys())
        }

    return app

def read_candidates_from_path(path: Optional[str]) -> List[str]:
    def extract(obj: Any) -> str:
        if isinstance(obj, str): return obj.strip()
        if isinstance(obj, dict):
            for k in ("text","document","passage","content"):
                if k in obj and isinstance(obj[k], str): return obj[k].strip()
            return json.dumps(obj, ensure_ascii=False)
        return str(obj)
    if not path or path == "-": return [line.strip() for line in sys.stdin if line.strip()]
    import pathlib
    p = pathlib.Path(path).expanduser()
    if not p.exists(): raise FileNotFoundError(path)
    txt = p.read_text(encoding="utf-8"); s = txt.lstrip()
    if s.startswith("{") or s.startswith("["):
        try:
            data = json.loads(txt)
            if isinstance(data, dict) and "documents" in data: return [extract(x) for x in data["documents"]]
            if isinstance(data, list): return [extract(x) for x in data]
            return [extract(data)]
        except json.JSONDecodeError:
            docs = []
            for line in txt.splitlines():
                line=line.strip()
                if not line: continue
                try: docs.append(extract(json.loads(line)))
                except json.JSONDecodeError: docs.append(line)
            return docs
    else:
        return [line.strip() for line in txt.splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Rerank & Embedding Service (Cohere-compatible).")
    parser.add_argument("--config","-c",default=None); parser.add_argument("--serve",action="store_true")
    parser.add_argument("--query","-q",default=None); parser.add_argument("--candidates",default="-")
    parser.add_argument("--mode",choices=["cross","bi"],default=None); parser.add_argument("--model",default=None)
    parser.add_argument("--top-k",type=int,default=10); parser.add_argument("--with-text",action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    _setup_logging(config)
    global _GLOBAL_CONFIG; _GLOBAL_CONFIG = config
    hf_summary = ensure_hf_auth_and_cache(config)
    warmup_summary = warmup_models(config)   # 👈 charge en mémoire + premier forward
    if args.config:
        config_display = os.path.abspath(args.config)
        if not os.path.exists(args.config):
            config_display += " (missing, defaults used)"
    else:
        config_display = "<embedded defaults>"
    print_boot_banner(config, hf_summary, warmup_summary, config_display)
    if args.serve:
        import uvicorn
        uvicorn.run(build_app(config), host=config.get("server",{}).get("host","0.0.0.0"), port=int(config.get("server",{}).get("port",8000)))
        return
    if not args.query:
        print("Missing --query for CLI mode. Use --serve to start the API.", file=sys.stderr); sys.exit(2)
    docs = read_candidates_from_path(args.candidates)
    if not docs: print("No candidate documents found.", file=sys.stderr); sys.exit(1)
    m = config.get("model", {}); mode = args.mode or m.get("mode","cross")
    trc = bool(m.get("trust_remote_code", True))
    bs_cross = int(m.get("batch_size_cross",32)); bs_bi = int(m.get("batch_size_bi",64))
    norm = bool(m.get("normalize_embeddings", True))
    chosen = args.model or (m.get("cross_name") if mode=="cross" else m.get("bi_name"))
    ranked = rerank_cross(args.query, docs, chosen, bs_cross, trc) if mode=="cross" else rerank_bi(args.query, docs, chosen, bs_bi, norm, trc)
    print(json.dumps(cohere_like_response(args.query, docs, ranked, args.top_k, args.with_text, chosen), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

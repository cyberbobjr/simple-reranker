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

import os, sys, argparse, json, time, logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Depends, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

# Import our core, services and routes
from core.config import Config
from core.security import BruteForceProtector
from services.embedding_service import EmbeddingService
from services.rerank_service import RerankService
from routes.embedding import create_embedding_routes
from routes.rerank import create_rerank_routes
from routes.system import create_system_routes
from version import get_version

import uuid as _uuid

_LOADED_MODELS: Dict[str, Any] = {}


def _setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt_mode = str(log_cfg.get("format", "json")).lower()
    file_path = log_cfg.get("file")
    logger = logging.getLogger("rerank")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    if file_path:
        import pathlib
        log_dir = pathlib.Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file_path)
    else:
        handler = logging.StreamHandler()
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


def warmup_models(config: Config) -> Dict[str, Any]:
    """Warmup models based on configuration."""
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

    texts = w.get("texts") or ["warmup", "test"]
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

    # Create services for warmup
    embedding_service = EmbeddingService(config.data, _LOADED_MODELS)
    rerank_service = RerankService(config.data, _LOADED_MODELS)

    def warm_embedding() -> None:
        model_name = m.get("embedding_name") or m.get("bi_name")
        if not model_name:
            record_stage("embedding", "skipped", None, reason="no model configured")
            return
        if not w.get("embedding", True):
            record_stage("embedding", "skipped", model_name, reason="disabled")
            return
        try:
            success = embedding_service.warmup_embedding_model(model_name, texts)
            if success:
                logger.info("warmup_embedding_ok", extra={"extra":{"model": model_name}})
                record_stage("embedding", "ok", model_name)
            else:
                raise Exception("Warmup failed")
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
            success = rerank_service.warmup_cross_model(model_name, texts)
            if success:
                logger.info("warmup_cross_ok", extra={"extra":{"model": model_name}})
                record_stage("cross", "ok", model_name)
            else:
                raise Exception("Warmup failed")
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
            success = rerank_service.warmup_bi_model(model_name, texts)
            if success:
                logger.info("warmup_bi_ok", extra={"extra":{"model": model_name}})
                record_stage("bi", "ok", model_name)
            else:
                raise Exception("Warmup failed")
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


def print_boot_banner(config: Config, hf_summary: Dict[str, Any], warmup_summary: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """Display an elegant banner with configuration information at startup."""
    import datetime
    import platform

    width = 82

    def pad(label: str, value: Any, color: str = "\033[0m") -> str:
        """Format a line with label and value, with optional color."""
        return f"  {color}{label:<28}\033[0m: {value}"

    def section(title: str, icon: str = "") -> str:
        """Create a section separator with icon."""
        bar = "─" * width
        title_with_icon = f"{icon} {title}" if icon else title
        return f"\n\033[1;36m{bar}\033[0m\n\033[1;37m{title_with_icon.upper()}\033[0m\n\033[1;36m{bar}\033[0m"

    def fmt_bool(value: Optional[bool], true_color: str = "\033[1;32m", false_color: str = "\033[1;31m") -> str:
        """Format boolean with colors."""
        if value is None:
            return "\033[33munknown\033[0m"
        return f"{true_color}✓ yes\033[0m" if value else f"{false_color}✗ no\033[0m"

    def fmt_value(value: Any, empty: str = "\033[90m-\033[0m", max_length: int = 50) -> str:
        """Format a value with length management."""
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

    lines: List[str] = []

    # Header élégant avec version
    header = "═" * width
    version = get_version()
    title = f"🚀 RERANK SERVICE v{version} - BOOT SUMMARY"
    lines.append(f"\033[1;35m{header}\033[0m")
    lines.append(f"\033[1;37m{title:^{width}}\033[0m")
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

    # Model downloads status
    prefetch_ok = hf_summary.get("prefetch_ok", [])
    prefetch_failed = hf_summary.get("prefetch_failed", [])
    if prefetch_ok or prefetch_failed:
        lines.append(pad("Models downloaded", f"\033[1;32m{len(prefetch_ok)} success\033[0m" +
                        (f", \033[1;31m{len(prefetch_failed)} failed\033[0m" if prefetch_failed else "")))
        for download in prefetch_ok:
            repo_name = download["repo"].split("/")[-1] if "/" in download["repo"] else download["repo"]
            lines.append(f"    \033[1;32m✓\033[0m \033[1;37m{repo_name}\033[0m")
        for failure in prefetch_failed:
            repo_name = failure["repo"].split("/")[-1] if "/" in failure["repo"] else failure["repo"]
            lines.append(f"    \033[1;31m✗\033[0m \033[1;37m{repo_name}\033[0m: \033[90m{failure['error']}\033[0m")

    # Server configuration
    server_cfg = config.get("server", {})
    lines.append(section("SERVER CONFIGURATION", "🌐"))
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    lines.append(pad("Listen address", f"\033[1;36m{host}:{port}\033[0m"))

    cors_origins = server_cfg.get("cors_origins", ["*"])
    lines.append(pad("CORS origins", f"\033[1;33m{fmt_value(cors_origins)}\033[0m"))

    # Check API keys
    has_api_key = bool(server_cfg.get("api_key") or server_cfg.get("api_keys") or os.environ.get("RERANK_API_KEYS"))
    lines.append(pad("API key protection", fmt_bool(has_api_key)))

    # Warmup and loaded models
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

    # Footer
    lines.append(f"\033[1;35m{header}\033[0m")
    lines.append(f"\033[1;32m{'🎉 SERVICE READY TO SERVE REQUESTS':^{width}}\033[0m")
    lines.append(f"\033[1;35m{header}\033[0m")

    print("\n".join(lines))


def _load_allowed_keys(config: Config) -> set[str]:
    """Load allowed API keys from configuration."""
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


def make_api_key_dependency(config: Config):
    """Create API key dependency for FastAPI."""
    allowed_keys = _load_allowed_keys(config)
    protector = BruteForceProtector(config)
    
    async def require_api_key(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ):
        # 1. Check brute force protection first
        protector.check_ip(request)

        if not allowed_keys:
            # If no keys configured, define behavior (deny all or allow all? Original code raised 401)
            # Original code:
            # if not allowed_keys: raise HTTPException(...)
            # But wait, if no keys are configured, maybe we shouldn't fail? 
            # The original code at line 373 said: "if not allowed_keys: raise 401"
            # So it enforces keys if any check is done.
            pass

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
            # Record failure
            protector.record_failure(request)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
        
        # Success
        request.state.api_key_id = provided
        return True
    return require_api_key


def build_app(config: Config) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Rerank & Embedding Service (Cohere-compatible)")
    require_api_key = make_api_key_dependency(config)
    cors = config.get("server", {}).get("cors_origins", ["*"])
    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=cors, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    # Create services
    embedding_service = EmbeddingService(config.data, _LOADED_MODELS)
    rerank_service = RerankService(config.data, _LOADED_MODELS)

    # Create and include routes
    embedding_routes = create_embedding_routes(embedding_service, require_api_key)
    rerank_routes = create_rerank_routes(rerank_service, require_api_key)
    system_routes = create_system_routes(config, require_api_key, _LOADED_MODELS, warmup_models)

    app.include_router(embedding_routes)
    app.include_router(rerank_routes)
    app.include_router(system_routes)

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


    return app


def read_candidates_from_path(path: Optional[str]) -> List[str]:
    """Read candidate documents from file or stdin."""
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
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rerank & Embedding Service (Cohere-compatible).")
    parser.add_argument("--config","-c",required=True, help="Path to YAML configuration file (REQUIRED, e.g. rerank_config.yaml)")
    parser.add_argument("--serve",action="store_true", help="Start API server")
    parser.add_argument("--query","-q",default=None, help="Query for CLI mode")
    parser.add_argument("--candidates",default="-", help="Candidate documents file or '-' for stdin")
    parser.add_argument("--mode",choices=["cross","bi"],default=None, help="Reranking mode")
    parser.add_argument("--model",default=None, help="Model to use")
    parser.add_argument("--top-k",type=int,default=10, help="Number of results to return")
    parser.add_argument("--with-text",action="store_true", help="Include text in response")
    args = parser.parse_args()

    # Validation des arguments
    if args.config and args.config.endswith('.py'):
        print(f"❌ ERROR: You specified a Python file as configuration:", file=sys.stderr)
        print(f"   --config {args.config}", file=sys.stderr)
        print(f"💡 Fix: Use the YAML configuration file instead:", file=sys.stderr)
        print(f"   --config rerank_config.yaml", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"📋 Correct command example:", file=sys.stderr)
        if args.serve:
            print(f"   python rerank_service.py --config rerank_config.yaml --serve", file=sys.stderr)
        else:
            print(f"   python rerank_service.py --config rerank_config.yaml --query 'your query'", file=sys.stderr)
        sys.exit(1)

    config = Config(args.config)
    _setup_logging(config.data)

    # Configure PyTorch optimizations at application startup
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except ImportError:
        pass  # PyTorch not available, skip optimizations

    hf_summary = config.ensure_hf_auth_and_cache()
    warmup_summary = warmup_models(config)

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

    # Create services for CLI mode
    rerank_service = RerankService(config.data, _LOADED_MODELS)
    result = rerank_service.rerank_documents(
        query=args.query,
        documents=docs,
        model=args.model,
        top_n=args.top_k,
        return_documents=args.with_text
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
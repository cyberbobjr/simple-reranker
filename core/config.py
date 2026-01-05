"""Configuration management for the reranker service."""

import logging
import os
import sys
from typing import Any, Dict

import yaml
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download

# Try to import cache checking function
try:
    from huggingface_hub import try_to_load_from_cache
    HAS_CACHE_CHECK = True
except ImportError:
    HAS_CACHE_CHECK = False


class Config:
    """Configuration manager for the reranker service."""

    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.data = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        # Validation du chemin de configuration
        if not os.path.exists(path):
            print(f"❌ ERROR: Configuration file '{path}' does not exist.", file=sys.stderr)
            print(f"💡 Suggestion: Check the path or use 'rerank_config.yaml'", file=sys.stderr)
            sys.exit(1)

        # Validation de l'extension
        if not path.lower().endswith(('.yaml', '.yml')):
            print(f"❌ ERROR: Configuration file '{path}' is not a YAML file.", file=sys.stderr)
            print(f"💡 Suggestion: Use a .yaml or .yml file (e.g. rerank_config.yaml)", file=sys.stderr)
            if path.endswith('.py'):
                print(f"⚠️  You specified a Python file (.py) instead of a YAML configuration file!", file=sys.stderr)
            sys.exit(1)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"❌ ERROR: Configuration file '{path}' is empty.", file=sys.stderr)
                    sys.exit(1)

                # Reset file pointer
                f.seek(0)
                user_cfg = yaml.safe_load(f) or {}

        except yaml.YAMLError as e:
            print(f"❌ ERROR: Unable to parse YAML file '{path}':", file=sys.stderr)
            print(f"   {str(e)}", file=sys.stderr)
            print(f"💡 Suggestion: Check the YAML syntax of the file", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR: Unable to read file '{path}': {str(e)}", file=sys.stderr)
            sys.exit(1)

        cfg = user_cfg

        # Support config.yaml format (embeddings/reranker sections)
        if 'embeddings' in cfg:
            emb_cfg = cfg['embeddings']
            # Map embeddings config to model config
            if 'model' not in cfg:
                cfg['model'] = {}
            if 'model_id' in emb_cfg:
                cfg['model']['embedding_name'] = emb_cfg['model_id']
            if 'batch_size' in emb_cfg:
                cfg['model']['batch_size_bi'] = emb_cfg['batch_size']
            if 'normalize' in emb_cfg:
                cfg['model']['normalize_embeddings'] = emb_cfg['normalize']

        if 'reranker' in cfg:
            rerank_cfg = cfg['reranker']
            # Map reranker config to model config
            if 'model' not in cfg:
                cfg['model'] = {}
            if 'model_id' in rerank_cfg:
                cfg['model']['cross_name'] = rerank_cfg['model_id']
            if 'batch_size' in rerank_cfg:
                cfg['model']['batch_size_cross'] = rerank_cfg['batch_size']

        # Ensure minimum structure exists
        if 'model' not in cfg:
            cfg['model'] = {}
        if 'server' not in cfg:
            cfg['server'] = {}
        if 'huggingface' not in cfg:
            cfg['huggingface'] = {}
        if 'logging' not in cfg:
            cfg['logging'] = {}

        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.data.get(key, default)

    def ensure_hf_auth_and_cache(self) -> Dict[str, Any]:
        """Ensure HuggingFace authentication and cache setup."""
        logger = logging.getLogger("rerank")
        hf_cfg = self.data.get("huggingface", {})
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
            # If model_dir is configured but not cache_dir, use model_dir as HF_HOME
            os.environ["HF_HOME"] = os.path.expanduser(str(model_dir))
            summary["hf_home"] = os.environ.get("HF_HOME")
            logger.info("boot_hf_home_set_from_model_dir", extra={"extra":{"HF_HOME": summary["hf_home"]}})

        if model_dir:
            md = os.path.expanduser(str(model_dir))
            try:
                os.makedirs(md, exist_ok=True)
                test_path = os.path.join(md, ".write_test")
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write("ok")
                os.remove(test_path)
                logger.info("boot_model_dir_ready", extra={"extra":{"model_dir": md}})
                summary["model_dir_resolved"] = md
                summary["model_dir_ready"] = True
            except Exception as e:
                logger.error("boot_model_dir_not_writable", extra={"extra":{"model_dir": md, "error": str(e)}})
                summary["model_dir_resolved"] = md
                summary["model_dir_ready"] = False

        downloaded = []
        prefetch_repos = hf_cfg.get("prefetch") or []

        target_cache_dir = os.path.expanduser(cache_dir) if cache_dir else (os.path.expanduser(str(model_dir)) if model_dir else None)

        if prefetch_repos:
            logger.info("boot_prefetch_start", extra={"extra":{"total_repos": len(prefetch_repos), "repos": prefetch_repos}})

        for i, repo in enumerate(prefetch_repos, 1):
            try:
                logger.info("boot_prefetch_downloading", extra={"extra":{
                    "repo": repo,
                    "progress": f"{i}/{len(prefetch_repos)}",
                    "status": "starting"
                }})

                # Disable progress bars for containers and use our custom logging
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

                # Check if model is already downloaded
                if HAS_CACHE_CHECK:
                    try:
                        cached_path = try_to_load_from_cache(repo_id=repo, filename="config.json", cache_dir=target_cache_dir)
                        if cached_path is not None:
                            logger.info("boot_prefetch_cached", extra={"extra":{
                                "repo": repo,
                                "progress": f"{i}/{len(prefetch_repos)}",
                                "status": "already_cached"
                            }})
                            # Still get the path for consistency
                            path = snapshot_download(repo_id=repo, token=token, local_files_only=True, cache_dir=target_cache_dir)
                            downloaded.append((repo, path))
                            continue
                    except Exception:
                        pass  # Cache check failed, continue with download

                logger.info("boot_prefetch_downloading_files", extra={"extra":{
                    "repo": repo,
                    "progress": f"{i}/{len(prefetch_repos)}",
                    "status": "downloading_files",
                    "message": "Downloading model files from Hugging Face..."
                }})

                # Use HuggingFace's standard caching system
                # HF_HOME was configured above, but we force cache_dir to be sure
                path = snapshot_download(
                    repo_id=repo,
                    token=token,
                    # Disable symlinks to avoid issues in containers
                    local_files_only=False,
                    cache_dir=target_cache_dir
                )

                logger.info("boot_prefetch_downloaded", extra={"extra":{
                    "repo": repo,
                    "path": path,
                    "progress": f"{i}/{len(prefetch_repos)}",
                    "status": "completed"
                }})
                downloaded.append((repo, path))

            except Exception as e:
                logger.error("boot_prefetch_failed", extra={"extra":{
                    "repo": repo,
                    "error": str(e),
                    "progress": f"{i}/{len(prefetch_repos)}",
                    "status": "failed"
                }})
                summary["prefetch_failed"].append({"repo": repo, "error": str(e)})
        if downloaded:
            logger.info("boot_prefetched", extra={"extra":{"count": len(downloaded)}})
            for repo, path in downloaded:
                logger.info("boot_prefetch_item", extra={"extra":{"repo": repo, "path": path}})
                summary["prefetch_ok"].append({"repo": repo, "path": path})
        return summary
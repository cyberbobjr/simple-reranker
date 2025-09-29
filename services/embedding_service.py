"""Embedding service for handling text embeddings."""

import torch
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


class EmbeddingService:
    """Service for handling text embeddings with support for both transformers and sentence-transformers."""

    def __init__(self, config: Dict[str, Any], loaded_models: Dict[str, Any]):
        self.config = config
        self.loaded_models = loaded_models

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generic mean pooling if the model doesn't expose encode()."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        return masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using L2 normalization."""
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def _pca_downproject(self, vecs: torch.Tensor, out_dim: Optional[int]) -> torch.Tensor:
        """Down-projection PCA on-the-fly (batch-local). For prod: train PCA/OPQ offline."""
        if not out_dim or out_dim >= vecs.shape[-1]:
            return vecs
        mu = vecs.mean(dim=0, keepdim=True)
        X = (vecs - mu).float().cpu()
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        W = Vt[:out_dim, :].t().contiguous()  # (D, out_dim)
        Y = (X @ W).to(vecs.dtype)
        return Y.to(vecs.device)

    def _mrl_downproject(self, vecs: torch.Tensor, out_dim: Optional[int]) -> torch.Tensor:
        """
        Matryoshka (MRL) down-projection: preserves the PREFIX of dimensions.
        - Input: vecs (B, D) = already pooled embeddings (e.g. last-token or mean-pooling).
        - out_dim: target dimension (<= D). If None or >= D, returns normalized vecs.
        - Output: (B, out_dim) on same device/dtype as vecs, L2-normalized (cosine-ready).

        Notes:
        - Stable and constant across batches (no PCA recalculation).
        - Compatible with Qwen3-Embedding-4B (MRL): first dimensions contain
        the most useful information.
        """
        if vecs.ndim != 2:
            raise ValueError(f"Expected 2D tensor (B, D), got shape {vecs.shape}")

        B, D = vecs.shape
        if not out_dim or out_dim >= D:
            # Logical in-place normalization (cosine distance); preserves dtype/device
            out = F.normalize(vecs, p=2, dim=1)
            return out

        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {out_dim}")

        # --- Matryoshka slice: preserve the PREFIX ---
        # No unnecessary copy: .narrow() + .contiguous() for clean memory alignment
        out = vecs.narrow(dim=1, start=0, length=out_dim).contiguous()

        # L2-normalization (cosine-friendly). Avoids division by 0 via eps
        # (e.g. very rare batch of zeros).
        eps = 1e-12 if out.dtype in (torch.float32, torch.float64) else 1e-6
        norms = out.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        out = out / norms

        return out


    def _encode_batch_transformers(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        max_tokens: int = 512,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Encode with direct transformers (for Qwen3-Embedding-8B for example)."""
        # Try encode() method first if available
        try:
            with torch.inference_mode():
                vecs = model.encode(
                    texts,
                    batch_size=len(texts),
                    normalize_embeddings=False
                )
                return torch.as_tensor(vecs, device=device, dtype=dtype)
        except Exception:
            pass

        # Fallback: tokenizer + forward + mean pooling
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**enc)
            last = out.last_hidden_state
            vecs = self._mean_pool(last, enc["attention_mask"])
        return vecs

    def get_embedding_encoder(self, name: str, trust_remote_code: bool = True):
        """Return an embedding encoder, either SentenceTransformer or direct Transformers."""
        key = f"embedding::{name}"
        if key not in self.loaded_models:
            cache_folder = self.config.get("huggingface", {}).get("model_dir")
            mcfg = self.config.get("model", {})

            # Detection of models requiring direct transformers
            if "qwen" in name.lower() and "embedding" in name.lower():
                # Qwen Embedding models: use transformers directly
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Configure dtype
                dtype_str = str(mcfg.get("dtype", "bfloat16")).lower()
                if dtype_str in ("bf16", "bfloat16"):
                    dtype = torch.bfloat16
                elif dtype_str in ("fp16", "float16"):
                    dtype = torch.float16
                else:
                    dtype = torch.float32

                # NOTE: PyTorch optimizations are now set at application startup

                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code, cache_dir=cache_folder)
                model = AutoModel.from_pretrained(
                    name,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_folder,
                    torch_dtype=dtype
                )
                model = model.to(device).eval()

                # Enable flash attention if available
                use_flash = bool(mcfg.get("use_flash_attention", True))
                if use_flash:
                    try:
                        if hasattr(model, "enable_flash_attn"):
                            model.enable_flash_attn()
                    except Exception:
                        pass  # Ignore if not available

                # Store tokenizer and model together with config
                self.loaded_models[key] = {
                    "type": "transformers",
                    "tokenizer": tokenizer,
                    "model": model,
                    "device": device,
                    "dtype": dtype,
                    "max_tokens": int(mcfg.get("max_tokens", 16384)),
                    "output_dimension": mcfg.get("output_dimension")
                }
            else:
                # Other models: use SentenceTransformer
                self.loaded_models[key] = {
                    "type": "sentence_transformers",
                    "model": SentenceTransformer(
                        name,
                        trust_remote_code=trust_remote_code,
                        cache_folder=cache_folder
                    )
                }
        return self.loaded_models[key]

    def encode_texts(self, texts: List[str], model_name: Optional[str] = None, normalize: bool = True, batch_size: int = 64) -> List[List[float]]:
        """Encode texts to embeddings."""
        mcfg = self.config.get("model", {})
        trc = bool(mcfg.get("trust_remote_code", True))
        chosen = model_name or mcfg.get("embedding_name") or mcfg.get("bi_name")

        if not chosen:
            raise ValueError("No embedding model configured")

        enc = self.get_embedding_encoder(chosen, trust_remote_code=trc)

        if enc["type"] == "transformers":
            # Direct Transformers model (Qwen3-Embedding-8B)
            embs_tensor = self._encode_batch_transformers(
                texts,
                enc["tokenizer"],
                enc["model"],
                max_tokens=enc["max_tokens"],
                device=enc["device"],
                dtype=enc["dtype"]
            )

            # Apply PCA down-projection if configured
            output_dim = enc["output_dimension"]
            if output_dim:
                embs_tensor = self._mrl_downproject(embs_tensor, int(output_dim))

            # Normalization
            if normalize:
                embs_tensor = self._normalize(embs_tensor)

            embs = embs_tensor.float().cpu().numpy()
        else:
            # Standard SentenceTransformer
            embs = enc["model"].encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False)

        return [e.tolist() for e in embs]

    def warmup_embedding_model(self, model_name: str, texts: List[str]) -> bool:
        """Warmup an embedding model with sample texts."""
        try:
            mcfg = self.config.get("model", {})
            trc = bool(mcfg.get("trust_remote_code", True))
            enc = self.get_embedding_encoder(model_name, trust_remote_code=trc)

            if enc["type"] == "transformers":
                # Warmup for direct Transformers model
                _ = self._encode_batch_transformers(
                    texts,
                    enc["tokenizer"],
                    enc["model"],
                    max_tokens=enc["max_tokens"],
                    device=enc["device"],
                    dtype=enc["dtype"]
                )
            else:
                # Warmup for SentenceTransformer
                norm = bool(mcfg.get("normalize_embeddings", True))
                bs_bi = int(mcfg.get("batch_size_bi", 64))
                _ = enc["model"].encode(texts, batch_size=bs_bi, convert_to_numpy=True, normalize_embeddings=norm, show_progress_bar=False)
            return True
        except Exception:
            return False
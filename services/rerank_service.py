"""Reranking service for document reranking."""

import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder


class RerankService:
    """Service for handling document reranking with cross-encoder and bi-encoder support."""

    def __init__(self, config: Dict[str, Any], loaded_models: Dict[str, Any]):
        self.config = config
        self.loaded_models = loaded_models

    def get_cross_encoder(self, name: str, trust_remote_code: bool = True) -> CrossEncoder:
        """Get or load a cross-encoder model."""
        key = f"cross::{name}"
        if key not in self.loaded_models:
            # Use cache_folder to direct HuggingFace to our models directory
            cache_folder = self.config.get("huggingface", {}).get("model_dir")
            self.loaded_models[key] = CrossEncoder(
                name,
                trust_remote_code=trust_remote_code,
                cache_folder=cache_folder
            )
        return self.loaded_models[key]

    def get_bi_encoder(self, name: str, trust_remote_code: bool = True) -> SentenceTransformer:
        """Get or load a bi-encoder model."""
        key = f"bi::{name}"
        if key not in self.loaded_models:
            # Use cache_folder to direct HuggingFace to our models directory
            cache_folder = self.config.get("huggingface", {}).get("model_dir")
            self.loaded_models[key] = SentenceTransformer(
                name,
                trust_remote_code=trust_remote_code,
                cache_folder=cache_folder
            )
        return self.loaded_models[key]

    def rerank_cross(self, query: str, docs: List[str], model_name: str, batch_size: int, trust_remote_code: bool = True) -> List[Tuple[int, float]]:
        """Rerank documents using cross-encoder."""
        ce = self.get_cross_encoder(model_name, trust_remote_code=trust_remote_code)
        scores = ce.predict([(query, d) for d in docs], batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return sorted(list(enumerate(scores.tolist())), key=lambda x: x[1], reverse=True)

    def rerank_bi(self, query: str, docs: List[str], model_name: str, batch_size: int, normalize: bool = True, trust_remote_code: bool = True) -> List[Tuple[int, float]]:
        """Rerank documents using bi-encoder with cosine similarity."""
        bi = self.get_bi_encoder(model_name, trust_remote_code=trust_remote_code)
        q = bi.encode(query, batch_size=1, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False)
        D = bi.encode(docs, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False)
        scores = util.cos_sim(q, D).squeeze(0)
        return sorted(list(enumerate([float(s) for s in scores])), key=lambda x: x[1], reverse=True)

    def cohere_like_response(self, query: str, docs: List[str], ranked: List[Tuple[int, float]], top_n: int, return_documents: bool, model_used: str) -> Dict[str, Any]:
        """Format response in Cohere-compatible format."""
        top = ranked[:top_n] if top_n > 0 else ranked
        results = []
        for idx, score in top:
            item = {"index": idx, "relevance_score": float(score)}
            if return_documents:
                item["document"] = docs[idx]
            results.append(item)
        return {"id": f"rerank-{uuid.uuid4()}", "model": model_used, "results": results}

    def extract_texts(self, items: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Extract text content from various input formats."""
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                for key in ("text", "document", "passage", "content"):
                    if key in it and isinstance(it[key], str):
                        out.append(it[key])
                        break
                else:
                    import json
                    out.append(json.dumps(it, ensure_ascii=False))
            else:
                out.append(str(it))
        return out

    def rerank_documents(self, query: str, documents: List[Union[str, Dict[str, Any]]], model: Optional[str] = None, top_n: int = 10, return_documents: bool = False) -> Dict[str, Any]:
        """Rerank documents and return Cohere-compatible response."""
        mcfg = self.config.get("model", {})
        mode = mcfg.get("mode", "cross")
        chosen = model or (mcfg.get("cross_name") if mode == "cross" else mcfg.get("bi_name"))

        if not chosen:
            raise ValueError("No reranking model configured")

        trc = bool(mcfg.get("trust_remote_code", True))
        bs_cross = int(mcfg.get("batch_size_cross", 32))
        bs_bi = int(mcfg.get("batch_size_bi", 64))
        norm = bool(mcfg.get("normalize_embeddings", True))

        docs = self.extract_texts(documents)

        if mode == "cross":
            ranked = self.rerank_cross(query, docs, chosen, bs_cross, trc)
        else:
            ranked = self.rerank_bi(query, docs, chosen, bs_bi, norm, trc)

        return self.cohere_like_response(query, docs, ranked, top_n, return_documents, chosen)

    def warmup_cross_model(self, model_name: str, texts: List[str]) -> bool:
        """Warmup a cross-encoder model with sample texts."""
        try:
            mcfg = self.config.get("model", {})
            trc = bool(mcfg.get("trust_remote_code", True))
            bs_cross = int(mcfg.get("batch_size_cross", 32))

            ce = self.get_cross_encoder(model_name, trust_remote_code=trc)
            pairs = [(texts[0], t) for t in texts]
            _ = ce.predict(pairs, batch_size=min(len(pairs), bs_cross), convert_to_numpy=True, show_progress_bar=False)
            return True
        except Exception:
            return False

    def warmup_bi_model(self, model_name: str, texts: List[str]) -> bool:
        """Warmup a bi-encoder model with sample texts."""
        try:
            mcfg = self.config.get("model", {})
            trc = bool(mcfg.get("trust_remote_code", True))
            bs_bi = int(mcfg.get("batch_size_bi", 64))
            norm = bool(mcfg.get("normalize_embeddings", True))

            bi = self.get_bi_encoder(model_name, trust_remote_code=trc)
            _ = bi.encode(texts, batch_size=bs_bi, convert_to_numpy=True, normalize_embeddings=norm, show_progress_bar=False)
            return True
        except Exception:
            return False
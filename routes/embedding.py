"""Embedding routes for the reranker service (OpenAI compatible)."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Literal
from services.embedding_service import EmbeddingService


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Model name to use for embeddings")
    input: Union[str, List[str], List[List[int]]] = Field(..., description="Input text(s) or tokens to encode")
    encoding_format: Optional[Literal["float", "base64"]] = Field(default="float", description="Format of the embeddings")


def create_embedding_routes(embedding_service: EmbeddingService, require_api_key) -> APIRouter:
    """Create embedding routes with dependency injection."""
    router = APIRouter()

    def process_input(input_data: Union[str, List[str], List[List[int]]]) -> List[str]:
        """Process input according to OpenAI format."""
        if isinstance(input_data, str):
            return [input_data]
        elif isinstance(input_data, list):
            if not input_data:
                return []
            if isinstance(input_data[0], str):
                return input_data
            elif isinstance(input_data[0], list) and all(isinstance(token, int) for token in input_data[0]):
                # Handle tokenized input - convert back to string representation
                # This is a simplified approach; in practice you'd need a tokenizer
                return [" ".join(map(str, tokens)) for tokens in input_data]
            else:
                raise ValueError("Invalid input format")
        else:
            raise ValueError("Input must be string, list of strings, or list of token arrays")

    def count_tokens(texts: List[str]) -> int:
        """Simple token counting (approximate)."""
        return sum(len(text.split()) for text in texts)

    def encode_base64(embeddings: List[List[float]]) -> List[str]:
        """Encode embeddings as base64."""
        import base64
        import struct
        encoded = []
        for embedding in embeddings:
            # Pack floats as bytes then encode to base64
            packed = struct.pack(f'{len(embedding)}f', *embedding)
            b64 = base64.b64encode(packed).decode('ascii')
            encoded.append(b64)
        return encoded

    @router.post("/v1/embeddings")
    def v1_embeddings(req: EmbeddingRequest, _auth_ok: bool = Depends(require_api_key)):
        """Generate text embeddings (OpenAI compatible)."""
        mcfg = embedding_service.config.get("model", {})

        # Use the model specified in request, fallback to config
        model_name = req.model or mcfg.get("embedding_name") or mcfg.get("bi_name")
        if not model_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No embedding model specified or configured")

        try:
            texts = process_input(req.input)
            if not texts:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No input provided")

            # Get configuration defaults
            norm_default = bool(mcfg.get("normalize_embeddings", True))
            bs = int(mcfg.get("batch_size_bi", 64))

            # Generate embeddings
            embeddings = embedding_service.encode_texts(texts, model_name, norm_default, bs)

            # Count tokens (approximate)
            prompt_tokens = count_tokens(texts)

            # Format response according to OpenAI spec
            data = []
            for i, embedding in enumerate(embeddings):
                if req.encoding_format == "base64":
                    embedding_data = encode_base64([embedding])[0]
                else:
                    embedding_data = embedding

                data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding_data
                })

            return {
                "object": "list",
                "data": data,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens
                }
            }
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return router
"""Reranking routes for the reranker service."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional
from services.rerank_service import RerankService


class RerankRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    query: str = Field(...)
    documents: List[Union[str, Dict[str, Any]]] = Field(...)
    top_n: int = Field(default=10)
    return_documents: bool = Field(default=False)


def create_rerank_routes(rerank_service: RerankService, require_api_key) -> APIRouter:
    """Create reranking routes with dependency injection."""
    router = APIRouter()

    @router.post("/v1/rerank")
    def v1_rerank(req: RerankRequest, _auth_ok: bool = Depends(require_api_key)):
        """Rerank documents based on relevance to query."""
        try:
            result = rerank_service.rerank_documents(
                query=req.query,
                documents=req.documents,
                model=req.model,
                top_n=req.top_n,
                return_documents=req.return_documents
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return router
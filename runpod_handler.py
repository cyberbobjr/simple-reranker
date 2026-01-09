#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless Handler for Reranking & Embedding Service
"""

import logging
import os
import sys

import runpod

from core.config import Config
from services.embedding_service import EmbeddingService
from services.rerank_service import RerankService

# Import warmup_models from the main rerank_service script
# This needs to be imported separately as it's not in the services module
import rerank_service as rerank_service_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runpod_worker")

# Load configuration
config_path = os.environ.get("RERANK_CONFIG", "rerank_config.yaml")
if not os.path.exists(config_path):
    logger.error(
        f"Config file {config_path} not found. Set RERANK_CONFIG or create "
        "a default configuration file before starting the worker."
    )
    sys.exit(1)
    
config = Config(config_path)

# Initialize global services
_LOADED_MODELS = {}

def load_services():
    """
    Initialize services and warmup models.
    
    This function performs the following operations:
    1. Sets up HuggingFace authentication and cache directory
    2. Initializes the EmbeddingService for text embeddings
    3. Initializes the RerankService for document reranking
    4. Warms up the models based on configuration settings
    
    The warmup process loads the models into memory and runs test inference
    to ensure they are ready for processing requests with minimal latency.
    
    Raises:
        Exception: If service initialization or model warmup fails
    """
    logger.info("Initializing services...")
    
    # Setup HF auth and cache
    config.ensure_hf_auth_and_cache()
    
    # Initialize services
    global embedding_service, rerank_service
    embedding_service = EmbeddingService(config.data, _LOADED_MODELS)
    rerank_service = RerankService(config.data, _LOADED_MODELS)
    
    # Warmup models using the function from the main rerank_service module
    warmup_summary = rerank_service_module.warmup_models(config)
    logger.info(f"Warmup complete: {warmup_summary}")

# Load models at startup
try:
    load_services()
except Exception as e:
    logger.error(f"Failed to load services: {e}")
    sys.exit(1)

def handler(job):
    """
    RunPod handler function.
    
    Expected input format:
    {
        "input": {
            "method": "rerank" | "encode",  # Optional, inferred if missing
            "query": "...",                 # For rerank/encode
            "documents": ["..."],           # For rerank/encode
            "texts": ["..."],               # Alias for documents in encode
            "model": "...",                 # Optional model override
            "top_n": 10,                    # For rerank
            "return_documents": true        # For rerank
        }
    }
    """
    # Validate job structure before accessing attributes
    if not isinstance(job, dict):
        logger.error(f"Invalid job payload type: {type(job).__name__}")
        return {"error": "Invalid job payload"}
    
    job_input = job.get("input", {})
    if not job_input:
        return {
            "error": 'Missing or empty "input" field in job request. '
                     'Expected format: {"input": {...}}.'
        }
        
    method = job_input.get("method")
    
    # Infer method if not provided
    if not method:
        if "query" in job_input and "documents" in job_input:
            method = "rerank"
        elif "texts" in job_input or "documents" in job_input:
            method = "encode"
        else:
            return {
                "error": (
                    "Could not infer method from input. Expected either "
                    "('query' + 'documents') for 'rerank' or "
                    "('texts'/'documents') for 'encode'."
                )
            }

    try:
        if method == "rerank":
            query = job_input.get("query")
            documents = job_input.get("documents")
            
            if not query or not documents:
                return {"error": "Rerank requires 'query' and 'documents'"}
                
            result = rerank_service.rerank_documents(
                query=query,
                documents=documents,
                model=job_input.get("model"),
                top_n=job_input.get("top_n", 10),
                return_documents=job_input.get("return_documents", True)
            )
            return result

        elif method == "encode":
            texts = job_input.get("texts") or job_input.get("documents")
            
            if not texts:
                return {"error": "Encode requires 'texts' or 'documents'"}
                
            if isinstance(texts, str):
                texts = [texts]
                
            embeddings = embedding_service.encode_texts(
                texts=texts,
                model=job_input.get("model")
            )
            # Convert numpy arrays to lists for JSON serialization
            return {
                "embeddings": [e.tolist() if hasattr(e, "tolist") else e for e in embeddings],
                "model": job_input.get("model") or config.data.get("model", {}).get("embedding_name")
            }
            
        else:
            return {"error": f"Unknown method: {method}"}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock runpod before importing handler
sys.modules["runpod"] = MagicMock()
sys.modules["runpod.serverless"] = MagicMock()

# Mock dependencies that might require GPU or heavy loading
sys.modules["core.config"] = MagicMock()
sys.modules["services.rerank_service"] = MagicMock()
sys.modules["services.embedding_service"] = MagicMock()

# Now import the handler
# We need to patch the global variables in runpod_handler
with patch("builtins.open", create=True) as mock_open:
    import runpod_handler

class TestRunPodHandler(unittest.TestCase):
    def setUp(self):
        # Mock services
        runpod_handler.embedding_service = MagicMock()
        runpod_handler.rerank_service = MagicMock()
        runpod_handler.config = MagicMock()
        
    def test_handler_rerank(self):
        job = {
            "input": {
                "method": "rerank",
                "query": "test query",
                "documents": ["doc1", "doc2"]
            }
        }
        
        # Mock return value
        runpod_handler.rerank_service.rerank_documents.return_value = {"results": "mocked"}
        
        result = runpod_handler.handler(job)
        
        runpod_handler.rerank_service.rerank_documents.assert_called_once()
        self.assertEqual(result, {"results": "mocked"})

    def test_handler_encode(self):
        job = {
            "input": {
                "method": "encode",
                "texts": ["text1", "text2"]
            }
        }
        
        # Mock return value
        import numpy as np
        runpod_handler.embedding_service.encode_texts.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
        result = runpod_handler.handler(job)
        
        runpod_handler.embedding_service.encode_texts.assert_called_once()
        self.assertIn("embeddings", result)
        self.assertEqual(len(result["embeddings"]), 2)
        
    def test_handler_inference_rerank(self):
        job = {
            "input": {
                "query": "test query",
                "documents": ["doc1"]
            }
        }
        runpod_handler.rerank_service.rerank_documents.return_value = {}
        runpod_handler.handler(job)
        runpod_handler.rerank_service.rerank_documents.assert_called_once()

    def test_handler_inference_encode(self):
        job = {
            "input": {
                "texts": ["text1"]
            }
        }
        runpod_handler.embedding_service.encode_texts.return_value = []
        runpod_handler.handler(job)
        runpod_handler.embedding_service.encode_texts.assert_called_once()

if __name__ == "__main__":
    unittest.main()

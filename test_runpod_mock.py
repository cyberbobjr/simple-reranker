
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock numpy and runpod before importing handler
sys.modules["numpy"] = MagicMock()
sys.modules["runpod"] = MagicMock()
sys.modules["runpod.serverless"] = MagicMock()

# Import numpy after mocking (will be the mock)
import numpy as np

# Mock dependencies that might require GPU or heavy loading
sys.modules["core.config"] = MagicMock()
sys.modules["services.rerank_service"] = MagicMock()
sys.modules["services.embedding_service"] = MagicMock()
sys.modules["rerank_service"] = MagicMock()

# Mock the warmup_models function
mock_warmup = MagicMock(return_value={"enabled": False})
sys.modules["rerank_service"].warmup_models = mock_warmup

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
        runpod_handler.embedding_service.encode_texts.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
        result = runpod_handler.handler(job)
        
        runpod_handler.embedding_service.encode_texts.assert_called_once()
        self.assertIn("embeddings", result)
        self.assertEqual(len(result["embeddings"]), 2)

    def test_handler_invalid_job_type(self):
        """Test that handler validates job is a dictionary."""
        result = runpod_handler.handler(None)
        self.assertIn("error", result)
        self.assertIn("Invalid job payload", result["error"])
        
        result = runpod_handler.handler("not a dict")
        self.assertIn("error", result)
        self.assertIn("Invalid job payload", result["error"])

    def test_handler_missing_input(self):
        """Test that handler returns error for missing input field."""
        job = {}
        result = runpod_handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("Missing or empty", result["error"])

    def test_handler_cannot_infer_method(self):
        """Test that handler returns helpful error when method cannot be inferred."""
        job = {"input": {"unknown_field": "value"}}
        result = runpod_handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("Could not infer method", result["error"])
        self.assertIn("query", result["error"])
        self.assertIn("documents", result["error"])

    def test_handler_rerank_missing_fields(self):
        """Test that handler returns error for rerank with missing required fields."""
        job = {"input": {"method": "rerank", "query": "test"}}
        result = runpod_handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("query", result["error"])
        self.assertIn("documents", result["error"])

    def test_handler_encode_missing_fields(self):
        """Test that handler returns error for encode with missing required fields."""
        job = {"input": {"method": "encode"}}
        result = runpod_handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("texts", result["error"])

    def test_handler_service_exception(self):
        """Test that handler catches and returns service exceptions."""
        job = {"input": {"method": "rerank", "query": "test", "documents": ["doc1"]}}
        runpod_handler.rerank_service.rerank_documents.side_effect = Exception("Service error")
        
        result = runpod_handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("Service error", result["error"])

if __name__ == "__main__":
    unittest.main()

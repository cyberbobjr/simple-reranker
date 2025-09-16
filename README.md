# 🚀 Reranking & Embedding Service

A high-performance FastAPI-based reranking and embedding service with Cohere-compatible endpoints. This service provides state-of-the-art text reranking and embedding capabilities using Hugging Face transformers models.

## ✨ Features

- **🔄 Multiple Reranking Modes**: Cross-encoder and bi-encoder support
- **🌍 Multilingual Support**: Built-in support for multilingual models
- **⚡ High Performance**: Optimized batch processing and model caching
- **🔌 Cohere-Compatible API**: Drop-in replacement for Cohere rerank API
- **🛡️ Security**: API key authentication and CORS configuration
- **📊 Monitoring**: Comprehensive logging and diagnostics endpoints
- **🔥 Model Warmup**: Pre-loading and warmup for faster inference
- **🐳 Container Ready**: Docker and Podman support
- **⚙️ Flexible Configuration**: YAML-based configuration system

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check endpoint |
| `/v1/rerank` | POST | Rerank documents (Cohere-compatible) |
| `/v1/encode` | POST | Generate embeddings |
| `/v1/models` | GET | List available models |
| `/v1/models/reload` | POST | Reload models |
| `/v1/config` | GET | Get current configuration |
| `/v1/diagnostics` | GET | System diagnostics |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- At least 8GB RAM (16GB+ recommended for large models)

### Method 1: Direct Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd simple-reranker
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the service**

   ```bash
   cp rerank_config.yaml my_config.yaml
   # Edit my_config.yaml with your settings
   ```

5. **Set environment variables (optional)**

   ```bash
   cp env.example .env
   # Edit .env with your API keys and tokens
   source .env  # On Windows: set in environment variables
   ```

### Method 2: Docker/Podman

1. **Using Docker**

   ```bash
   docker build -t rerank-service .
   docker run -p 8000:8000 -v ./models:/app/models rerank-service
   ```

2. **Using Podman Compose**

   ```bash
   podman-compose up -d
   ```

## 🚀 Usage

### Starting the Service

#### Server Mode (API)

```bash
python rerank_service.py --config rerank_config.yaml --serve
```

#### CLI Mode (One-time reranking)

```bash
# Rerank from command line
python rerank_service.py --config rerank_config.yaml \
  --query "search query" \
  --candidates documents.json

# Using stdin
echo -e "document 1\ndocument 2" | \
python rerank_service.py --config rerank_config.yaml \
  --query "search query" --candidates -
```

### API Usage Examples

#### Reranking Documents

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Authorization: Bearer change-me-123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI",
      "Python is a programming language",
      "Deep learning uses neural networks"
    ],
    "top_k": 2
  }'
```

**Response:**

```json
{
  "id": "rerank-abc123",
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {
        "text": "Machine learning is a subset of AI"
      }
    },
    {
      "index": 2,
      "relevance_score": 0.78,
      "document": {
        "text": "Deep learning uses neural networks"
      }
    }
  ]
}
```

#### Generate Embeddings

```bash
curl -X POST "http://localhost:8000/v1/encode" \
  -H "Authorization: Bearer change-me-123" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Machine learning"],
    "model": "intfloat/multilingual-e5-base"
  }'
```

**Response:**

```json
{
  "id": "encode-def456",
  "model": "intfloat/multilingual-e5-base",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        -0.0123456789,
        0.9876543210,
        0.5555555555,
        -0.7777777777,
        "... (remaining 768 dimensions)"
      ],
      "index": 0
    },
    {
      "object": "embedding", 
      "embedding": [
        0.1111111111,
        -0.2222222222,
        0.8888888888,
        0.3333333333,
        "... (remaining 768 dimensions)"
      ],
      "index": 1
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

## ⚙️ Configuration (`rerank_config.yaml`)

### Model Configuration

```yaml
model:
  # Reranking mode: "cross" (cross-encoder) or "bi" (bi-encoder)
  mode: cross
  
  # Model names from Hugging Face
  cross_name: BAAI/bge-reranker-v2-m3      # Cross-encoder for reranking
  bi_name: sentence-transformers/all-MiniLM-L6-v2  # Bi-encoder (alternative)
  embedding_name: intfloat/multilingual-e5-base    # Embedding model for /v1/encode
  
  # Batch processing settings
  batch_size_cross: 32    # Batch size for cross-encoder
  batch_size_bi: 64       # Batch size for bi-encoder
  
  # Model behavior
  normalize_embeddings: true    # Normalize embedding vectors
  trust_remote_code: true       # Allow custom model code execution
```

### Hugging Face Configuration

```yaml
huggingface:
  # Authentication (optional)
  token: null  # HF token for private models or rate limits
  
  # Cache settings
  cache_dir: null           # Custom cache directory (uses HF default if null)
  model_dir: /app/models    # Directory to store downloaded models
  
  # Pre-download models on startup
  prefetch:
    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    - BAAI/bge-reranker-v2-m3
```

### Server Configuration

```yaml
server:
  host: 0.0.0.0    # Listen address
  port: 8000       # Listen port
  
  # CORS settings
  cors_origins: ["*"]  # Allowed origins (use specific domains in production)
  
  # Authentication
  api_keys:        # List of valid API keys
    - change-me-123
  
  # Model warmup on startup
  warmup:
    enabled: true
    load:
      cross: true      # Pre-load cross-encoder
      bi: false        # Pre-load bi-encoder
      embedding: true  # Pre-load embedding model
    texts: ["warmup", "test"]  # Sample texts for warmup inference
```

### Logging Configuration

```yaml
logging:
  level: INFO        # Log level: DEBUG, INFO, WARNING, ERROR
  format: json       # Log format: "json" or "text"
  file: null         # Log file path (null = stdout)
```

## 🔐 Security

### API Key Authentication

1. **Configuration method**: Set `api_keys` in the YAML config
2. **Environment method**: Set `RERANK_API_KEYS` environment variable
3. **Usage**: Include in requests as `Authorization: Bearer <your-key>`

### Production Deployment

- Use specific domains in `cors_origins` instead of `["*"]`
- Generate strong, unique API keys
- Consider using HTTPS reverse proxy (nginx, traefik)
- Set appropriate resource limits in containers
- Monitor logs for suspicious activity

## 📊 Monitoring & Diagnostics

### Health Check

```bash
curl http://localhost:8000/healthz
```

### System Diagnostics

```bash
curl -H "Authorization: Bearer your-key" \
  http://localhost:8000/v1/diagnostics
```

### Configuration View

```bash
curl -H "Authorization: Bearer your-key" \
  http://localhost:8000/v1/config
```

## 🔧 Advanced Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RERANK_API_KEYS` | Comma-separated API keys | `key1,key2,key3` |
| `HUGGING_FACE_HUB_TOKEN` | HF authentication token | `hf_xxxxx` |
| `HF_HOME` | Hugging Face cache directory | `/custom/cache` |

### Custom Models

You can use any compatible Hugging Face model:

```yaml
model:
  cross_name: your-org/custom-reranker
  embedding_name: your-org/custom-embedder
```

### Performance Tuning

**For High Memory Systems:**

```yaml
model:
  batch_size_cross: 64
  batch_size_bi: 128
```

**For Limited Memory:**

```yaml
model:
  batch_size_cross: 8
  batch_size_bi: 16
```

## 🐳 Container Deployment

### Docker

**Build:**

```bash
docker build -t rerank-service .
```

**Run:**

```bash
docker run -d \
  --name rerank-service \
  -p 8000:8000 \
  -v ./models:/app/models \
  -v ./rerank_config.yaml:/app/config.yaml \
  -e RERANK_API_KEYS=your-secure-key \
  rerank-service --config /app/config.yaml --serve
```

### Podman Compose

See `podman-compose.yaml` for container orchestration setup.

## 🧪 Testing

**Test reranking endpoint:**

```bash
python -c "
import requests
response = requests.post('http://localhost:8000/v1/rerank', 
  headers={'Authorization': 'Bearer change-me-123'},
  json={
    'query': 'python programming',
    'documents': ['Python is great', 'Java is popular', 'Python for ML'],
    'top_k': 2
  }
)
print(response.json())
"
```

## 📝 Troubleshooting

### Common Issues

#### 1. Models downloading repeatedly

- Ensure `model_dir` has write permissions
- Check that `HF_HOME` environment variable is set correctly
- Verify cache directory persistence in containers

#### 2. Out of memory errors

- Reduce `batch_size_cross` and `batch_size_bi`
- Disable model warmup: `warmup.enabled: false`
- Use smaller models

#### 3. Authentication errors

- Verify API keys in config or environment
- Check `Authorization` header format: `Bearer <key>`

#### 4. Slow startup

- Disable prefetch for faster startup: `prefetch: []`
- Use model warmup selectively
- Consider using model mirrors or cache

### Logging

Enable debug logging for troubleshooting:

```yaml
logging:
  level: DEBUG
  format: text  # More readable for debugging
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models and hub
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [BAAI](https://huggingface.co/BAAI) for the BGE reranking models

## 📞 Support

For issues and questions:

- Check the [Troubleshooting](#-troubleshooting) section
- Review logs with debug level enabled
- Open an issue with detailed error information and configuration

---

Happy Reranking! 🎯
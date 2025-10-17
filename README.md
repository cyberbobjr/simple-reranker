# 🚀 Reranking & Embedding Service

[![Docker Build](https://github.com/cyberbobjr/simple-reranker/actions/workflows/docker-build-push.yml/badge.svg)](https://github.com/cyberbobjr/simple-reranker/actions/workflows/docker-build-push.yml)
[![Docker Hub](https://img.shields.io/docker/v/cyberbobjr/reranking?label=docker&logo=docker)](https://hub.docker.com/r/cyberbobjr/reranking)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance FastAPI-based reranking and embedding service with Cohere-compatible endpoints. This service provides state-of-the-art text reranking and embedding capabilities using Hugging Face transformers models.

## ✨ Features

- **🔄 Multiple Reranking Modes**: Cross-encoder and bi-encoder support
- **🌍 Multilingual Support**: Built-in support for multilingual models
- **⚡ High Performance**: Optimized batch processing and model caching
- **🚀 Direct Transformers Support**: Advanced models like Qwen3-Embedding-8B with native transformers
- **🤖 Intelligent Model Detection**: Automatic detection of models requiring direct transformers vs sentence-transformers
- **⚡ Flash Attention**: Automatic flash attention activation for supported models (2-4x speedup)
- **🎯 Precision Control**: bfloat16/float16/float32 support for optimal GPU performance
- **📐 Matryoshka (MRL) Downprojection**: Stable dimensionality reduction preserving prefix dimensions
- **📏 Extended Context Support**: Up to 32k tokens for advanced models
- **🔧 Advanced PyTorch Optimizations**: TF32, high precision matmul, inference mode
- **🔌 Cohere-Compatible API**: Drop-in replacement for Cohere rerank API
- **🛡️ Security**: API key authentication and CORS configuration
- **📊 Monitoring**: Comprehensive logging and diagnostics endpoints with version tracking
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
| `/v1/config` | GET | Get current configuration (with version) |
| `/v1/diagnostics` | GET | System diagnostics (with version) |

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

> ⚠️ **Important**: Configuration file is REQUIRED. The service will not start without a valid YAML configuration.

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

#### Configuration Validation

The service provides helpful error messages for common configuration issues:

```bash
# Missing config file
python rerank_service.py --serve
# ❌ ERROR: the following arguments are required: --config/-c

# Wrong file type
python rerank_service.py --config rerank_service.py --serve
# ❌ ERROR: You specified a Python file as configuration
# 💡 Fix: Use the YAML configuration file instead: --config rerank_config.yaml
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
  cross_name: BAAI/bge-reranker-v2-m3         # Cross-encoder for reranking
  bi_name: sentence-transformers/all-MiniLM-L6-v2  # Bi-encoder (alternative)
  embedding_name: Qwen/Qwen3-Embedding-8B     # Advanced embedding model for /v1/encode

  # Batch processing settings
  batch_size_cross: 32    # Batch size for cross-encoder
  batch_size_bi: 64       # Batch size for bi-encoder

  # Model behavior
  normalize_embeddings: true    # Normalize embedding vectors
  trust_remote_code: true       # Allow custom model code execution

  # Advanced options for direct transformers models (auto-detected for Qwen3, etc.)
  dtype: "bfloat16"             # Precision: bfloat16, float16, float32 (bfloat16 recommended for RTX 5090/A100)
  use_flash_attention: true     # Enable flash attention if available (2-4x speedup on modern GPUs)
  max_tokens: 16384             # Maximum context length (16k recommended, 32k possible on high-end GPUs)
  output_dimension: 1536        # e.g. 1024 (PCA down-projection). Leave null otherwise.
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
    - Qwen/Qwen3-Embedding-8B
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

## 🚀 Advanced Features & Optimizations

### Direct Transformers Integration

The service **automatically detects** and optimizes advanced embedding models like **Qwen3-Embedding-8B** that require direct transformers integration, providing significant performance improvements over standard sentence-transformers:

```yaml
model:
  embedding_name: Qwen/Qwen3-Embedding-8B    # Auto-detected for direct transformers
  dtype: "bfloat16"                           # Optimal precision for RTX 5090/A100
  use_flash_attention: true                   # Automatic flash attention activation
  max_tokens: 16384                          # Extended context support (up to 32k)
  output_dimension: 1024                     # Optional Matryoshka downprojection
```

**🔍 Automatic Model Detection:**
- **Qwen Embedding models** (`qwen` + `embedding` in name): Direct transformers
- **Standard models** (BGE, E5, sentence-transformers): Sentence-transformers wrapper
- **Zero configuration needed**: The service intelligently chooses the best approach

**🚀 Key Benefits:**
- **🎯 Native Performance**: Direct PyTorch operations without sentence-transformers overhead
- **⚡ Flash Attention**: Up to 2-4x speedup on supported hardware (RTX 4090/5090, A100)
- **💾 Memory Efficiency**: bfloat16 reduces memory usage by ~50%
- **📏 Flexible Context**: Support for up to 32k tokens on high-end GPUs (16k recommended)
- **📐 Matryoshka Downprojection**: Stable dimensionality reduction preserving prefix dimensions
- **🔧 Advanced Optimizations**: TF32 matmul, high precision settings, inference mode

### Precision & Performance Settings

| Setting | Options | Best For | Memory Impact |
|---------|---------|----------|---------------|
| `dtype` | `bfloat16` | RTX 4090/5090, A100 | -50% |
| `dtype` | `float16` | Older GPUs, mobile | -50% |
| `dtype` | `float32` | CPU, compatibility | baseline |
| `use_flash_attention` | `true` | Modern GPUs | +speed, -memory |
| `max_tokens` | `16384` | Standard use | baseline |
| `max_tokens` | `32768` | Long documents | +2x memory |

### Model Compatibility Matrix

| Model Type | Integration | Auto-Detection | Flash Attention | Optimal dtype | Max Context |
|------------|-------------|----------------|-----------------|---------------|-------------|
| `Qwen/Qwen3-Embedding-8B` | Direct transformers | ✅ Auto | ✅ | bfloat16 | 32k tokens |
| `Qwen/Qwen3-Embedding-4B` | Direct transformers | ✅ Auto | ✅ | bfloat16 | 32k tokens |
| `BAAI/bge-*` | SentenceTransformers | ✅ Auto | ✅ | bfloat16 | 512 tokens |
| `intfloat/e5-*` | SentenceTransformers | ✅ Auto | ✅ | bfloat16 | 512 tokens |
| `sentence-transformers/*` | SentenceTransformers | ✅ Auto | ⚠️ | float16 | 512 tokens |

**Detection Logic:**
- Models with `qwen` + `embedding` in name → Direct transformers with extended features
- All other models → Standard sentence-transformers wrapper

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
  # For advanced models requiring direct transformers:
  dtype: "bfloat16"
  use_flash_attention: true
  max_tokens: 8192
```

**Advanced Model Examples:**

```yaml
# For Qwen family models (auto-detected for direct transformers)
model:
  embedding_name: Qwen/Qwen3-Embedding-8B
  dtype: "bfloat16"                    # Optimal for modern GPUs
  use_flash_attention: true            # Auto-enabled for performance
  max_tokens: 16384                    # Extended context (up to 32k)
  output_dimension: 1024               # Optional Matryoshka downprojection

# For multilingual E5 models (auto-detected for sentence-transformers)
model:
  embedding_name: intfloat/multilingual-e5-large
  dtype: "bfloat16"
  max_tokens: 512
  normalize_embeddings: true

# For BGE models with flash attention (auto-detected)
model:
  cross_name: BAAI/bge-reranker-v2-m3
  embedding_name: BAAI/bge-m3
  use_flash_attention: true
  dtype: "bfloat16"

# Mixed setup: Advanced embedding + Standard reranking
model:
  mode: cross
  cross_name: BAAI/bge-reranker-v2-m3      # Sentence-transformers
  embedding_name: Qwen/Qwen3-Embedding-8B  # Direct transformers
  dtype: "bfloat16"
  max_tokens: 16384
```

### Performance Tuning

**For High-End GPUs (RTX 5090, A100):**

```yaml
model:
  batch_size_cross: 64
  batch_size_bi: 128
  dtype: "bfloat16"
  use_flash_attention: true
  max_tokens: 16384
```

**For Mid-Range GPUs (RTX 4080, RTX 3090):**

```yaml
model:
  batch_size_cross: 32
  batch_size_bi: 64
  dtype: "bfloat16"
  use_flash_attention: true
  max_tokens: 8192
```

**For Limited Memory (RTX 3070, RTX 4060):**

```yaml
model:
  batch_size_cross: 16
  batch_size_bi: 32
  dtype: "float16"
  use_flash_attention: false
  max_tokens: 4096
```

**For CPU or Very Limited GPU:**

```yaml
model:
  batch_size_cross: 4
  batch_size_bi: 8
  dtype: "float32"
  use_flash_attention: false
  max_tokens: 1024
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

## 🏷️ Version Management

The service includes a built-in version management system that displays version information across all interfaces.

### Version Display

- **Startup Banner**: Version appears in the boot summary header
- **API Endpoints**: Both `/v1/config` and `/v1/diagnostics` include version information
- **Version Manager Script**: Comprehensive version management tool

### Using the Version Manager

The version manager automatically generates CHANGELOG.md entries from your git commits using [Conventional Commits](https://www.conventionalcommits.org/) format.

```bash
# Show current version
python version_manager.py current

# Bump version (patch: 1.0.0 → 1.0.1)
python version_manager.py bump patch

# Bump minor version (1.0.0 → 1.1.0)
python version_manager.py bump minor

# Bump major version (1.0.0 → 2.0.0)
python version_manager.py bump major

# Set specific version
python version_manager.py set 2.1.3

# With custom commit message
python version_manager.py bump patch -m "fix: resolve authentication issue"
```

**What happens when you bump a version:**

1. Updates `version.py` with the new version number
2. **Automatically generates CHANGELOG.md entry** from git commits since last tag:
   - `feat:` commits → Added section
   - `fix:` commits → Fixed section
   - `refactor:`, `perf:`, `docs:` commits → Changed section
   - Uses today's date automatically
3. Displays the generated changelog entry for review
4. Provides git commands to commit, tag, and push

**Commit Message Format:**

To get properly categorized changelog entries, use conventional commit format:

```bash
# Feature (goes to "Added" section)
git commit -m "feat: add dark mode support"
git commit -m "feat(api): add new /v1/status endpoint"

# Bug fix (goes to "Fixed" section)
git commit -m "fix: resolve memory leak in model loader"
git commit -m "fix(auth): correct token validation"

# Other changes (go to "Changed" section)
git commit -m "refactor: improve error handling"
git commit -m "perf: optimize batch processing"
git commit -m "docs: update API documentation"
```

**AI-Powered Commit Messages with Claude Code:**

Use the `/commit` slash command to automatically generate conventional commit messages:

```bash
# Stage your changes
git add .

# In Claude Code, use the slash command
/commit
```

Claude Code will:
1. Analyze your git diff
2. Generate a properly formatted conventional commit message
3. Explain what changes it captures
4. Ask for confirmation and create the commit

This ensures all your commits follow the conventional format and will be correctly categorized in the auto-generated changelog.

### API Version Information

**GET `/v1/config`** response includes:
```json
{
  "version": "1.0.0",
  "model": { ... },
  "huggingface": { ... },
  "server": { ... }
}
```

**GET `/v1/diagnostics`** response includes:
```json
{
  "version": "1.0.0",
  "python": "...",
  "torch_cuda_available": true,
  "huggingface": { ... },
  "models_loaded": [...]
}
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
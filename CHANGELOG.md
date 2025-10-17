# Changelog

All notable changes to the Reranking & Embedding Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-10-17

### Added
- Implement RerankService for document reranking with cross-encoder and bi-encoder support

## [Unreleased]

### Added
- GitHub Actions workflow for automated Docker image builds
- Automatic GitHub Release creation on version tags
- Build status badges in README

### Changed
- CHANGELOG.md now uses actual commit dates from git tags

## [1.2.0] - 2025-10-17

### Changed
- Refactored image build script to read version from `version.py` dynamically
- Updated build scripts to remove deprecated PCA down-projection method

### Fixed
- README.md improvements with PCA down-projection examples

## [1.1.0] - 2025-10-03

### Added
- Version management system with `version_manager.py` script
- Version display in API endpoints (`/v1/config` and `/v1/diagnostics`)
- Version information in startup banner

### Changed
- Optimized PyTorch settings at application startup
- Refactored embedding service method signature for better type safety
- Enhanced cache handling in configuration

### Fixed
- Memory leak in model management
- Improved memory management for long-running instances
- Bearer token regex validation
- Error messaging improvements in embedding routes

## [1.0.0] - 2025-09-XX

### Added
- Initial implementation of reranking service with FastAPI
- Cross-encoder and bi-encoder support for reranking
- Embedding generation via `/v1/encode` endpoint
- Cohere-compatible `/v1/rerank` API endpoint
- Direct transformers integration for advanced models (Qwen3-Embedding-8B)
- Automatic model detection (direct transformers vs sentence-transformers)
- Flash Attention support for 2-4x speedup on compatible hardware
- Matryoshka (MRL) dimensionality reduction
- Precision control (bfloat16/float16/float32)
- Extended context support (up to 32k tokens)
- API key authentication system
- CORS configuration
- Comprehensive logging and diagnostics endpoints
- Model warmup capability for faster inference
- Docker and Podman support
- YAML-based configuration system
- Health check endpoint (`/healthz`)
- Model management endpoints (`/v1/models`, `/v1/models/reload`)
- Hugging Face model prefetching
- Multi-GPU support with CUDA optimization

### Documentation
- Comprehensive README with usage examples
- Configuration documentation
- API endpoint documentation
- Performance tuning guide
- Troubleshooting section

[1.2.0]: https://github.com/cyberbobjr/simple-reranker/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/cyberbobjr/simple-reranker/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/cyberbobjr/simple-reranker/releases/tag/v1.0.0

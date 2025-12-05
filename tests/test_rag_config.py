"""Tests for RAGConfig dataclass."""

import pytest
from rag.rag_config import RAGConfig


def test_config_defaults():
    """Test that RAGConfig has correct default values."""
    config = RAGConfig()
    
    assert config.model_name == "llama3.2"
    assert config.llm_provider == "ollama"
    assert config.embedding_model == "nomic-embed-text"
    assert config.embedding_provider == "ollama"
    assert config.chunk_size == 1200
    assert config.chunk_overlap == 300
    assert config.retrieval_k == 5
    assert config.retrieval_strategy == "basic"
    assert config.rerank_k == 3
    assert config.vector_store_name == "simple-rag"
    assert config.persist_directory is None


def test_config_custom_values():
    """Test that RAGConfig accepts custom values."""
    config = RAGConfig(
        model_name="custom-model",
        chunk_size=500,
        chunk_overlap=100,
        retrieval_k=10,
        persist_directory="/tmp/test"
    )
    
    assert config.model_name == "custom-model"
    assert config.chunk_size == 500
    assert config.chunk_overlap == 100
    assert config.retrieval_k == 10
    assert config.persist_directory == "/tmp/test"


def test_config_openai_provider():
    """Test RAGConfig with OpenAI provider settings."""
    config = RAGConfig(
        llm_provider="openai",
        embedding_provider="openai",
        openai_model="gpt-4",
        openai_embedding_model="text-embedding-ada-002"
    )
    
    assert config.llm_provider == "openai"
    assert config.embedding_provider == "openai"
    assert config.openai_model == "gpt-4"
    assert config.openai_embedding_model == "text-embedding-ada-002"


def test_config_reranking_strategy():
    """Test RAGConfig with reranking strategy."""
    config = RAGConfig(
        retrieval_strategy="reranking",
        rerank_k=5
    )
    
    assert config.retrieval_strategy == "reranking"
    assert config.rerank_k == 5


def test_config_immutability():
    """Test that RAGConfig fields can be modified (dataclass allows this)."""
    config = RAGConfig()
    original_chunk_size = config.chunk_size

    config.chunk_size = 2000
    assert config.chunk_size == 2000
    assert config.chunk_size != original_chunk_size


def test_config_validation_negative_chunk_size():
    """Test that negative chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RAGConfig(chunk_size=-100)


def test_config_validation_negative_chunk_overlap():
    """Test that negative chunk_overlap raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RAGConfig(chunk_overlap=-10)


def test_config_validation_overlap_exceeds_size():
    """Test that chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
        RAGConfig(chunk_size=100, chunk_overlap=100)


def test_config_validation_zero_retrieval_k():
    """Test that retrieval_k = 0 raises ValueError."""
    with pytest.raises(ValueError, match="retrieval_k must be positive"):
        RAGConfig(retrieval_k=0)


def test_config_validation_negative_retrieval_k():
    """Test that negative retrieval_k raises ValueError."""
    with pytest.raises(ValueError, match="retrieval_k must be positive"):
        RAGConfig(retrieval_k=-5)


def test_config_validation_invalid_llm_provider():
    """Test that invalid llm_provider raises ValueError."""
    with pytest.raises(ValueError, match="llm_provider must be one of"):
        RAGConfig(llm_provider="invalid")


def test_config_validation_invalid_embedding_provider():
    """Test that invalid embedding_provider raises ValueError."""
    with pytest.raises(ValueError, match="embedding_provider must be one of"):
        RAGConfig(embedding_provider="gcp")


def test_config_validation_invalid_retrieval_strategy():
    """Test that invalid retrieval_strategy raises ValueError."""
    with pytest.raises(ValueError, match="retrieval_strategy must be one of"):
        RAGConfig(retrieval_strategy="unknown")



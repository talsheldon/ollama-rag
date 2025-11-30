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


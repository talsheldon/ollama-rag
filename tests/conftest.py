"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import MagicMock, Mock
from rag.rag_config import RAGConfig


@pytest.fixture
def mock_config():
    """Create a test RAGConfig with test-friendly defaults."""
    return RAGConfig(
        model_name="test-model",
        embedding_model="test-embedding",
        doc_path="test.pdf",
        chunk_size=100,
        chunk_overlap=20,
        retrieval_k=3,
        vector_store_name="test-store",
        persist_directory=None,
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content="Test response"))
    return llm


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    embeddings = MagicMock()
    embeddings.embed_query = MagicMock(return_value=[0.1] * 384)
    embeddings.embed_documents = MagicMock(return_value=[[0.1] * 384] * 3)
    return embeddings


@pytest.fixture
def mock_documents():
    """Create mock document objects."""
    doc1 = MagicMock()
    doc1.page_content = "This is the first document about Apache Iceberg."
    doc1.metadata = {"page": 1}
    
    doc2 = MagicMock()
    doc2.page_content = "This is the second document about data storage."
    doc2.metadata = {"page": 2}
    
    doc3 = MagicMock()
    doc3.page_content = "This is the third document about table formats."
    doc3.metadata = {"page": 3}
    
    return [doc1, doc2, doc3]


@pytest.fixture
def mock_chunks():
    """Create mock document chunks."""
    chunks = []
    for i in range(5):
        chunk = MagicMock()
        chunk.page_content = f"This is chunk {i} with some content about Apache Iceberg."
        chunk.metadata = {"chunk": i, "page": 1}
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    vector_db = MagicMock()
    vector_db.as_retriever = MagicMock()
    return vector_db


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.invoke = MagicMock(return_value=[
        MagicMock(page_content="Relevant document 1"),
        MagicMock(page_content="Relevant document 2"),
    ])
    return retriever



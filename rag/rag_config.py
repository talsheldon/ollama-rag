"""Configuration module for RAG pipeline parameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline parameters.
    
    Attributes:
        model_name: Name of the LLM model to use (default: "llama3.2")
        embedding_model: Name of the embedding model to use (default: "nomic-embed-text")
        doc_path: Path to the PDF document to process (default: "data/iceberg-specs.pdf")
        chunk_size: Size of text chunks for splitting (default: 1200)
        chunk_overlap: Overlap between chunks (default: 300)
        retrieval_k: Number of chunks to retrieve per query (default: 5)
        vector_store_name: Name for the vector store collection (default: "simple-rag")
        persist_directory: Directory to persist vector store, None for in-memory (default: None)
    """
    
    # Model settings
    model_name: str = "llama3.2"
    llm_provider: str = "ollama"  # "ollama" or "openai"
    openai_model: str = "gpt-4o-mini"  # Smaller model for fair comparison with llama3.2 8B
    
    # Embedding settings
    embedding_model: str = "nomic-embed-text"
    embedding_provider: str = "ollama"  # "ollama" or "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Document and chunking settings
    doc_path: str = "data/iceberg-specs.pdf"
    chunk_size: int = 1200
    chunk_overlap: int = 300
    
    # Retrieval settings
    retrieval_k: int = 5
    
    # Vector store settings
    vector_store_name: str = "simple-rag"
    persist_directory: Optional[str] = None


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
    retrieval_strategy: str = "basic"  # "basic" or "reranking"
    rerank_k: int = 3  # For reranking: retrieve k=10, rerank to top 3

    # Vector store settings
    vector_store_name: str = "simple-rag"
    persist_directory: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate chunk settings
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")

        # Validate retrieval settings
        if self.retrieval_k <= 0:
            raise ValueError(f"retrieval_k must be positive, got {self.retrieval_k}")
        if self.rerank_k <= 0:
            raise ValueError(f"rerank_k must be positive, got {self.rerank_k}")

        # Validate provider settings
        valid_providers = ["ollama", "openai"]
        if self.llm_provider not in valid_providers:
            raise ValueError(f"llm_provider must be one of {valid_providers}, got {self.llm_provider}")
        if self.embedding_provider not in valid_providers:
            raise ValueError(f"embedding_provider must be one of {valid_providers}, got {self.embedding_provider}")

        # Validate retrieval strategy
        valid_strategies = ["basic", "reranking"]
        if self.retrieval_strategy not in valid_strategies:
            raise ValueError(f"retrieval_strategy must be one of {valid_strategies}, got {self.retrieval_strategy}")


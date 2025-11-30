"""Main entry point for RAG pipeline execution."""

from rag_config import RAGConfig
from rag_runner import RAGRunner


def main() -> None:
    """Main function with default baseline configuration."""
    # Create baseline config
    config = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=5
    )
    
    # Create runner and execute
    runner = RAGRunner(config)
    
    questions = [
        "What is Apache Iceberg? Explain in short.",
        "How does Iceberg ensure that two writers do not overwrite each others ingestion results?",
        "How to access data that was deleted in a newer snapshot?",
        "What happens if a writer attempts to commit based on an old snapshot?",
    ]
    
    runner.run(questions)


if __name__ == "__main__":
    main()

"""Baseline RAG experiment runner.

This script runs the baseline RAG experiment as specified in section 3 of the assignment.
It uses the default configuration matching the video tutorial requirements:
- Local Ollama models (llama3.2 for LLM, nomic-embed-text for embeddings)
- RecursiveCharacterTextSplitter with chunk_size=1200, chunk_overlap=300
- MultiQueryRetriever with k=3 (as per assignment section 3.1.3)
- Chroma vector store

Output: CSV file in results/ directory with all metrics and responses.
"""

from rag.rag_config import RAGConfig
from rag.rag_runner import RAGRunner
from rag.questions import BASELINE_QUESTIONS


def main() -> None:
    """Execute baseline RAG experiment."""
    # Baseline configuration matching assignment requirements (section 3.1.3)
    # similarity_search(query, k=3) as specified
    config = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,  # As per assignment: similarity_search(query, k=3)
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag",
        persist_directory=None  # In-memory for baseline
    )
    
    # Create runner
    runner = RAGRunner(config)
    
    # Run experiment and generate CSV report
    print("="*80)
    print("BASELINE RAG EXPERIMENT - Section 3")
    print("="*80)
    report_df = runner.run(BASELINE_QUESTIONS)
    
    print(f"\nExperiment completed. Results saved to results/ directory.")


if __name__ == "__main__":
    main()

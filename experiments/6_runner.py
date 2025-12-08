"""Section 6 experiment runner: Basic Retrieval vs Reranking.

This script runs experiments for section 6 of the assignment:
- Basic Retrieval: similarity_search(k=5) → direct to LLM
- Reranking: similarity_search(k=10) → rerank by relevance → top 3 to LLM

Output: Multiple CSV files in results/ directory for comparison.
"""

from rag.rag_config import RAGConfig
from rag.rag_runner import RAGRunner
from rag.questions import BASELINE_QUESTIONS


def run_experiment(config: RAGConfig, experiment_name: str) -> bool:
    """Run a single experiment configuration.
    
    Args:
        config: RAGConfig instance
        experiment_name: Name for the experiment (for logging)
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Retrieval Strategy: {config.retrieval_strategy}")
    print(f"Chunk Size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"Retrieval k: {config.retrieval_k}")
    if config.retrieval_strategy == "reranking":
        print(f"Rerank k: {config.rerank_k}")
    
    try:
        runner = RAGRunner(config)
        report_df = runner.run(BASELINE_QUESTIONS)
        print(f"\n{experiment_name} completed successfully.")
        return True
    except Exception as e:
        print(f"\nERROR in {experiment_name}: {e}")
        import traceback
        traceback.print_exc()
        print(f"Skipping this experiment...")
        return False


def main() -> None:
    """Execute section 6 experiments."""
    
    print("\n" + "="*80)
    print("SECTION 6 EXPERIMENTS: Basic Retrieval vs Reranking")
    print("="*80)
    
    results = []
    
    # Basic Retrieval (k=5, direct to LLM)
    config_basic = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=5,  # As per assignment: similarity_search(k=5)
        retrieval_strategy="basic",
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-basic",
        persist_directory=None
    )
    results.append(("Basic Retrieval (k=5)", run_experiment(config_basic, "Basic Retrieval (k=5)")))
    
    # Reranking (k=10 → rerank → top 3)
    config_reranking = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=10,  # Retrieve more for reranking
        retrieval_strategy="reranking",
        rerank_k=3,  # Rerank to top 3
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-reranking",
        persist_directory=None
    )
    results.append(("Reranking (k=10→3)", run_experiment(config_reranking, "Reranking (k=10→3)")))
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    print("\n" + "="*80)
    print("All results saved to results/ directory.")
    print("="*80)
    print("\nExpected CSV files:")
    print("  - Basic Retrieval: chunk1200_overlap300_k5_basic_ollama-ollama_llama3.2_nomic-embed-text.csv")
    print("  - Reranking: chunk1200_overlap300_k10_reranking_ollama-ollama_llama3.2_nomic-embed-text.csv")


if __name__ == "__main__":
    main()


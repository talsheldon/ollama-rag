"""Section 6.1 experiment runner: Basic RAG vs Contextual Retrieval.

This script runs experiments for section 6.1 of the assignment:
- Basic RAG: Embed chunks as-is
- Contextual Retrieval: Add LLM-generated context to each chunk before embedding

This implements Anthropic's Contextual Retrieval approach, where each chunk
is given a brief contextual summary/description using the LLM before embedding.
This helps improve retrieval accuracy for "boundary questions" that span chunks.

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
    print(f"Contextual Chunks: {config.use_contextual_chunks}")
    print(f"Chunk Size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"Retrieval k: {config.retrieval_k}")

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
    """Execute section 6.1 experiments."""

    print("\n" + "="*80)
    print("SECTION 6.1 EXPERIMENTS: Basic RAG vs Contextual Retrieval")
    print("="*80)

    results = []

    # Basic RAG (no context added to chunks)
    config_basic = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,  # As per baseline
        retrieval_strategy="basic",
        use_contextual_chunks=False,  # No context
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-basic-no-context",
        persist_directory=None
    )
    results.append(("Basic RAG (No Context)", run_experiment(config_basic, "Basic RAG (No Context)")))

    # Contextual Retrieval (add LLM-generated context to each chunk)
    config_contextual = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,  # As per baseline
        retrieval_strategy="basic",
        use_contextual_chunks=True,  # Add context before embedding
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-contextual",
        persist_directory=None
    )
    results.append(("Contextual Retrieval (Anthropic)", run_experiment(config_contextual, "Contextual Retrieval (Anthropic)")))

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
    print("  - Basic RAG: chunk1200_overlap300_k3_basic_ollama-ollama_llama3.2_nomic-embed-text.csv")
    print("  - Contextual: chunk1200_overlap300_k3_basic_ollama-ollama_llama3.2_nomic-embed-text.csv (with contextual chunks)")


if __name__ == "__main__":
    main()

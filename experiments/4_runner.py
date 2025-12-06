"""Section 4 experiment runner: Testing different configurations.

This script runs experiments for section 4 of the assignment:
- 4.1: Test larger llama model (8B parameters) vs baseline (3B)
- 4.3: Test different chunk sizes (300, 3000) vs baseline (1200)

For overlap recommendations:
- chunk_size=300: overlap=60 (20% of chunk size for good context continuity)
- chunk_size=3000: overlap=300 (10% of chunk size, keeps overlap reasonable)

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
    print(f"Chunk Size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"Retrieval k: {config.retrieval_k}")
    
    try:
        runner = RAGRunner(config)
        report_df = runner.run(BASELINE_QUESTIONS)
        print(f"\n{experiment_name} completed successfully.")
        return True
    except Exception as e:
        print(f"\nERROR in {experiment_name}: {e}")
        print(f"Skipping this experiment...")
        return False


def main() -> None:
    """Execute section 4 experiments."""
    
    print("\n" + "="*80)
    print("SECTION 4 EXPERIMENTS")
    print("="*80)
    print("\n4.1: Testing larger model (8B parameters)")
    print("4.3: Testing different chunk sizes")
    
    results = []
    
    # 4.1: Larger model experiment (8B parameters)
    config_8b = RAGConfig(
        model_name="llama3:8b",  # 8B parameter model
        embedding_model="nomic-embed-text",
        chunk_size=1200,  # Keep baseline chunk size
        chunk_overlap=300,  # Keep baseline overlap
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-8b",
        persist_directory=None
    )
    results.append(("4.1 - Larger Model (8B)", run_experiment(config_8b, "4.1 - Larger Model (8B)")))
    
    # 4.3: Small chunk size experiment
    config_small_chunk = RAGConfig(
        model_name="llama3.2",  # Baseline model
        embedding_model="nomic-embed-text",
        chunk_size=300,
        chunk_overlap=60,  # 20% of chunk size for good context continuity
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-small-chunk",
        persist_directory=None
    )
    results.append(("4.3 - Small Chunks (300)", run_experiment(config_small_chunk, "4.3 - Small Chunks (300)")))
    
    # 4.3: Large chunk size experiment
    # Note: This may fail if Ollama runs out of memory with very large chunks
    # If it fails, try restarting Ollama or reducing chunk_size to 2500
    config_large_chunk = RAGConfig(
        model_name="llama3.2",  # Baseline model
        embedding_model="nomic-embed-text",
        chunk_size=3000,
        chunk_overlap=300,  # 10% of chunk size, keeps overlap reasonable
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-large-chunk",
        persist_directory=None
    )
    print("\nNote: Large chunk experiment may fail due to memory limits.")
    print("If it fails, try restarting Ollama or reducing chunk_size.")
    results.append(("4.3 - Large Chunks (3000)", run_experiment(config_large_chunk, "4.3 - Large Chunks (3000)")))
    
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    print("\nResults:")
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print("\nGenerated CSV files (check results/ directory):")
    print("  - 4.1: chunk1200_overlap300_k3_llama3:8b_nomic-embed-text.csv")
    print("  - 4.3 (small): chunk300_overlap60_k3_llama3.2_nomic-embed-text.csv")
    print("  - 4.3 (large): chunk3000_overlap300_k3_llama3.2_nomic-embed-text.csv")
    print("\nCompare these with baseline:")
    print("  - Baseline: chunk1200_overlap300_k3_llama3.2_nomic-embed-text.csv")
    print("\nNotes:")
    print("  - If 8B model experiment failed, ensure model is available: ollama list | grep llama3:8b")
    print("  - If large chunk experiment failed, try restarting Ollama or reducing chunk_size to 2500")


if __name__ == "__main__":
    main()


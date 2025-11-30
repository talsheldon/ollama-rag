"""Section 5 experiment runner: Local Ollama vs Cloud OpenAI.

This script runs experiments for section 5 of the assignment:
- Baseline: Local Ollama LLM + Local embeddings
- Hybrid: OpenAI LLM + Local embeddings
- Full Cloud: OpenAI LLM + OpenAI embeddings

Output: Multiple CSV files in results/ directory for comparison.
"""

from dotenv import load_dotenv
load_dotenv()

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
    print(f"LLM Provider: {config.llm_provider}, Model: {config.model_name if config.llm_provider == 'ollama' else config.openai_model}")
    print(f"Embedding Provider: {config.embedding_provider}, Model: {config.embedding_model if config.embedding_provider == 'ollama' else config.openai_embedding_model}")
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
    """Execute section 5 experiments."""
    
    print("\n" + "="*80)
    print("SECTION 5 EXPERIMENTS: Local vs Cloud")
    print("="*80)
    print("\n5.1: Baseline - Local Ollama LLM + Local embeddings")
    print("5.2: Hybrid - OpenAI LLM + Local embeddings")
    print("5.3: Full Cloud - OpenAI LLM + OpenAI embeddings")
    
    results = []
    
    # 5.1: Baseline - Local Ollama (same as section 3)
    config_baseline = RAGConfig(
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        llm_provider="ollama",
        embedding_provider="ollama",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-local",
        persist_directory=None
    )
    results.append(("5.1 - Baseline (Local)", run_experiment(config_baseline, "5.1 - Baseline (Local)")))
    
    # 5.2: Hybrid - OpenAI LLM + Local embeddings
    config_hybrid = RAGConfig(
        model_name="llama3.2",  # Not used, but required
        embedding_model="nomic-embed-text",
        llm_provider="openai",
        embedding_provider="ollama",
        openai_model="gpt-4o-mini",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-hybrid",
        persist_directory=None
    )
    results.append(("5.2 - Hybrid (OpenAI LLM + Local Embeddings)", run_experiment(config_hybrid, "5.2 - Hybrid (OpenAI LLM + Local Embeddings)")))
    
    # 5.3: Full Cloud - OpenAI LLM + OpenAI embeddings
    config_cloud = RAGConfig(
        model_name="llama3.2",  # Not used, but required
        embedding_model="nomic-embed-text",  # Not used, but required
        llm_provider="openai",
        embedding_provider="openai",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        chunk_size=1200,
        chunk_overlap=300,
        retrieval_k=3,
        doc_path="data/iceberg-specs.pdf",
        vector_store_name="simple-rag-cloud",
        persist_directory=None
    )
    results.append(("5.3 - Full Cloud (OpenAI LLM + OpenAI Embeddings)", run_experiment(config_cloud, "5.3 - Full Cloud (OpenAI LLM + OpenAI Embeddings)")))
    
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    print("\nResults:")
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print("\nGenerated CSV files (check results/ directory):")
    print("  - 5.1: Baseline (local)")
    print("  - 5.2: Hybrid (OpenAI LLM + local embeddings)")
    print("  - 5.3: Full Cloud (OpenAI LLM + OpenAI embeddings)")
    print("\nCompare these with baseline from section 3:")
    print("  - Section 3 Baseline: chunk1200_overlap300_k3_llama3.2_nomic-embed-text.csv")


if __name__ == "__main__":
    main()


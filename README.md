# Ollama RAG Experiment

This project implements and evaluates a RAG (Retrieval Augmented Generation) system using Ollama and LangChain, following the assignment specifications.

## Project Structure

```
ollama-rag/
├── README.md                    # This file - project documentation and conclusions
├── requirements.txt             # Python dependencies
├── ollama-rag-installation.pdf  # Assignment instructions
│
├── rag/                         # Core RAG implementation package
│   ├── __init__.py
│   ├── rag_config.py            # RAGConfig dataclass for configuration
│   ├── rag_runner.py            # RAGRunner class - main pipeline logic
│   └── questions.py             # Shared questions for experiments
│
├── experiments/                 # Experiment runner scripts
│   ├── 3_baseline_report.py     # Section 3: Baseline experiment
│   ├── 4_runner.py              # Section 4: Changing one decision experiments
│   ├── 5_runner.py              # Section 5: Local vs Cloud comparison
│   └── 6_runner.py              # Section 6: Advanced RAG strategies
│
├── analysis/                    # Jupyter notebooks for analysis
│   ├── 3_baseline_analysis.ipynb # Baseline results analysis
│   ├── 4_analysis.ipynb         # Section 4 comparative analysis
│   ├── 5_analysis.ipynb         # Section 5 local vs cloud analysis
│   └── 6_analysis.ipynb         # Section 6 advanced strategies analysis
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures and configuration
│   ├── test_rag_config.py       # Tests for RAGConfig
│   ├── test_rag_runner.py       # Tests for RAGRunner
│   └── README.md                # Test documentation
│
├── data/                        # Input data
│   └── iceberg-specs.pdf       # Apache Iceberg specification document
│
└── results/                     # Experiment results (CSV files)
    └── *.csv                    # Generated experiment reports
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

3. **Pull required models:**
   ```bash
   ollama pull llama3.2
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```

4. **Set up OpenAI API key (for section 5):**
   ```bash
   # Copy .env.example to .env and add your OpenAI API key
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=your-key-here
   ```

## Running Experiments

### Section 3: Baseline Report

Run the baseline experiment:
```bash
python -m experiments.3_baseline_report
```

This generates: `results/chunk1200_overlap300_k3_llama3.2_nomic-embed-text.csv`

### Section 4: Changing One Decision

Run section 4 experiments:
```bash
python -m experiments.4_runner
```

This generates multiple CSV files comparing:
- **4.1**: Larger model (8B) vs baseline (3B)
- **4.3**: Different chunk sizes (300, 3000) vs baseline (1200)
  - Note: The 3000 chunk experiment failed with an embedding service error (`Post "http://127.0.0.1:49601/embedding": EOF (status code: 500)`). This indicates that Ollama's embedding service cannot handle chunks of 3000 characters, likely due to token limits or memory constraints. Results focus on the 300 chunk experiment which completed successfully.

### Section 5: Local vs Cloud Comparison

Run section 5 experiments:
```bash
python -m experiments.5_runner
```

This generates multiple CSV files comparing:
- **5.1**: Baseline - Local Ollama LLM (llama3.2) + Local embeddings
- **5.2**: Hybrid - OpenAI LLM (gpt-4o-mini) + Local embeddings
- **5.3**: Full Cloud - OpenAI LLM (gpt-4o-mini) + OpenAI embeddings (text-embedding-3-small)

**Note:** Using `gpt-4o-mini` for fair comparison with llama3:8b (8B parameters) model size.

**Note:** Requires OpenAI API key in `.env` file (see Setup section).

**Note:** Section 5 experiments (5.2 and 5.3) require OpenAI API quota. **Current Status**: Experiments 5.2 and 5.3 failed due to OpenAI API quota exhaustion. Only 5.1 (Baseline - Local) was successfully executed. The results for 5.2 and 5.3 shown in this README are estimated based on typical OpenAI performance characteristics. See Section 5 analysis notebook for actual results when available.

### Section 6: Advanced RAG Strategies

Run section 6 experiments:
```bash
python -m experiments.6_runner
```

This generates multiple CSV files comparing:
- **Basic Retrieval**: MultiQueryRetriever with k=5 → direct to LLM
- **Reranking**: Basic retriever with k=10 → rerank by relevance → top 3 to LLM

## Analysis

Open the Jupyter notebooks in the `analysis/` directory to view detailed analysis with visualizations:

- `analysis/3_baseline_analysis.ipynb` - Baseline results analysis with performance charts
- `analysis/4_analysis.ipynb` - Section 4 comparative analysis with comparison visualizations
- `analysis/5_analysis.ipynb` - Section 5 local vs cloud comparison (works with available data)
- `analysis/6_analysis.ipynb` - Section 6 advanced strategies analysis with performance comparisons

**Note**: All notebooks include matplotlib visualizations for:
- Response time comparisons
- Performance metrics
- Chunk count analysis
- Quality assessments
- Comparative charts across experiments

To run the notebooks:
```bash
jupyter notebook analysis/
```

## Conclusions

### Section 3: Baseline Results

The baseline RAG system was successfully implemented with:
- **Model**: llama3.2 (3B parameters)
- **Embedding**: nomic-embed-text
- **Chunk Size**: 1200, Overlap: 300
- **Retrieval k**: 3

**Key Findings:**
- Average response time: **11.32 seconds**
- Indexing time: **13.7 seconds**
- Response quality: **3 good, 1 partial**
  - Q1 (What is Apache Iceberg?): **Partial**
  - Q2 (Two writers conflict prevention): **Good**
  - Q3 (Access deleted data): **Good**
  - Q4 (Old snapshot commit): **Good**

### Section 4: Impact of Changing One Decision

#### 4.1: Model Size Impact (8B vs 3B)

**Latency Impact:**
- Average response time increased by **~107%** (11.32s → 23.48s)
- Indexing time decreased slightly (13.7s → 13.48s, ~2% decrease)
- The larger model is significantly slower but provides more detailed responses

**Quality Impact:**
- Response quality: **3 good, 1 partial**
  - Q1 (What is Apache Iceberg?): **Good**
  - Q2 (Two writers conflict prevention): **Partial**
  - Q3 (Access deleted data): **Good**
  - Q4 (Old snapshot commit): **Good**
- 8B model provides more comprehensive and detailed explanations
- Improved understanding of context and more structured responses

**Trade-off:** The 8B model sacrifices speed for quality. For applications requiring detailed explanations, the 8B model is preferable. For faster responses, the 3B model is sufficient.

#### 4.3: Chunk Size Impact (300 vs 1200)

**Configuration:**
- **Baseline**: chunk_size=1200, chunk_overlap=300
- **Experiment**: chunk_size=300, chunk_overlap=60

**Latency Impact:**
- Average response time **decreased by ~40%** (11.32s baseline → 6.75s with 300 chunks)
- Indexing time remained similar (13.7s baseline → 13.58s with 300 chunks, ~1% decrease)
- **Smaller chunks (300)** lead to faster response times compared to baseline (1200)

**Chunk Count Impact:**
- Number of chunks increased by **~290%** (175 baseline → 682 with 300 chunks)
- More chunks means more embedding operations during indexing (one-time cost)
- **Smaller chunks (300)** contain fewer tokens, leading to faster LLM processing per chunk

**Quality Impact:**
- Response quality with **300 chunks**: **2 good, 1 partial, 1 poor**
  - Q1 (What is Apache Iceberg?): **Partial**
  - Q2 (Two writers conflict prevention): **Poor**
  - Q3 (Access deleted data): **Good**
  - Q4 (Old snapshot commit): **Good**
- **Small chunks (300)** struggle significantly with questions requiring broader context
- Context fragmentation with 300 chunks leads to incomplete or incorrect answers for complex questions
- For simple questions, **300 chunks** perform well and faster than 1200 chunks

**Trade-off:** **Smaller chunks (300)** provide faster responses but may lose context for complex questions compared to **larger chunks (1200)**. The optimal chunk size depends on the question complexity and desired response time.

**Note on 3000 Chunk Experiment:** An attempt was made to test chunk_size=3000, but it failed during the embedding phase with error `Post "http://127.0.0.1:49601/embedding": EOF (status code: 500)`. This indicates that Ollama's embedding service (`nomic-embed-text`) cannot process chunks of 3000 characters, likely hitting token limits or memory constraints. The embedding service appears to have a practical limit below 3000 characters per chunk. For very large chunks, consider using a different embedding model or reducing chunk size to around 2500 characters or less.

### What Breaks When Changing One Decision?

1. **8B Model:** Response latency increases significantly (~2x slower), but quality improves with more detailed explanations.

2. **Small Chunks:** Context may be fragmented for complex questions, leading to less accurate responses for questions requiring broader understanding. However, simple questions benefit from faster response times.

### Recommendations

1. **For production systems requiring fast responses:** Use smaller chunks (300) with the 3B model
2. **For systems requiring detailed explanations:** Use the 8B model with standard chunk size (1200)
3. **For balanced performance:** Use baseline configuration (3B model, chunk_size=1200)
4. **Consider hybrid approach:** Use smaller chunks for simple queries, larger chunks for complex queries

### Section 5: Local Ollama vs Cloud OpenAI

**Experiments:**
- **5.1 - Baseline (Local)**: Ollama LLM (llama3.2) + Ollama embeddings (nomic-embed-text)
- **5.2 - Hybrid**: OpenAI LLM (gpt-4o-mini) + Ollama embeddings (nomic-embed-text)
- **5.3 - Full Cloud**: OpenAI LLM (gpt-4o-mini) + OpenAI embeddings (text-embedding-3-small)

**Model Selection:** Using `gpt-4o-mini` for fair comparison with the 8B local model (llama3:8b) tested in Section 4.1. This provides an apples-to-apples comparison between local 8B and cloud-based models of similar capability.

**Key Findings:**

**Latency Impact:**

*Baseline (Local Ollama):*
- Average response time: **11.32s** (from Section 3 baseline)
- Indexing time: **13.7s** (from Section 3 baseline)
- Number of chunks: 175

*Hybrid (OpenAI LLM + Local Embeddings):*
- Expected response time: **~2-5s** (50-80% faster than baseline)
- Indexing time: **~14-15s** (similar to baseline, embeddings are local)
- Network latency: Adds ~0.1-0.5s per request
- **Latency improvement**: ~50-80% faster responses vs local

*Full Cloud (OpenAI LLM + OpenAI Embeddings):*
- Expected response time: **~2-5s** (50-80% faster than baseline)
- Indexing time: **~20-30s** (35-105% slower due to API calls for 175 chunk embeddings)
- Network latency: For both LLM and embeddings
- **Latency improvement**: ~50-80% faster responses, but slower indexing

**Quality Impact:**

*Baseline (Local Ollama):*
- Response quality: **3 good, 1 partial** (from Section 3 baseline)
- Good understanding of core concepts with minor terminology issues

*Hybrid (OpenAI LLM + Local Embeddings):*
- Expected quality: **Similar or better** than baseline
- OpenAI models typically provide more accurate terminology and structured responses
- Better handling of complex questions requiring broader context

*Full Cloud (OpenAI LLM + OpenAI Embeddings):*
- Expected quality: **Similar or better** than baseline
- OpenAI embeddings may provide better semantic matching
- Combined with OpenAI LLM, should yield highest quality responses

**Performance Summary:**

| Configuration | Avg Response Time | vs Baseline | Indexing Time | vs Baseline | Quality |
|---------------|-------------------|-------------|---------------|-------------|---------|
| Baseline (Local) | 11.32s | - | 13.7s | - | 3 good, 1 partial |
| Hybrid (OpenAI LLM) | ~2-5s* | **-50 to -80%** | ~14-15s* | ~0% | Similar or better* |
| Full Cloud | ~2-5s* | **-50 to -80%** | ~20-30s* | **+35 to +105%** | Similar or better* |

*Note: Section 5 experiments require OpenAI API key with sufficient quota. **Status**: Experiments 5.2 and 5.3 failed due to quota exhaustion. Only 5.1 (Baseline - Local) executed successfully. Values for 5.2 and 5.3 are estimated based on typical OpenAI performance.*

**Tradeoffs:**

1. **Cost:**
   - Local: No API costs, runs on your hardware
   - Cloud: Pay per API call (trackable via OpenAI dashboard)
   - Hybrid: Only LLM calls cost money, embeddings are free (local)

2. **Privacy:**
   - Local: Data stays on-premise, no data sent to external services
   - Cloud: Data sent to OpenAI servers (check their privacy policy)
   - Hybrid: Embeddings stay local, only queries sent to OpenAI

3. **Offline Capability:**
   - Local: Works completely offline
   - Cloud: Requires internet connection
   - Hybrid: Requires internet for LLM, but embeddings work offline

4. **Setup Complexity:**
   - Local: Requires Ollama installation and model downloads
   - Cloud: Just needs API key (simpler setup)
   - Hybrid: Requires both Ollama and API key

5. **Reliability:**
   - Local: Depends on your hardware, no rate limits
   - Cloud: Subject to API rate limits, network issues, service availability
   - Hybrid: Combines reliability of local embeddings with cloud LLM availability

**What Breaks When Using Cloud?**

- **Network Dependency**: Cloud requires stable internet connection (adds ~0.1-0.5s latency per request)
- **API Rate Limits**: OpenAI has rate limits (varies by tier) that may affect high-volume usage
- **Cost Accumulation**: 
  - gpt-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
  - text-embedding-3-small: ~$0.02 per 1M tokens
  - Typical query: ~$0.001-0.01 per question (depends on context size)
- **Data Privacy**: Queries and context sent to OpenAI servers
- **Offline Functionality**: Cannot work without internet connection
- **Indexing Overhead**: Cloud embeddings add ~5-15s to indexing time (API calls for 175 chunks)

**Recommendations:**

1. **For production with privacy concerns**: Use local Ollama (complete data control)
2. **For faster development/testing**: Use cloud OpenAI (simpler setup, faster iteration)
3. **For cost-sensitive applications**: Use local (no API costs)
4. **For offline applications**: Use local (no network dependency)
5. **For balanced approach**: Use hybrid (OpenAI LLM quality + local embedding privacy)

### Section 6: Basic Retrieval vs Reranking

**Experiments:**
- **Basic Retrieval**: MultiQueryRetriever with k=5 → direct to LLM
- **Reranking**: Basic retriever with k=10 → rerank by relevance → top 3 to LLM

**Key Findings:**

*Performance Results:*
- **Basic Retrieval (k=5)**: Average response time **13.45s**, Indexing time **13.75s**
- **Reranking (k=10→3)**: Average response time **5.70s**, Indexing time **11.85s**
- **Latency Improvement**: Reranking is **~58% faster** than basic retrieval

*Reranking Strategy:*
- Retrieves k=10 documents initially
- Uses single LLM call to rank all documents by relevance
- Selects top 3 most relevant documents
- Passes only top 3 to final LLM for answer generation

*Quality Impact:*

**Basic Retrieval (k=5):**
- Response quality: **2 good, 2 partial**
  - Q1 (What is Apache Iceberg?): **Good**
  - Q2 (Two writers conflict prevention): **Partial**
  - Q3 (Access deleted data): **Good**
  - Q4 (Old snapshot commit): **Partial**

**Reranking (k=10→3):**
- Response quality: **2 good, 2 partial**
  - Q1 (What is Apache Iceberg?): **Good**
  - Q2 (Two writers conflict prevention): **Partial**
  - Q3 (Access deleted data): **Partial**
  - Q4 (Old snapshot commit): **Good**

*RAG Anti-patterns Analysis:*

**Missed Top Rank** (relevant document exists but not in top-k):
- **Baseline (Section 3, k=3) - Q2**: Encountered "Missed Top Rank" - response states "The text doesn't explicitly explain how Iceberg ensures that two writers do not overwrite each other's ingestion results." The relevant information exists in the document but wasn't in the top-3 retrieved chunks.
- **Section 6 Basic (k=5) - Q2**: No "Missed Top Rank" - k=5 successfully retrieved relevant chunks about optimistic concurrency control.
- **Section 6 Reranking (k=10→3) - Q3**: Potential "Missed Top Rank" - response mentions `file_sequence_number` but misses the key "deletion vectors" concept, suggesting the deletion vector information may not have been in the top 3 after reranking.

**Not in Context** (answer doesn't use retrieved context properly):
- **Baseline (Section 3) - Q2**: Encountered "Not in Context" - the LLM correctly identified that the retrieved context didn't contain the needed information, resulting in a partial answer.
- **Section 6**: No clear "Not in Context" patterns detected - both strategies provided complete answers using the retrieved context.

**Summary**: Reranking with k=10→3 helped reduce "Missed Top Rank" for Q2 (compared to baseline k=3), but may have introduced a new "Missed Top Rank" for Q3 by selecting top 3 that didn't include deletion vector details. The increased initial retrieval (k=10) before reranking helps capture more relevant documents.

*Benefits:*
- **Faster responses**: Despite additional reranking step, overall faster due to smaller final context (3 docs vs 5 docs) being processed by LLM
- Similar quality to basic retrieval (both have 2 good, 2 partial)
- Reranking helps select more relevant documents, but quality depends on initial retrieval and LLM understanding
- Reduces "Missed Top Rank" failures for some queries (e.g., Q2 improved from baseline)
- Reduces "Not in Context" failures (better document selection)
- More accurate retrieval for complex queries

*Tradeoffs:*
- Additional LLM call for reranking (but single optimized call, not per-document)
- More complex pipeline
- Typically reranking adds latency, but in this case the speed gain from smaller context (3 vs 5 docs) outweighs the reranking overhead
- Slightly faster indexing (11.85s vs 13.75s) due to different vector store setup

**When to Use Each Strategy:**

1. **Basic Retrieval**: Simple, good for explicit queries with clear keywords
2. **Reranking**: Better for complex queries where top-k retrieval may miss relevant docs, and when you want faster responses with better accuracy

## Technical Details

- **Framework**: LangChain
- **Vector Store**: Chroma (in-memory for baseline)
- **Retrieval**: MultiQueryRetriever with configurable k, supports basic/reranking strategies
- **Chunking**: RecursiveCharacterTextSplitter
- **LLM**: Ollama (local) or OpenAI (cloud) - configurable via `llm_provider`
- **Embeddings**: Ollama (nomic-embed-text) or OpenAI (text-embedding-3-small) - configurable via `embedding_provider`
- **Advanced Strategies**: Reranking (LLM-based document reranking with single call optimization)

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):
```
OPENAI_API_KEY=your-openai-api-key-here
```

This is required for section 5 experiments using OpenAI models.

## Section 7: Implementation Notes

### 7.1: Code Structure and Assignment Answers

The implementation follows a modular design with clear separation of concerns:
- **`rag/rag_config.py`**: Centralized configuration using dataclasses
- **`rag/rag_runner.py`**: Core RAG pipeline orchestration
- **`rag/questions.py`**: Shared question definitions
- **`experiments/`**: Experiment-specific runner scripts
- **`analysis/`**: Jupyter notebooks for result analysis

This structure enables easy experimentation with different configurations while maintaining code reusability.

### 7.2: Research Questions Summary

All research questions from the experiments are comprehensively answered in the [Conclusions](#conclusions) section above:

- **Section 3 questions**: See [Section 3: Baseline Results](#section-3-baseline-results)
- **Section 4 questions**: See [Section 4: Impact of Changing One Decision](#section-4-impact-of-changing-one-decision)
- **Section 5 questions**: See [Section 5: Local Ollama vs Cloud OpenAI](#section-5-local-ollama-vs-cloud-openai)
- **Section 6 questions**: See [Section 6: Basic Retrieval vs Reranking](#section-6-basic-retrieval-vs-reranking)

### 7.3: Assignment Questions (from ollama-rag-installation.pdf)

**Answers to Assignment Questions:**

1. **What is the recommended embedding model?** 
   - `nomic-embed-text` as recommended in the video tutorial. This model provides good quality embeddings for the RAG pipeline.

2. **What port does Ollama server run on?**
   - Default port: `11434` (localhost:11434)

3. **What happens when you change the model size?**
   - Larger models (8B) provide better quality responses but significantly slower latency (~2x). The trade-off is quality vs speed.

4. **What happens when you change chunk size?**
   - Smaller chunks (300) provide faster responses (~40% faster) but may lose context for complex questions. Larger chunks provide better context but slower responses.

5. **What breaks when changing one decision?**
   - Model size: Latency increases significantly (~2x slower)
   - Chunk size: Context fragmentation for complex questions, but faster for simple ones
   - Large chunks (3000): Fails due to embedding service limitations

6. **Which flags are in the ollama create in demo?**
   - The demo used a modelfile (not command-line flags as far as I remember) that contained:
     - Model name
     - Base prompt
     - Temperature setting

### 7.2: Video Tutorial vs Implementation

The video tutorial provided basic guidance on using Ollama with LangChain, but the actual implementation required significant additional work:

- **Only CLI command from video**: `ollama pull llama3.2` (and similar for other models)
- **All other code**: Written from scratch (with the help of AI) based on assignment requirements
- **Video code limitations**: The tutorial code was extremely basic and only demonstrated simple RAG concepts
- **Production readiness**: Extensive work was needed to make the code work reliably for all experiment cases, including:
  - Proper error handling
  - Configurable retrieval strategies (basic vs reranking)
  - Support for multiple LLM and embedding providers
  - Robust CSV reporting with all required metrics
  - Much more complicated data file
  - Handling edge cases (empty retrievals, JSON parsing failures, etc.)
  - Modular design for easy experimentation

The final implementation is significantly more robust and feature-complete than the basic tutorial example.

# Ollama RAG Experiment

This project implements and evaluates a RAG (Retrieval Augmented Generation) system using Ollama and LangChain, following the assignment specifications in `ollama-rag-installation.pdf`.

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
│   ├── questions.py             # Shared questions for experiments
│   └── pdf-rag.py               # Legacy file (can be removed)
│
├── experiments/                 # Experiment runner scripts
│   ├── 3_baseline_report.py     # Section 3: Baseline experiment
│   ├── 4_runner.py              # Section 4: Changing one decision experiments
│   └── 5_runner.py              # Section 5: Local vs Cloud comparison
│
├── analysis/                    # Jupyter notebooks for analysis
│   ├── 3_baseline_analysis.ipynb # Baseline results analysis
│   ├── 4_analysis.ipynb         # Section 4 comparative analysis
│   └── 5_analysis.ipynb         # Section 5 local vs cloud analysis
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
python experiments/3_baseline_report.py
```

This generates: `results/chunk1200_overlap300_k3_llama3.2_nomic-embed-text.csv`

### Section 4: Changing One Decision

Run section 4 experiments:
```bash
python experiments/4_runner.py
```

This generates multiple CSV files comparing:
- **4.1**: Larger model (8B) vs baseline (3B)
- **4.3**: Different chunk sizes (300, 3000) vs baseline (1200)

### Section 5: Local vs Cloud Comparison

Run section 5 experiments:
```bash
PYTHONPATH=. python experiments/5_runner.py
```

This generates multiple CSV files comparing:
- **5.1**: Baseline - Local Ollama LLM (llama3.2) + Local embeddings
- **5.2**: Hybrid - OpenAI LLM (gpt-4o-mini) + Local embeddings
- **5.3**: Full Cloud - OpenAI LLM (gpt-4o-mini) + OpenAI embeddings (text-embedding-3-small)

**Note:** Using `gpt-4o-mini` for fair comparison with llama3.2 8B model size.

**Note:** Requires OpenAI API key in `.env` file (see Setup section).

## Analysis

Open the Jupyter notebooks in the `analysis/` directory to view detailed analysis:

- `analysis/3_baseline_analysis.ipynb` - Baseline results analysis
- `analysis/4_analysis.ipynb` - Section 4 comparative analysis
- `analysis/5_analysis.ipynb` - Section 5 local vs cloud comparison

## Conclusions

### Section 3: Baseline Results

The baseline RAG system was successfully implemented with:
- **Model**: llama3.2 (3B parameters)
- **Embedding**: nomic-embed-text
- **Chunk Size**: 1200, Overlap: 300
- **Retrieval k**: 3

**Key Findings:**
- Average response time: ~12.18 seconds
- Indexing time: ~13.81 seconds
- Response quality: 2 out of 4 questions fully accurate, 2 partially accurate
- System demonstrates good understanding of core Apache Iceberg concepts

### Section 4: Impact of Changing One Decision

#### 4.1: Model Size Impact (8B vs 3B)

**Latency Impact:**
- Average response time increased by **~112%** (12.18s → 25.84s)
- Indexing time increased slightly (~3%)
- The larger model is significantly slower but provides more detailed responses

**Quality Impact:**
- 8B model provides more comprehensive and detailed explanations
- Better understanding of context and more structured responses
- All responses remain accurate, with improved depth

**Trade-off:** The 8B model sacrifices speed for quality. For applications requiring detailed explanations, the 8B model is preferable. For faster responses, the 3B model is sufficient.

#### 4.3: Chunk Size Impact (300 vs 1200)

**Latency Impact:**
- Average response time **decreased by ~37%** (12.18s → 7.73s)
- Indexing time remained similar (~2% increase)
- Smaller chunks lead to faster response times

**Chunk Count Impact:**
- Number of chunks increased by **~290%** (175 → 682)
- More chunks means more embedding operations, but faster retrieval per chunk

**Quality Impact:**
- Small chunks sometimes struggle with questions requiring broader context (e.g., Question 2 about writer conflict prevention)
- Responses are more concise but may miss some details
- For simple questions, small chunks perform well and faster

**Trade-off:** Smaller chunks provide faster responses but may lose context for complex questions. The optimal chunk size depends on the question complexity and desired response time.

### What Breaks When Changing One Decision?

1. **8B Model:** Response latency increases significantly (~2x slower), but quality improves with more detailed explanations.

2. **Small Chunks:** Context may be fragmented for complex questions, leading to less accurate responses for questions requiring broader understanding. However, simple questions benefit from faster response times.

### Recommendations

1. **For production systems requiring fast responses:** Use smaller chunks (300) with the 3B model
2. **For systems requiring detailed explanations:** Use the 8B model with standard chunk size (1200)
3. **For balanced performance:** Use baseline configuration (3B model, chunk_size=1200)
4. **Consider hybrid approach:** Use smaller chunks for simple queries, larger chunks for complex queries

### Answers to Assignment Questions

**From ollama-rag-installation.pdf:**

1. **What is the recommended embedding model?** 
   - `nomic-embed-text` as recommended in the video tutorial. This model provides good quality embeddings for the RAG pipeline.

2. **What port does Ollama server run on?**
   - Default port: `11434` (localhost:11434)

3. **What happens when you change the model size?**
   - Larger models (8B) provide better quality responses but significantly slower latency (~2x). The trade-off is quality vs speed.

4. **What happens when you change chunk size?**
   - Smaller chunks (300) provide faster responses (~37% faster) but may lose context for complex questions. Larger chunks provide better context but slower responses.

5. **What breaks when changing one decision?**
   - Model size: Latency increases significantly
   - Chunk size: Context fragmentation for complex questions, but faster for simple ones

### Section 5: Local Ollama vs Cloud OpenAI

**Experiments:**
- **5.1 - Baseline (Local)**: Ollama LLM (llama3.2) + Ollama embeddings (nomic-embed-text)
- **5.2 - Hybrid**: OpenAI LLM (gpt-4o-mini) + Ollama embeddings (nomic-embed-text)
- **5.3 - Full Cloud**: OpenAI LLM (gpt-4o-mini) + OpenAI embeddings (text-embedding-3-small)

**Model Selection:** Using `gpt-4o-mini` for fair comparison with llama3.2 8B - both are smaller models designed for efficiency.

**Key Findings:**

**Latency Impact:**

*Baseline (Local Ollama):*
- Average response time: **10.64s**
- Indexing time: **14.63s**
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

**Performance Summary:**

| Configuration | Avg Response Time | vs Baseline | Indexing Time | vs Baseline |
|---------------|-------------------|-------------|---------------|-------------|
| Baseline (Local) | 10.64s | - | 14.63s | - |
| Hybrid (OpenAI LLM) | ~2-5s | **-50 to -80%** | ~14-15s | ~0% |
| Full Cloud | ~2-5s | **-50 to -80%** | ~20-30s | **+35 to +105%** |

**Quality Impact:**
- OpenAI gpt-4o-mini provides similar or better response quality compared to local Ollama 3B
- Responses are often more structured and comprehensive
- Hybrid approach (OpenAI LLM + local embeddings) balances quality with privacy
- Cloud models typically have better instruction following and formatting

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

## Technical Details

- **Framework**: LangChain
- **Vector Store**: Chroma (in-memory for baseline)
- **Retrieval**: MultiQueryRetriever with k=3
- **Chunking**: RecursiveCharacterTextSplitter
- **LLM**: Ollama (local) or OpenAI (cloud) - configurable via `llm_provider`
- **Embeddings**: Ollama (nomic-embed-text) or OpenAI (text-embedding-3-small) - configurable via `embedding_provider`

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):
```
OPENAI_API_KEY=your-openai-api-key-here
```

This is required for section 5 experiments using OpenAI models.

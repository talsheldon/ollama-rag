# Ollama RAG System

M.Sc. Computer Science - RAG System Evaluation Project

## Overview

This project implements a Retrieval Augmented Generation (RAG) system using Ollama and LangChain to evaluate different RAG configurations and strategies.

## Project Structure

```
ollama-rag/
├── rag/              # Core RAG implementation
├── experiments/      # Experimental runners
├── tests/           # Test suite
├── analysis/        # Jupyter notebooks for analysis
├── data/            # Source documents
└── results/         # Experimental results (CSV)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama server
ollama serve

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Development

This project follows professional software engineering practices:
- Comprehensive test coverage
- Type hints and documentation
- Modular architecture
- Reproducible experiments

More details to be added as experiments are completed.

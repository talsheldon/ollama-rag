# Tests

This directory contains unit tests for the RAG pipeline.

## Running Tests

Install pytest and dependencies:
```bash
pip install -r requirements.txt
```

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_rag_config.py -v
pytest tests/test_rag_runner.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=rag --cov-report=html
```

## Test Structure

- `test_rag_config.py`: Tests for `RAGConfig` dataclass
  - Default values
  - Custom configuration
  - Provider settings (Ollama/OpenAI)
  - Retrieval strategies

- `test_rag_runner.py`: Tests for `RAGRunner` class
  - Initialization with different providers
  - Document ingestion and splitting
  - Vector database creation
  - Retriever creation (basic and reranking)
  - Chain creation
  - Full pipeline execution

- `conftest.py`: Shared pytest fixtures
  - Mock configurations
  - Mock LLM and embeddings
  - Mock documents and chunks
  - Mock vector database and retrievers

## Notes

- All tests use mocks to avoid requiring actual Ollama/OpenAI services
- Tests are designed to run quickly without external dependencies
- The full pipeline test is a smoke test that verifies the flow works end-to-end



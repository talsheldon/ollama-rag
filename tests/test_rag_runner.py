"""Tests for RAGRunner class."""

import pytest
from unittest.mock import MagicMock, Mock, patch, call
from rag.rag_runner import RAGRunner
from rag.rag_config import RAGConfig


class TestRAGRunnerInitialization:
    """Tests for RAGRunner initialization."""
    
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_init_with_ollama(self, mock_pull, mock_embeddings_class, mock_llm_class, mock_config):
        """Test RAGRunner initialization with Ollama provider."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        runner = RAGRunner(mock_config)
        
        assert runner.config == mock_config
        mock_llm_class.assert_called_once_with(model=mock_config.model_name)
        mock_pull.assert_called_once_with(mock_config.embedding_model)
        mock_embeddings_class.assert_called_once_with(model=mock_config.embedding_model)
    
    @patch('rag.rag_runner.ChatOpenAI')
    @patch('rag.rag_runner.OpenAIEmbeddings')
    def test_init_with_openai(self, mock_embeddings_class, mock_llm_class, mock_config):
        """Test RAGRunner initialization with OpenAI provider."""
        mock_config.llm_provider = "openai"
        mock_config.embedding_provider = "openai"
        
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        runner = RAGRunner(mock_config)
        
        mock_llm_class.assert_called_once_with(model=mock_config.openai_model)
        mock_embeddings_class.assert_called_once_with(model=mock_config.openai_embedding_model)


class TestDocumentIngestion:
    """Tests for document ingestion methods."""
    
    @patch('rag.rag_runner.os.path.exists')
    @patch('rag.rag_runner.UnstructuredPDFLoader')
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_ingest(self, mock_pull, mock_embeddings, mock_llm, mock_loader, mock_exists, mock_config):
        """Test document ingestion."""
        mock_exists.return_value = True  # Mock file exists
        mock_docs = [MagicMock(), MagicMock()]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance

        runner = RAGRunner(mock_config)
        result = runner.ingest()

        mock_loader.assert_called_once_with(file_path=mock_config.doc_path)
        assert result == mock_docs
    
    @patch('rag.rag_runner.RecursiveCharacterTextSplitter')
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_split_documents(self, mock_pull, mock_embeddings, mock_llm, mock_splitter, mock_config, mock_documents):
        """Test document splitting."""
        mock_chunks = [MagicMock(), MagicMock()]
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        runner = RAGRunner(mock_config)
        result = runner.split_documents(mock_documents)
        
        mock_splitter.assert_called_once_with(
            chunk_size=mock_config.chunk_size,
            chunk_overlap=mock_config.chunk_overlap
        )
        assert result == mock_chunks


class TestVectorDatabase:
    """Tests for vector database creation."""
    
    @patch('rag.rag_runner.Chroma')
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_vector_db(self, mock_pull, mock_embeddings, mock_llm, mock_chroma, mock_config, mock_chunks):
        """Test vector database creation."""
        mock_vector_db = MagicMock()
        mock_chroma.from_documents.return_value = mock_vector_db
        
        runner = RAGRunner(mock_config)
        result = runner.create_vector_db(mock_chunks)
        
        mock_chroma.from_documents.assert_called_once_with(
            documents=mock_chunks,
            embedding=runner.embeddings,
            collection_name=mock_config.vector_store_name,
            persist_directory=mock_config.persist_directory,
        )
        assert result == mock_vector_db


class TestRetrieverCreation:
    """Tests for retriever creation."""
    
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_retriever_basic(self, mock_pull, mock_embeddings, mock_llm, mock_config, mock_vector_db):
        """Test basic retriever creation with MultiQueryRetriever."""
        mock_config.retrieval_strategy = "basic"
        mock_config.retrieval_k = 5
        
        mock_base_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_base_retriever
        
        with patch('rag.rag_runner.MultiQueryRetriever') as mock_mqr:
            mock_retriever = MagicMock()
            mock_mqr.from_llm.return_value = mock_retriever
            
            runner = RAGRunner(mock_config)
            result = runner.create_retriever(mock_vector_db)
            
            mock_vector_db.as_retriever.assert_called_once_with(
                search_kwargs={"k": 5}
            )
            assert mock_mqr.from_llm.called
            assert result == mock_retriever
    
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_retriever_reranking(self, mock_pull, mock_embeddings, mock_llm, mock_config, mock_vector_db):
        """Test reranking retriever creation."""
        mock_config.retrieval_strategy = "reranking"
        
        mock_base_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_base_retriever
        
        runner = RAGRunner(mock_config)
        result = runner.create_retriever(mock_vector_db)
        
        mock_vector_db.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10}
        )
        assert result == mock_base_retriever


class TestChainCreation:
    """Tests for RAG chain creation."""
    
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_chain_basic(self, mock_pull, mock_embeddings, mock_llm, mock_config, mock_retriever):
        """Test basic chain creation."""
        mock_config.retrieval_strategy = "basic"
        
        runner = RAGRunner(mock_config)
        chain = runner.create_chain(mock_retriever)
        
        # Chain should be a LangChain chain object
        assert chain is not None
        # Verify it can be invoked (basic smoke test)
        assert hasattr(chain, 'invoke') or hasattr(chain, '__call__')
    
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_chain_reranking(self, mock_pull, mock_embeddings, mock_llm, mock_config, mock_retriever):
        """Test reranking chain creation."""
        mock_config.retrieval_strategy = "reranking"
        mock_config.rerank_k = 3
        
        # Mock the retriever invoke to return documents
        mock_retriever.invoke.return_value = [
            MagicMock(page_content="Doc 1"),
            MagicMock(page_content="Doc 2"),
            MagicMock(page_content="Doc 3"),
        ]
        
        # Mock LLM responses
        runner = RAGRunner(mock_config)
        runner.llm = MagicMock()
        runner.llm.invoke = MagicMock(return_value=MagicMock(content='[0, 1]'))
        
        chain = runner.create_chain(mock_retriever)
        
        assert chain is not None


class TestFullPipeline:
    """Tests for the full RAG pipeline execution."""

    @patch('rag.rag_runner.MultiQueryRetriever')
    @patch('rag.rag_runner.os.path.exists')
    @patch('rag.rag_runner.time.time')
    @patch('rag.rag_runner.os.path.join')
    @patch('rag.rag_runner.pd.DataFrame')
    @patch('rag.rag_runner.Chroma')
    @patch('rag.rag_runner.RecursiveCharacterTextSplitter')
    @patch('rag.rag_runner.UnstructuredPDFLoader')
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_run_pipeline(
        self, mock_pull, mock_embeddings, mock_llm, mock_loader,
        mock_splitter, mock_chroma, mock_df, mock_join, mock_time, mock_exists, mock_multi_query, mock_config
    ):
        """Test the full pipeline execution."""
        # Setup mocks
        mock_exists.return_value = True  # Mock file exists
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0]  # indexing start, indexing end, q1 start, q1 end
        mock_join.return_value = "results/test.csv"

        mock_docs = [MagicMock()]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance

        mock_chunks = [MagicMock(), MagicMock()]
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance

        mock_vector_db = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [MagicMock(page_content="Context")]
        mock_vector_db.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vector_db

        # Mock MultiQueryRetriever
        mock_multi_query_instance = MagicMock()
        mock_multi_query.from_llm.return_value = mock_multi_query_instance

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Test answer"

        # Create runner and patch chain creation
        runner = RAGRunner(mock_config)
        runner.create_chain = MagicMock(return_value=mock_chain)

        # Run pipeline
        questions = ["What is Apache Iceberg?"]
        result = runner.run(questions)

        # Verify calls
        assert mock_loader.called
        assert mock_splitter.called
        assert mock_chroma.from_documents.called
        assert mock_chain.invoke.called
        assert mock_df.called  # DataFrame should be created


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_config_validation_negative_chunk_size(self):
        """Test that negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RAGConfig(chunk_size=-100)

    def test_config_validation_negative_chunk_overlap(self):
        """Test that negative chunk_overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            RAGConfig(chunk_overlap=-10)

    def test_config_validation_overlap_greater_than_size(self):
        """Test that overlap >= size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RAGConfig(chunk_size=100, chunk_overlap=100)

    def test_config_validation_zero_retrieval_k(self):
        """Test that retrieval_k <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="retrieval_k must be positive"):
            RAGConfig(retrieval_k=0)

    def test_config_validation_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="llm_provider must be one of"):
            RAGConfig(llm_provider="invalid_provider")

    def test_config_validation_invalid_strategy(self):
        """Test that invalid retrieval_strategy raises ValueError."""
        with pytest.raises(ValueError, match="retrieval_strategy must be one of"):
            RAGConfig(retrieval_strategy="invalid_strategy")

    @patch('rag.rag_runner.os.path.exists')
    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_ingest_file_not_found(self, mock_pull, mock_embeddings, mock_llm, mock_exists, mock_config):
        """Test that ingest raises FileNotFoundError when file doesn't exist."""
        mock_exists.return_value = False

        runner = RAGRunner(mock_config)

        with pytest.raises(FileNotFoundError, match="Document not found at path"):
            runner.ingest()

    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_split_documents_empty_list(self, mock_pull, mock_embeddings, mock_llm, mock_config):
        """Test that split_documents raises ValueError for empty document list."""
        runner = RAGRunner(mock_config)

        with pytest.raises(ValueError, match="Cannot split empty documents list"):
            runner.split_documents([])

    @patch('rag.rag_runner.ChatOllama')
    @patch('rag.rag_runner.OllamaEmbeddings')
    @patch('ollama.pull')
    def test_create_vector_db_empty_chunks(self, mock_pull, mock_embeddings, mock_llm, mock_config):
        """Test that create_vector_db raises ValueError for empty chunks list."""
        runner = RAGRunner(mock_config)

        with pytest.raises(ValueError, match="Cannot create vector database from empty chunks list"):
            runner.create_vector_db([])


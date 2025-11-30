"""RAG pipeline runner module."""

import ollama
import time
import os
from typing import List, Dict, Any

import pandas as pd
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from rag_config import RAGConfig


class RAGRunner:
    """Main runner class for RAG pipeline execution.
    
    Handles the complete RAG pipeline: document ingestion, chunking,
    vector database creation, retrieval, and question answering.
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """Initialize the RAG runner with configuration.
        
        Args:
            config: RAGConfig instance with all pipeline parameters
        """
        self.config: RAGConfig = config
        self.llm: ChatOllama = self._create_llm()
        self.embeddings: OllamaEmbeddings = self._create_embeddings()
    
    def _create_llm(self) -> ChatOllama:
        """Create LLM instance based on config.
        
        Currently supports local Ollama only. To add cloud support:
        - Add model_type and model_provider to RAGConfig
        - Add conditional logic here for cloud providers (OpenAI, Anthropic, etc.)
        
        Returns:
            ChatOllama instance configured with the model name
        """
        return ChatOllama(model=self.config.model_name)
    
    def _create_embeddings(self) -> OllamaEmbeddings:
        """Create embeddings instance based on config.
        
        Currently supports local Ollama only. To add cloud support:
        - Add embedding_type and embedding_provider to RAGConfig
        - Add conditional logic here for cloud providers (OpenAI, etc.)
        
        Returns:
            OllamaEmbeddings instance configured with the embedding model
        """
        # Pull the embedding model if not already available
        ollama.pull(self.config.embedding_model)
        return OllamaEmbeddings(model=self.config.embedding_model)
    
    def ingest(self) -> List[Any]:
        """Load PDF documents from the configured path.
        
        Returns:
            List of document objects loaded from the PDF
        """
        loader = UnstructuredPDFLoader(file_path=self.config.doc_path)
        return loader.load()
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of document objects to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_db(self, chunks: List[Any]) -> Chroma:
        """Create a vector database from document chunks.
        
        Args:
            chunks: List of document chunks to embed and store
            
        Returns:
            Chroma vector store instance
        """
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.config.vector_store_name,
            persist_directory=self.config.persist_directory,
        )
        return vector_db
    
    def create_retriever(self, vector_db: Chroma) -> MultiQueryRetriever:
        """Create a multi-query retriever.
        
        Args:
            vector_db: Chroma vector store instance
            
        Returns:
            MultiQueryRetriever instance configured with the vector store
        """
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
        )
        base_retriever = vector_db.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        retriever = MultiQueryRetriever.from_llm(
            base_retriever, self.llm, prompt=QUERY_PROMPT
        )
        return retriever
    
    def create_chain(self, retriever: MultiQueryRetriever) -> Any:
        """Create the RAG chain for question answering.
        
        Args:
            retriever: MultiQueryRetriever instance
            
        Returns:
            LangChain chain that combines retrieval and generation
        """
        template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def run(self, questions: List[str]) -> pd.DataFrame:
        """Run the full RAG pipeline with given questions.
        
        Executes the complete pipeline: ingestion, chunking, vector DB creation,
        retrieval setup, and question answering. Generates a CSV report with results.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            DataFrame containing all results and metrics
        """
        # Indexing
        print("Indexing...")
        start = time.time()
        data = self.ingest()
        chunks = self.split_documents(data)
        vector_db = self.create_vector_db(chunks)
        indexing_time = time.time() - start
        
        # Create chain
        retriever = self.create_retriever(vector_db)
        chain = self.create_chain(retriever)
        
        # Run questions
        print("\nRunning questions...")
        report_data: List[Dict[str, Any]] = []
        
        for question in questions:
            start = time.time()
            response: str = chain.invoke(input=question)
            elapsed = time.time() - start
            
            report_data.append({
                "Question": question,
                "Response": response,
                "Response Time (s)": round(elapsed, 2),
                "Accuracy": "[To be evaluated]",
                "LLM Model": self.config.model_name,
                "Embedding Model": self.config.embedding_model,
                "Vector Store": "Chroma",
                "Persist Directory": self.config.persist_directory or "None (in-memory)",
                "Chunk Size": self.config.chunk_size,
                "Chunk Overlap": self.config.chunk_overlap,
                "Retrieval K": self.config.retrieval_k,
                "Number of Chunks": len(chunks),
                "Indexing Time (s)": round(indexing_time, 2),
            })
            print('Question:', question)
            print('Answer:', response)
            print()
        
        # Calculate and add average response time
        avg_time = round(sum(r["Response Time (s)"] for r in report_data) / len(report_data), 2)
        for row in report_data:
            row["Average Response Time (s)"] = avg_time
        
        report_df = pd.DataFrame(report_data)
        
        # Save report as single CSV with parameters in filename
        filename = f"chunk{self.config.chunk_size}_overlap{self.config.chunk_overlap}_k{self.config.retrieval_k}_{self.config.model_name}_{self.config.embedding_model}.csv"
        filepath = os.path.join("results", filename)
        report_df.to_csv(filepath, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("BASELINE REPORT")
        print("="*60)
        print(report_df.to_string(index=False))
        print(f"\nReport saved to: {filepath}")
        
        return report_df


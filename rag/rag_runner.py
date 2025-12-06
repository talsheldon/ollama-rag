"""RAG pipeline runner module."""

import ollama
import time
import os
import json
import re
from typing import List, Dict, Any, Union

import pandas as pd
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from rag.rag_config import RAGConfig


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
        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
    
    def _create_llm(self) -> Union[ChatOllama, ChatOpenAI]:
        """Create LLM instance based on config.

        Supports local Ollama or cloud OpenAI.

        Returns:
            LLM instance (ChatOllama or ChatOpenAI)
        """
        if self.config.llm_provider == "openai":
            return ChatOpenAI(model=self.config.openai_model)
        else:
            return ChatOllama(model=self.config.model_name)

    def _create_embeddings(self) -> Union[OllamaEmbeddings, OpenAIEmbeddings]:
        """Create embeddings instance based on config.

        Supports local Ollama or cloud OpenAI.

        Returns:
            Embeddings instance (OllamaEmbeddings or OpenAIEmbeddings)
        """
        if self.config.embedding_provider == "openai":
            return OpenAIEmbeddings(model=self.config.openai_embedding_model)
        else:
            # Pull the embedding model if not already available
            ollama.pull(self.config.embedding_model)
            return OllamaEmbeddings(model=self.config.embedding_model)

    def ingest(self) -> List[Document]:
        """Load PDF documents from the configured path.

        Returns:
            List of document objects loaded from the PDF

        Raises:
            FileNotFoundError: If the document path does not exist
        """
        if not os.path.exists(self.config.doc_path):
            raise FileNotFoundError(f"Document not found at path: {self.config.doc_path}")
        loader = UnstructuredPDFLoader(file_path=self.config.doc_path)
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of document objects to split

        Returns:
            List of document chunks

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot split empty documents list")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_db(self, chunks: List[Document]) -> Chroma:
        """Create a vector database from document chunks.

        Args:
            chunks: List of document chunks to embed and store

        Returns:
            Chroma vector store instance

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot create vector database from empty chunks list")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.config.vector_store_name,
            persist_directory=self.config.persist_directory,
        )
        return vector_db

    def create_retriever(self, vector_db: Chroma) -> Union[MultiQueryRetriever, BaseRetriever]:
        """Create a retriever based on retrieval strategy.

        Args:
            vector_db: Chroma vector store instance

        Returns:
            Retriever instance (MultiQueryRetriever or basic retriever for reranking)
        """
        if self.config.retrieval_strategy == "reranking":
            # For reranking: use basic retriever with k=10, reranking happens in chain
            base_retriever = vector_db.as_retriever(
                search_kwargs={"k": 10}  # Retrieve more for reranking
            )
            return base_retriever
        else:
            # Basic: use MultiQueryRetriever
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

    def create_chain(self, retriever: Union[MultiQueryRetriever, BaseRetriever]) -> Runnable:
        """Create the RAG chain for question answering.

        Args:
            retriever: Retriever instance (MultiQueryRetriever or basic retriever)

        Returns:
            LangChain chain that combines retrieval and generation
        """
        if self.config.retrieval_strategy == "reranking":
            def rerank_and_answer(question: str) -> str:
                
                # Step 1: Retrieve documents from vector store
                docs = retriever.invoke(question)
                if not docs:
                    return "No relevant documents found."
                
                # Step 2: Rerank all documents in single LLM call
                # Format documents with indices for the LLM to rank
                docs_text = "\n".join([f"[{i}] {doc.page_content[:300]}" for i, doc in enumerate(docs)])
                rerank_prompt = ChatPromptTemplate.from_template(
                    "Rank by relevance. Return ONLY JSON array of top {top_k} indices: [0,2,5]\n\nQ: {question}\nDocs:\n{docs_text}\n\nJSON:"
                )
                result = (rerank_prompt | self.llm | StrOutputParser()).invoke({
                    "question": question,
                    "docs_text": docs_text,
                    "top_k": self.config.rerank_k
                })
                
                # Step 3: Parse JSON array from LLM response
                # Try direct JSON parse first, fallback to regex if LLM added extra text
                try:
                    top_indices = json.loads(result)
                except json.JSONDecodeError:
                    # LLM sometimes adds extra text, so extract JSON array with regex
                    match = re.search(r'\[[\d\s,]+\]', result)
                    top_indices = json.loads(match.group()) if match else list(range(min(self.config.rerank_k, len(docs))))
                
                # Ensure indices are integers and valid
                top_indices = [int(i) for i in top_indices if isinstance(i, (int, str)) and str(i).isdigit()]
                top_indices = [i for i in top_indices if 0 <= i < len(docs)][:self.config.rerank_k]
                if not top_indices:
                    top_indices = list(range(min(self.config.rerank_k, len(docs))))
                
                # Step 4: Get top documents and generate answer
                top_docs = [docs[i] for i in top_indices]
                context = "\n\n".join([doc.page_content for doc in top_docs])
                answer_prompt = ChatPromptTemplate.from_template("Answer based ONLY on:\n{context}\n\nQ: {question}")
                return (answer_prompt | self.llm | StrOutputParser()).invoke({"context": context, "question": question})
            
            return RunnablePassthrough() | rerank_and_answer
        else:
            # Basic: standard chain
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
                "Retrieval Strategy": self.config.retrieval_strategy,
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
        llm_model = self.config.model_name if self.config.llm_provider == "ollama" else self.config.openai_model
        embedding_model = self.config.embedding_model if self.config.embedding_provider == "ollama" else self.config.openai_embedding_model
        provider_tag = f"{self.config.llm_provider}-{self.config.embedding_provider}"
        strategy_tag = f"_{self.config.retrieval_strategy}" if self.config.retrieval_strategy != "basic" else ""
        filename = f"chunk{self.config.chunk_size}_overlap{self.config.chunk_overlap}_k{self.config.retrieval_k}{strategy_tag}_{provider_tag}_{llm_model}_{embedding_model}.csv"
        filepath = os.path.join("results", filename)
        report_df.to_csv(filepath, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("BASELINE REPORT")
        print("="*60)
        print(report_df.to_string(index=False))
        print(f"\nReport saved to: {filepath}")
        
        return report_df


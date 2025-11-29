import ollama
import time
import os
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

DOC_PATH = "data/iceberg-specs.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = None
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
RETRIEVAL_K = 5

def ingest_pdf(doc_path: str = DOC_PATH):
    loader = UnstructuredPDFLoader(file_path=doc_path)
    return loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

def create_vector_db(chunks):
    ollama.pull(EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

def create_retriever(vector_db, llm, k=RETRIEVAL_K):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )
    base_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return MultiQueryRetriever.from_llm(base_retriever, llm, prompt=QUERY_PROMPT)

def create_chain(retriever, llm):
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    questions = [
        "What is Apache Iceberg? Explain in short.",
        "How does Iceberg ensure that two writers do not overwrite each others ingestion results?",
        "How to access data that was deleted in a newer snapshot?",
        "What happens if a writer attempts to commit based on an old snapshot?",
    ]
    
    # Indexing
    print("Indexing...")
    start = time.time()
    data = ingest_pdf()
    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)
    indexing_time = time.time() - start
    
    # Create chain
    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm, k=RETRIEVAL_K)
    chain = create_chain(retriever, llm)
    
    # Run questions
    print("\nRunning questions...")
    report_data = []
    
    for question in questions:
        start = time.time()
        response = chain.invoke(input=question)
        elapsed = time.time() - start
        
        report_data.append({
            "Question": question,
            "Response": response,
            "Response Time (s)": round(elapsed, 2),
            "Accuracy": "[To be evaluated]",
            "LLM Model": MODEL_NAME,
            "Embedding Model": EMBEDDING_MODEL,
            "Vector Store": "Chroma",
            "Persist Directory": PERSIST_DIRECTORY or "None (in-memory)",
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP,
            "Retrieval K": RETRIEVAL_K,
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
    filename = f"chunk{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}_{MODEL_NAME}_{EMBEDDING_MODEL}.csv"
    filepath = os.path.join("results", filename)
    report_df.to_csv(filepath, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE REPORT")
    print("="*60)
    print(report_df.to_string(index=False))
    print(f"\nReport saved to: {filepath}")

if __name__ == "__main__":
    main()

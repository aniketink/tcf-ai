import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_pdfs(source_folder: str):
    """
    Reads PDFs from the given folder, splits them into chunks, 
    embeds them, and stores the vectors in ChromaDB.
    """
    folder_path = Path(source_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{source_folder}' does not exist.")
        return

    # Find all PDFs in the folder
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{source_folder}'.")
        return

    print(f"Found {len(pdf_files)} PDFs in '{source_folder}'.")
    
    documents: List[Document] = []
    
    # 1. Load Documents
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}...")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Ensure metadata includes the source filename cleanly
            for doc in docs:
                doc.metadata['source'] = pdf_file.name
                
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")

    if not documents:
        print("No content could be extracted from the PDFs.")
        return

    print(f"Total pages extracted: {len(documents)}")

    # 2. Text Splitting
    # Split into 500-token/character chunks with 100-token/character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    print("Splitting text into semantically meaningful chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No text chunks generated. Aborting ingestion.")
        return

    # 3. Embeddings & Storage
    print(f"Initializing HuggingFace embedding model ({EMBEDDING_MODEL})...")
    # This will download the model locally on first run
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Generating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"Ingestion complete. Vector database saved to '{CHROMA_DB_DIR}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF research papers into the knowledge base.")
    parser.add_argument("folder", type=str, help="Path to the folder containing PDF files.")
    args = parser.parse_args()
    
    ingest_pdfs(args.folder)

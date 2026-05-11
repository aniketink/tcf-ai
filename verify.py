import os
import time
import argparse
from typing import List, Literal
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table

load_dotenv()

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Data Models (Pydantic schemas for GPT-4o structured output) ---
class ExtractedClaims(BaseModel):
    claims: List[str] = Field(description="A list of distinct factual claims and scientific conclusions extracted from the paper.")

class VerdictResult(BaseModel):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"] = Field(
        description="The verdict on whether the claim is supported, contradicted, or inconclusive based on the evidence."
    )
    explanation: str = Field(description="A short explanation citing the exact evidence to justify the verdict.")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from all pages of the uploaded PDF."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join(doc.page_content for doc in docs)
    return text

def verify_paper(pdf_path: str):
    """
    Main pipeline to verify a new uploaded paper against the vector knowledge base.
    """
    console = Console()
    
    # Validation checks
    if not os.environ.get("GOOGLE_API_KEY"):
        console.print("[bold red]Error: GOOGLE_API_KEY is not set in the environment or .env file.[/bold red]")
        console.print("Get a free key at: https://aistudio.google.com/app/apikey")
        return
        
    if not Path(CHROMA_DB_DIR).exists():
        console.print(f"[bold red]Error: ChromaDB directory '{CHROMA_DB_DIR}' not found. Please run ingest.py first.[/bold red]")
        return

    # 1. Load User PDF
    console.print(f"[bold blue]Loading uploaded paper:[/bold blue] {pdf_path}")
    try:
        full_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        console.print(f"[bold red]Failed to read PDF: {e}[/bold red]")
        return

    if not full_text.strip():
        console.print("[bold red]Error: No text could be extracted from the PDF.[/bold red]")
        return

    # Initialize LLM (Using 'gemini-flash-latest' for best availability)
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # 2. Extract Claims
    console.print("[bold yellow]Extracting key claims from the paper using Gemini 2.0 Flash...[/bold yellow]")
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert scientific researcher. Analyze the following research paper text and extract all major factual claims and scientific conclusions. Avoid overly broad or trivial statements. Focus on specific methodologies, results, and scientific findings."),
        ("human", "{paper_text}")
    ])
    
    extraction_chain = extraction_prompt | llm.with_structured_output(ExtractedClaims)
    
    # Extract structural claims from full text
    extracted = extraction_chain.invoke({"paper_text": full_text})
    claims = extracted.claims
    
    if not claims:
        console.print("[bold red]No claims were extracted from the document.[/bold red]")
        return
        
    console.print(f"Extracted [bold green]{len(claims)}[/bold green] claims. Starting verification...\n")

    # 3. Initialize Vector DB for Verification
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    # Configure retriever to fetch top 3 most similar chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Verification Prompt
    verification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict, objective fact-checking AI.
You will be provided with a CLAIM and some retrieved EVIDENCE (chunks of text from a knowledge base).
Your task is to determine if the CLAIM is SUPPORTED, CONTRADICTED, or INCONCLUSIVE based strictly on the provided EVIDENCE.

CRITICAL RULES:
1. Do NOT use outside knowledge. Rely ONLY on the provided EVIDENCE. If the EVIDENCE is empty or irrelevant, you MUST output INCONCLUSIVE.
2. Provide a short, precise explanation citing the evidence to justify your verdict.
3. If the evidence does not clearly support or contradict the claim, output INCONCLUSIVE.
4. Hallucination is strictly forbidden."""),
        ("human", "CLAIM: {claim}\n\nEVIDENCE:\n{evidence}")
    ])
    
    verification_chain = verification_prompt | llm.with_structured_output(VerdictResult)

    # Setup Rich Table for Output
    table = Table(title=f"RAG Verification Results: {Path(pdf_path).name}", show_lines=True)
    table.add_column("Claim", style="cyan", max_width=40)
    table.add_column("Verdict", justify="center", max_width=15)
    table.add_column("Explanation", style="magenta", max_width=50)
    table.add_column("Sources", style="dim", max_width=30)

    # 4. Verify Each Claim individually
    for idx, claim in enumerate(claims, 1):
        console.print(f"Verifying claim {idx}/{len(claims)}...")
        
        # Retrieve top 3 relevant chunks
        docs = retriever.invoke(claim)
        
        if not docs:
            evidence_text = "No evidence found in the knowledge base."
            sources = "N/A"
        else:
            # Construct evidence string with source metadata
            evidence_text = "\n\n".join([f"Source ({doc.metadata.get('source', 'Unknown')}): {doc.page_content}" for doc in docs])
            unique_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
            sources = ", ".join(unique_sources)

        # Evaluate via GPT-4o
        try:
            result = verification_chain.invoke({
                "claim": claim,
                "evidence": evidence_text
            })
            verdict = result.verdict
            explanation = result.explanation
        except Exception as e:
            verdict = "ERROR"
            explanation = f"Failed to verify due to LLM error: {str(e)}"
            
        # Add rich styling for verdicts
        if verdict == "SUPPORTED":
            verdict_styled = "[bold green]SUPPORTED[/bold green]"
        elif verdict == "CONTRADICTED":
            verdict_styled = "[bold red]CONTRADICTED[/bold red]"
        elif verdict == "INCONCLUSIVE":
            verdict_styled = "[bold yellow]INCONCLUSIVE[/bold yellow]"
        else:
            verdict_styled = f"[bold red]{verdict}[/bold red]"

        # Add to results table
        table.add_row(claim, verdict_styled, explanation, sources)
        
        # Respect API Rate limits (Free tier has 15 RPM limit)
        if idx < len(claims):
            console.print("[dim]Waiting 5 seconds to respect free API rate limits...[/dim]")
            time.sleep(5)

    # 5. Display the final results table
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a research paper's claims against the knowledge base.")
    parser.add_argument("pdf_file", type=str, help="Path to the PDF file to verify.")
    args = parser.parse_args()
    
    verify_paper(args.pdf_file)

import os
import time
import argparse
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.table import Table

load_dotenv()

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RETRIEVAL_K = 20      # Initial ANN retrieval depth
RERANK_TOP_K = 5       # Chunks to keep after reranking
CRITIQUE_ROUNDS = 2    # Self-reflection iterations
API_RATE_LIMIT = 5     # Seconds between LLM calls

# --- Data Models ---
class ExtractedClaims(BaseModel):
    claims: List[str] = Field(
        description="A list of distinct factual claims and scientific conclusions extracted from the paper."
    )

class CritiqueResult(BaseModel):
    sub_claims: List[str] = Field(
        description="A list of specific, verifiable sub-claims derived from the original claim given the evidence."
    )
    focus_areas: List[str] = Field(
        description="Areas where evidence is weak or ambiguous and needs more scrutiny."
    )

class VerdictResult(BaseModel):
    verdict: str = Field(
        description="SUPPORTED, CONTRADICTED, or INCONCLUSIVE"
    )
    explanation: str = Field(
        description="Short explanation citing exact evidence."
    )

class VerdictResultList(BaseModel):
    verdicts: List[VerdictResult] = Field(
        description="A list of verdict results, one per sub-claim."
    )

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


def retrieve_and_rerank(query: str, vectorstore: Chroma, cross_encoder: CrossEncoder, k: int = RETRIEVAL_K) -> List[Document]:
    """ANN search + cross-encoder reranking."""
    raw_docs = vectorstore.similarity_search(query, k=k)
    if not raw_docs:
        return []

    texts = [doc.page_content for doc in raw_docs]
    pairs = [(query, text) for text in texts]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(raw_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for doc, score in scored_docs[:RERANK_TOP_K]:
        doc.metadata["rerank_score"] = float(score)
        reranked.append(doc)
    return reranked


def critique_evidence(llm: ChatOllama, claim: str, evidence_chunks: List[Document]) -> CritiqueResult:
    """Self-reflection: derive sub-claims and identify weak areas given evidence."""
    evidence_text = "\n\n".join([
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(evidence_chunks)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a rigorous scientific fact-checker. Given a CLAIM and retrieved EVIDENCE, "
         "break it down into specific sub-claims that can each be independently verified. "
         "Also identify areas where the evidence is ambiguous, incomplete, or contradictory."),
        ("human",
         "CLAIM: {claim}\n\nEVIDENCE:\n{evidence}")
    ])

    chain = prompt | llm.with_structured_output(CritiqueResult)
    return chain.invoke({"claim": claim, "evidence": evidence_text})


def verify_sub_claims(llm: ChatOllama, sub_claims: List[str], evidence_chunks: List[Document]) -> List[VerdictResult]:
    """Verify each sub-claim against the evidence chunks."""
    evidence_text = "\n\n".join([
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(evidence_chunks)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict, objective fact-checking AI.\n"
         "Given a list of SUB-CLAIMS and retrieved EVIDENCE, determine whether each sub-claim is "
         "SUPPORTED, CONTRADICTED, or INCONCLUSIVE strictly based on the provided EVIDENCE.\n\n"
         "CRITICAL RULES:\n"
         "1. Do NOT use outside knowledge. Rely ONLY on the provided EVIDENCE.\n"
         "2. If evidence is empty or irrelevant, output INCONCLUSIVE.\n"
         "3. Cite the evidence number in your explanation.\n"
         "4. Hallucination is strictly forbidden."),
        ("human",
         "SUB-CLAIMS:\n{claims}\n\nEVIDENCE:\n{evidence}")
    ])

    chain = prompt | llm.with_structured_output(VerdictResultList)
    result = chain.invoke({
        "claims": "\n".join([f"- {sc}" for sc in sub_claims]),
        "evidence": evidence_text
    })
    return result.verdicts


def aggregate_verdicts(sub_verdicts: List[VerdictResult]) -> VerdictResult:
    """Roll up sub-claim verdicts into a single verdict for the parent claim."""
    supported = sum(1 for v in sub_verdicts if v.verdict == "SUPPORTED")
    contradicted = sum(1 for v in sub_verdicts if v.verdict == "CONTRADICTED")
    inconclusive = sum(1 for v in sub_verdicts if v.verdict == "INCONCLUSIVE")
    total = len(sub_verdicts)

    explanations = [v.explanation for v in sub_verdicts]

    if contradicted > 0:
        verdict = "CONTRADICTED"
        explanation = f"{contradicted}/{total} sub-claims contradicted. " + " | ".join(explanations)
    elif supported == total:
        verdict = "SUPPORTED"
        explanation = f"All {total}/{total} sub-claims supported. " + " | ".join(explanations)
    elif supported > total / 2:
        verdict = "MOSTLY_SUPPORTED"
        explanation = f"{supported}/{total} supported, {inconclusive} inconclusive. " + " | ".join(explanations)
    else:
        verdict = "INCONCLUSIVE"
        explanation = f"{supported}/{total} supported, {inconclusive} inconclusive. " + " | ".join(explanations)

    return VerdictResult(verdict=verdict, explanation=explanation)


def verify_paper(pdf_path: str, verbose: bool = False):
    console = Console()

    # --- Validation ---
    # no API key needed for Ollama
    if not Path(CHROMA_DB_DIR).exists():
        console.print(f"[bold red]Error: ChromaDB not found at '{CHROMA_DB_DIR}'. Run ingest.py first.[/bold red]")
        return

    # --- Load PDF ---
    console.print(f"[bold blue]Loading:[/bold blue] {pdf_path}")
    try:
        full_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        console.print(f"[bold red]Failed to read PDF: {e}[/bold red]")
        return
    if not full_text.strip():
        console.print("[bold red]No text extracted from PDF.[/bold red]")
        return

    # --- Initialize Models ---
    llm = ChatOllama(model="llama3.2:3b", temperature=0, keep_alive="10m")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    console.print("[bold blue]Loading reranker model...[/bold blue]")
    cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu")

    # --- Extract Claims ---
    console.print("[bold yellow]Extracting claims via LLM...[/bold yellow]")
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert scientific researcher. Analyze the following paper and extract all major "
         "factual claims and scientific conclusions. Avoid trivial statements. Focus on specific "
         "methodologies, results, and findings."),
        ("human", "{paper_text}")
    ])
    extraction_chain = extraction_prompt | llm.with_structured_output(ExtractedClaims)
    extracted = extraction_chain.invoke({"paper_text": full_text})
    claims = extracted.claims

    if not claims:
        console.print("[bold red]No claims extracted.[/bold red]")
        return

    console.print(f"[bold green]Extracted {len(claims)} claims.[/bold green]\n")

    # --- Verification Loop ---
    table = Table(title=f"RAG Verification: {Path(pdf_path).name}", show_lines=True)
    table.add_column("Claim", style="cyan", max_width=40)
    table.add_column("Verdict", justify="center", max_width=15)
    table.add_column("Explanation", style="magenta", max_width=50)
    table.add_column("Sources", style="dim", max_width=25)

    for idx, claim in enumerate(claims, 1):
        console.print(f"[bold cyan]Claim {idx}/{len(claims)}:[/bold cyan] {claim[:80]}{'...' if len(claim) > 80 else ''}")

        # 1. Retrieve + Rerank
        if verbose:
            console.print("  [dim]Retrieving & reranking...[/dim]")
        evidence_chunks = retrieve_and_rerank(claim, vectorstore, cross_encoder, k=RETRIEVAL_K)

        if not evidence_chunks:
            table.add_row(claim, "[bold yellow]INCONCLUSIVE[/bold yellow]",
                         "No relevant evidence found in knowledge base.", "N/A")
            continue

        sources = list(set(doc.metadata.get("source", "Unknown") for doc in evidence_chunks))
        if verbose:
            console.print(f"  [dim]Top evidence sources: {sources}[/dim]")

        # 2. Self-Reflection / Critique Loop
        sub_claims = []
        focus_areas = []
        current_chunks = evidence_chunks

        for round_num in range(CRITIQUE_ROUNDS):
            if verbose:
                console.print(f"  [dim]Critique round {round_num + 1}...[/dim]")

            critique = critique_evidence(llm, claim, current_chunks)
            sub_claims = critique.sub_claims
            focus_areas = critique.focus_areas

            # If we have focus areas, do one more targeted retrieval
            if round_num < CRITIQUE_ROUNDS - 1 and focus_areas:
                additional_queries = focus_areas[:2]  # Top 2 focus areas
                for q in additional_queries:
                    extra = retrieve_and_rerank(q, vectorstore, cross_encoder, k=RETRIEVAL_K)
                    # Merge, deduplicate
                    existing_ids = {c.page_content[:50] for c in current_chunks}
                    for doc in extra:
                        if doc.page_content[:50] not in existing_ids:
                            current_chunks.append(doc)
                            existing_ids.add(doc.page_content[:50])

            time.sleep(API_RATE_LIMIT)

        if verbose:
            console.print(f"  [dim]Sub-claims: {len(sub_claims)}, Focus areas: {len(focus_areas)}[/dim]")

        # 3. Verify Sub-Claims
        if sub_claims:
            if verbose:
                console.print(f"  [dim]Verifying {len(sub_claims)} sub-claims...[/dim]")
            sub_verdicts = verify_sub_claims(llm, sub_claims, current_chunks)
        else:
            # Fallback: verify the original claim directly
            sub_verdicts = [VerdictResult(verdict="INCONCLUSIVE", explanation="No sub-claims derived.")]

        time.sleep(API_RATE_LIMIT)

        # 4. Aggregate
        result = aggregate_verdicts(sub_verdicts)

        # 5. Style verdict
        verdict_map = {
            "SUPPORTED": "[bold green]SUPPORTED[/bold green]",
            "CONTRADICTED": "[bold red]CONTRADICTED[/bold red]",
            "INCONCLUSIVE": "[bold yellow]INCONCLUSIVE[/bold yellow]",
            "MOSTLY_SUPPORTED": "[bold green]MOSTLY_SUPPORTED[/bold green]",
        }
        verdict_styled = verdict_map.get(result.verdict, f"[bold red]{result.verdict}[/bold red]")

        table.add_row(
            claim[:100] + ("..." if len(claim) > 100 else ""),
            verdict_styled,
            result.explanation[:150] + ("..." if len(result.explanation) > 150 else ""),
            ", ".join(sources)[:100]
        )

        console.print(f"  [bold]→ {result.verdict}[/bold]\n")
        if idx < len(claims):
            console.print("[dim]Waiting for API rate limit...[/dim]")
            time.sleep(API_RATE_LIMIT)

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RAG verification with reranking + self-reflection.")
    parser.add_argument("pdf_file", type=str, help="Path to the PDF to verify.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress.")
    args = parser.parse_args()
    verify_paper(args.pdf_file, verbose=args.verbose)

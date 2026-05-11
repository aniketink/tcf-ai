#!/usr/bin/env python3
"""
Auto-Corpus Engine — Automatically expands the corpus with relevant arXiv papers.
Monitors verification results; if evidence is weak, downloads relevant papers.
"""
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_verify")

SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_arxiv.py"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_v2.py"
INGEST_SCRIPT = SCRIPT_DIR / "ingest.py"


def get_chunk_count():
    if not CHROMA_DIR.exists():
        return 0
    db = CHROMA_DIR / "chroma.sqlite3"
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_corpus_pdfs():
    if not CORPUS_DIR.exists():
        return []
    return list(CORPUS_DIR.glob("*.pdf"))


def extract_topics_from_pdf(pdf_path: Path) -> list[str]:
    """Extract likely topic keywords from a PDF's filename."""
    import re

    topics = []
    name = pdf_path.stem.lower()
    # clean filename into search terms
    name = re.sub(r'[_\-]', ' ', name)
    name = re.sub(r'\b(paper|arxiv|proceedings|author|preprint|ieee|acm|nips|icml|iclr)\b', '', name)
    name = re.sub(r'\d{4}', '', name)
    name = re.sub(r'[^\w\s]', ' ', name)
    name = name.strip()

    if name:
        topics.append(name)

    # also try to extract common ML/AI keywords from name
    keywords = [
        # Medical imaging
        "chest X-ray pneumonia COVID tuberculosis",
        "brain MRI tumor segmentation",
        "histopathology cancer WSI",
        "retinal OCT fundus diabetic",
        "CT scan lung nodule",
        "medical imaging CNN transformer",
        # Clinical NLP & EHRs
        "clinical NLP EHR electronic health records",
        "medical named entity recognition",
        "deidentification privacy clinical notes",
        "clinical report generation",
        # Foundation models
        "medical LLM GPT clinical diagnosis",
        "biomedical QA MedQA PubMedBERT",
        "vision language model medical",
        "foundation model CheXbert RadGraph",
        # Fine-tuning methods
        "LoRA fine-tuning medical",
        "parameter efficient PEFT clinical",
        "instruction tuning medical domain",
        "RAG clinical decision retrieval",
        # Multi-modal & emerging
        "multimodal imaging text",
        "federated learning medical privacy",
        "self-supervised medical imaging",
        "zero-shot medical classification",
        "drug discovery molecular property",
        # General AI methods
        "transformer attention mechanism",
        "knowledge distillation clinical",
        "cancer prognosis survival analysis",
        "sepsis ICU early warning prediction",
    ]
    name_words = set(name.split())
    for kw in keywords:
        kw_words = set(kw.split())
        if name_words & kw_words:  # intersection
            topics.append(kw)

    return topics[:5]


def build_seed_corpus():
    """Download a foundational set of papers covering core ML/AI topics."""
    seed_topics = [
        # ── Medical Imaging ──
        ("chest X-ray pneumonia tuberculosis COVID deep learning diagnosis", 10),
        ("brain tumor MRI segmentation U-Net deep learning", 8),
        ("histopathology cancer detection CNN transformer pathology", 8),
        ("retinal OCT fundus diabetic retinopathy imaging AI", 8),
        ("skin lesion melanoma classification dermoscopy CNN", 8),
        # ── Clinical NLP & EHRs ──
        ("clinical NLP electronic health records EHR BERT", 10),
        ("medical named entity recognition clinical text extraction", 8),
        ("clinical report generation language model GPT radiology", 8),
        # ── Medical LLMs & Foundation Models ──
        ("medical language model GPT-4 clinical diagnosis PubMed", 10),
        ("biomedical question answering MedQA PubMedBERT", 8),
        ("vision language model medical imaging report generation", 8),
        # ── Diagnosis & Prognosis ──
        ("cancer diagnosis machine learning prognosis survival", 10),
        ("sepsis prediction ICU early warning deep learning", 6),
        ("diabetes retinopathy screening deep learning", 6),
        # ── Fine-tuning & Efficient Methods ──
        ("LoRA low-rank adaptation medical language model fine-tuning", 10),
        ("parameter efficient fine-tuning PEFT clinical NLP", 8),
        ("instruction tuning medical domain LLM alignment", 8),
        ("RAG retrieval augmented clinical decision support", 8),
        # ── Multi-modal, Self-supervised & Privacy ──
        ("multimodal medical imaging text report alignment", 8),
        ("self-supervised medical image representation learning", 8),
        ("federated learning medical data privacy", 6),
        ("AI drug discovery molecular property prediction", 6),
        # ── Benchmarks ──
        ("medical AI benchmark MIMIC ChestX-ray14 dataset", 6),
        ("radiology report NLP RadGraph i2b2 deidentification", 6),
    ]

    print("[*] Building seed corpus with foundational papers...")
    for topic, count in seed_topics:
        print(f"  → Searching: '{topic}'")
        result = subprocess.run(
            [str(VENV_PYTHON), str(DOWNLOAD_SCRIPT),
             "--query", topic, "--max", str(count), "--corpus", str(CORPUS_DIR)],
            capture_output=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        out = result.stdout.decode("utf-8", errors="replace")
        # count downloaded lines
        lines = [l for l in out.split("\n") if "[+]" in l or "[=]" in l]
        print(f"    Downloaded {len(lines)} papers")
        import time; time.sleep(4)  # respect arXiv rate limit

    pdfs = get_corpus_pdfs()
    print(f"[*] Seed corpus complete: {len(pdfs)} papers total")


def auto_ingest():
    """Re-ingest all corpus PDFs into ChromaDB."""
    print("[*] Building knowledge base...")
    result = subprocess.run(
        [str(VENV_PYTHON), str(INGEST_SCRIPT), str(CORPUS_DIR)],
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    chunks = get_chunk_count()
    if chunks > 0:
        print(f"[*] Knowledge base ready: {chunks:,} chunks indexed")
    else:
        print(f"[*] Warning: knowledge base is empty (0 chunks)")
    return chunks


def expand_corpus_for_pdf(pdf_path: Path, verbose: bool = False) -> bool:
    """
    Given a PDF being verified, detect topic and download relevant arXiv papers
    if the corpus seems insufficient.
    Returns True if expansion was performed.
    """
    topics = extract_topics_from_pdf(pdf_path)
    if not topics:
        topics = [pdf_path.stem.lower().replace('_', ' ')]

    print(f"[*] Detected topics from '{pdf_path.name}': {topics}")

    # check current coverage
    corpus_pdfs = get_corpus_pdfs()
    print(f"[*] Current corpus: {len(corpus_pdfs)} papers")

    print("[*] Expanding corpus with relevant arXiv papers...")
    downloaded = 0
    for topic in topics:
        print(f"  → Topic: '{topic}'")
        result = subprocess.run(
            [str(VENV_PYTHON), str(DOWNLOAD_SCRIPT),
             "--query", topic, "--max", "8", "--corpus", str(CORPUS_DIR)],
            capture_output=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        out = result.stdout.decode("utf-8", errors="replace")
        lines = [l for l in out.split("\n") if "[+]" in l or "[=]" in l]
        downloaded += len(lines)
        print(f"    Downloaded {len(lines)} papers")
        time.sleep(4)  # respect arXiv rate limit

    new_pdfs = get_corpus_pdfs()
    print(f"[*] Corpus expanded: {len(new_pdfs)} papers total (+{len(new_pdfs) - len(corpus_pdfs)})")

    if downloaded > 0:
        chunks = auto_ingest()
        print(f"[*] Knowledge base updated: {chunks:,} chunks")
        return True
    return False


def run_verify(pdf_path: Path, verbose: bool = False) -> tuple[str, str]:
    """Run verify_v2.py, return stdout + stderr."""
    cmd = [str(VENV_PYTHON), str(VERIFY_SCRIPT), str(pdf_path)]
    if verbose:
        cmd.append("--verbose")
    result = subprocess.run(
        cmd,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    out = result.stdout.decode("utf-8", errors="replace")
    err = result.stderr.decode("utf-8", errors="replace")
    return out, err


def main():
    print("=" * 60)
    print("RAG-Verify — Autonomous Corpus Expansion")
    print("=" * 60)

    # Step 1: build seed corpus if corpus is empty or tiny
    corpus_pdfs = get_corpus_pdfs()
    if len(corpus_pdfs) < 3:
        print(f"[*] Corpus has only {len(corpus_pdfs)} papers. Building seed corpus...")
        build_seed_corpus()
    else:
        print(f"[*] Corpus already has {len(corpus_pdfs)} papers — skipping seed build.")

    # Step 2: check if KB exists
    if get_chunk_count() == 0:
        print("[*] No knowledge base found. Building now...")
        auto_ingest()
    else:
        print(f"[*] Knowledge base already has {get_chunk_count():,} chunks.")

    # Step 3: if a PDF was passed as argument, verify + auto-expand
    if len(sys.argv) > 2 and sys.argv[1] == "--rebuild-only":
        print("[*] Rebuilding knowledge base...")
        auto_ingest()
        return

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"[!] PDF not found: {pdf_path}")
            sys.exit(1)

        print(f"\n[*] Verifying: {pdf_path.name}")
        out, err = run_verify(pdf_path, verbose=True)

        # quick check: did we get any INCONCLUSIVE results?
        inconclusive_count = out.upper().count("INCONCLUSIVE")
        supported_count = out.upper().count("SUPPORTED")
        contradicted_count = out.upper().count("CONTRADICTED")

        print(f"\n[*] Initial results: {supported_count} SUPPORTED, "
              f"{contradicted_count} CONTRADICTED, {inconclusive_count} INCONCLUSIVE")

        # If more than half are INCONCLUSIVE, expand corpus
        total_claims = supported_count + contradicted_count + inconclusive_count
        if total_claims > 0 and (inconclusive_count / total_claims) > 0.5:
            print(f"\n[!] {inconclusive_count}/{total_claims} claims inconclusive — corpus likely too small.")
            print("[*] Expanding corpus automatically...")

            expanded = expand_corpus_for_pdf(pdf_path)
            if expanded:
                print(f"\n[*] Re-verifying with expanded corpus...")
                out, err = run_verify(pdf_path, verbose=True)
        else:
            print("[*] Corpus coverage looks sufficient.")

        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        print(out)
        if err:
            print("[stderr]", err)
    else:
        print("\nUsage: python auto_corpus.py /path/to/paper.pdf")


if __name__ == "__main__":
    main()

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
        # Cancer genomics & TCGA
        "tumor mutation burden TMB microsatellite instability MSI",
        "BRCA1 BRCA2 breast cancer genetics",
        "TP53 KRAS BRAF oncogene mutation",
        "circulating tumor DNA ctDNA liquid biopsy",
        "precision oncology biomarker HER2 PD-L1",
        "checkpoint inhibitor immunotherapy cancer",
        "CAR-T cell therapy solid tumor",
        "chemotherapy resistance cancer recurrence",
        # Cancer pathology & imaging
        "whole slide image WSI histopathology cancer",
        "tumor grade stage TNM staging",
        "radiomics PET CT MRI cancer imaging",
        "Ki67 proliferative index breast cancer",
        "pathology deep learning cancer detection CNN transformer",
        # Clinical oncology NLP
        "oncology clinical trial structured extraction",
        "OncoKB variant interpretation cancer",
        "biomarker expression immunohistochemistry cancer",
        # Prognosis
        "cancer survival prognosis prediction model",
        "recurrence free survival RFS overall survival OS",
        "liquid biopsy early cancer detection",
        # Foundation models
        "pathology foundation model Virchow cancer",
        "LoRA fine-tuning cancer genomics model",
        "RAG clinical decision oncology support",
        "instruction tuning oncology domain LLM",
        # Drug discovery
        "targeted therapy TKI cancer drug",
        "PARP inhibitor BRCA synthetic lethality",
        "tumor microenvironment immune evasion",
        "metastasis cancer prediction organotropism",
        # General AI
        "transformer attention mechanism",
        "parameter efficient fine-tuning PEFT LoRA",
        "self-supervised medical image representation",
        "federated learning privacy-preserving medical",
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
        # ── Cancer Genomics & TCGA ──
        ("TCGA cancer genome atlas tumor mutation burden", 10),
        ("BRCA1 BRCA2 breast cancer susceptibility genetics", 10),
        ("TP53 KRAS MYC oncogene cancer mutation", 8),
        ("circulating tumor DNA ctDNA liquid biopsy cancer", 8),
        ("precision oncology tumor mutational signature", 8),
        ("methylation array cancer epigenomics biomarker", 6),
        ("CAR-T cell immunotherapy solid tumor", 8),
        ("checkpoint inhibitor pembrolizumab nivolumab cancer", 10),
        ("chemotherapy resistance docetaxel cisplatin cancer", 8),
        # ── Cancer Pathology & Imaging ──
        ("whole slide image WSI histopathology cancer grading", 10),
        ("pathology deep learning tumor segmentation CNN transformer", 8),
        ("histopathology cancer detection metastasis lymph node", 8),
        ("radiomics PET CT MRI cancer staging prognosis", 8),
        ("molecular imaging FDG PET cancer diagnosis", 6),
        # ── Clinical Oncology NLP ──
        ("oncology clinical trial NLP extraction structured", 10),
        ("clinical NLP tumor registry cancer reporting", 8),
        ("OncoKB variant interpretation cancer mutations", 8),
        ("biomarker HER2 ER PR Ki67 breast cancer pathology", 8),
        # ── Cancer Prognosis & Survival ──
        ("cancer survival prognosis Cox regression machine learning", 10),
        ("recurrence prediction breast cancer survival analysis", 8),
        ("PFS OS overall survival cancer clinical trial", 6),
        ("liquid biopsy early cancer detection ctDNA methyl", 8),
        # ── Cancer Foundation Models & Fine-tuning ──
        ("pathology foundation model cancer detection Virchow", 10),
        ("PLIP protein ligand interaction cancer drug", 8),
        ("LoRA fine-tuning cancer genomics model", 10),
        ("protein language model cancer mutation effect", 8),
        ("RAG clinical decision oncology support", 8),
        ("instruction tuning oncological domain LLM", 8),
        # ── Emerging & Drug Discovery ──
        ("targeted therapy tyrosine kinase inhibitor cancer", 8),
        ("antibody drug conjugate ADC cancer payload", 6),
        ("tumor microenvironment immune evasion microenvironment", 8),
        ("metastasis organotropism machine learning prediction", 6),
        ("cancer synthetic lethality BRCA PARP inhibitor", 6),
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

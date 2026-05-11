#!/usr/bin/env python3
"""
RAG-Verify CLI — One command to run the entire pipeline.

Usage:
    rag_verify.py --pdf paper.pdf                  Verify a paper (auto-expands corpus if needed)
    rag_verify.py --status                         Show corpus and KB status
    rag_verify.py --rebuild                        Rebuild knowledge base from corpus
    rag_verify.py --add "medical imaging CNN"      Download arXiv papers by topic
    rag_verify.py --add-id 2301.12345 2302.45678  Download specific papers by arXiv ID
    rag_verify.py --reset                          Clear corpus and knowledge base
"""
import argparse
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"


def run(script: Path, *args, cwd=None):
    result = subprocess.run(
        [str(VENV_PYTHON), str(script)] + list(args),
        cwd=cwd or SCRIPT_DIR,
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="rag_verify.py",
        description="RAG-Verify — Local research paper fact-checker",
        epilog="Example: rag_verify.py --pdf ./my_paper.pdf"
    )
    parser.add_argument("--pdf", "-p", type=str, help="Verify a PDF file")
    parser.add_argument("--status", "-s", action="store_true", help="Show status")
    parser.add_argument("--rebuild", "-r", action="store_true", help="Rebuild knowledge base")
    parser.add_argument("--add", "-a", type=str, help="Download arXiv papers by topic")
    parser.add_argument("--add-id", nargs="+", type=str, help="Download specific arXiv papers by ID")
    parser.add_argument("--reset", action="store_true", help="Clear corpus and knowledge base")
    args = parser.parse_args()

    if args.reset:
        import shutil, sqlite3
        corpus = SCRIPT_DIR / "corpus"
        chroma = SCRIPT_DIR / "chroma_db"
        for d in [corpus, chroma]:
            if d.exists():
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()
        print("[*] Corpus and knowledge base cleared.")
        return

    if args.status:
        import sqlite3
        corpus = SCRIPT_DIR / "corpus"
        chroma = SCRIPT_DIR / "chroma_db"
        pdfs = list(corpus.glob("*.pdf")) if corpus.exists() else []
        chunks = 0
        if chroma.exists():
            try:
                conn = sqlite3.connect(chroma / "chroma.sqlite3")
                chunks = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                conn.close()
            except Exception:
                pass
        print("=== RAG-Verify Status ===")
        print(f"  Corpus:  {len(pdfs)} PDFs in {corpus}")
        print(f"  KB:      {chunks:,} chunks in ChromaDB")
        return

    if args.rebuild:
        run(SCRIPT_DIR / "auto_corpus.py", "--rebuild-only")
        return

    if args.add:
        run(SCRIPT_DIR / "download_arxiv.py", "--query", args.add, "--corpus", str(SCRIPT_DIR / "corpus"))
        return

    if args.add_id:
        ids = args.add_id
        run(SCRIPT_DIR / "download_arxiv.py", "--ids", *ids, "--corpus", str(SCRIPT_DIR / "corpus"))
        return

    if args.pdf:
        run(SCRIPT_DIR / "auto_corpus.py", args.pdf)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

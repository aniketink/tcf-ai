# RAG-Verify — Project Overview

## What It Is
RAG-Verify is a fully local, privacy-preserving research paper fact-checker. Drop a PDF, it verifies claims against a knowledge base built from arXiv papers.

## Architecture
```
auto_corpus.py          # Entry point — orchestrates everything
├── download_arxiv.py  # Downloads papers from arXiv with rate limiting
├── ingest.py           # Chunks PDFs and embeds into ChromaDB
├── verify_v2.py        # Full pipeline: extract → retrieve → rerank → self-reflect → verify → verdict
├── rag_verify.py       # CLI wrapper
└── rag_verify_tui.py  # Interactive TUI
```

## Key Files
- `auto_corpus.py` — Main entry point for autonomous verification
- `verify_v2.py` — Core RAG pipeline with cross-encoder reranking + self-reflection
- `download_arxiv.py` — arXiv paper downloader with rate limiting
- `ingest.py` — PDF ingestion into ChromaDB
- `rag_verify.py` — Simple CLI
- `rag_verify_tui.py` — Interactive TUI

## Dependencies
- Python 3.13 (project venv: `.venv/`)
- Ollama (serves `llama3.2:3b` locally)
- ChromaDB (vector store)
- sentence-transformers (embeddings + cross-encoder)

## Key Behaviors
- Knowledge base stored in `./chroma_db/` (auto-built from `./corpus/`)
- Ollama runs `llama3.2:3b` on localhost:11434
- arXiv download: 1 request per ~4 seconds (rate limit enforcement)
- Corpus auto-expands if >50% of claims are INCONCLUSIVE
- Medical domain: 24 seed topics, ~190 foundational papers

## Commands
```bash
python auto_corpus.py paper.pdf     # Full pipeline (auto-expand corpus)
python rag_verify.py --status       # Show corpus + KB status
python rag_verify.py --add "medical imaging"  # Add papers by topic
python rag_verify.py --reset        # Clear everything
```

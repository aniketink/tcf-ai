# RAG-Verify

**A fully local, privacy-preserving research paper fact-checker.**

Built by [aniketink](https://github.com/aniketink). Drop a PDF, get structured verdicts on every major claim — SUPPORTED, CONTRADICTED, or INCONCLUSIVE — with evidence citations. No API keys, no cloud, runs on a MacBook.

---
<img width="595" height="1080" alt="Mermaid Diagram May 11 2026 (1)" src="https://github.com/user-attachments/assets/e609405e-3fa5-485c-97eb-d455db10d903" />

## What It Does

```
Your Paper → Extract Claims → ANN Retrieval → Cross-Encoder Rerank
                                          ↓
                              Self-Reflection (2 rounds)
                                          ↓ + targeted retrieval
                              Sub-Claim Verification
                                          ↓
                              Verdict Aggregation → Results Table
```

Given a paper, RAG-Verify:
1. Extracts all major factual claims via local Ollama LLM
2. Retrieves top-20 candidate evidence chunks from ChromaDB
3. Re-ranks them with a cross-encoder — keeps top 5
4. Runs a self-reflection critique loop to decompose claims into verifiable sub-claims
5. Identifies weak evidence areas and triggers targeted secondary retrieval
6. Aggregates sub-claim verdicts into SUPPORTED / CONTRADICTED / INCONCLUSIVE

---

## Features

- **Cross-encoder reranking** — re-scores evidence for precision, above raw similarity
- **Self-reflection critique loop** — decomposes claims into sub-claims + fills evidence gaps
- **Auto-corpus expansion** — downloads relevant arXiv papers when coverage is weak
- **Medical domain optimized** — seed corpus of ~190 foundational medical AI papers
- **Fully local** — Ollama + ChromaDB, no external API calls
- **One command** — `python auto_corpus.py paper.pdf`

---

## Requirements

| | |
|---|---|
| OS | macOS (Apple Silicon) or Linux |
| RAM | 8GB minimum |
| Disk | ~5GB for corpus + models |
| Ollama | `brew install ollama` |
| Python | 3.13 (managed by project venv) |

---

## Quick Start

```bash
# First time — run setup (creates venv, pulls llama3.2:3b)
chmod +x setup.sh && ./setup.sh

# Keep Ollama running in one terminal
ollama serve

# Verify a paper (auto-builds corpus on first run)
source .venv/bin/activate
python auto_corpus.py your_paper.pdf
```

First run downloads ~190 medical AI papers (5–10 minutes, arXiv rate-limited). Subsequent runs are instant.

---

## Usage

```bash
# Full pipeline — verify + auto-expand corpus if needed
python auto_corpus.py paper.pdf

# Status, add papers, rebuild KB
python rag_verify.py --status
python rag_verify.py --add "medical imaging CNN"
python rag_verify.py --rebuild
python rag_verify.py --reset

# Interactive TUI
python rag_verify_tui.py
```

---

## Output

A terminal table — one row per claim:

| Claim | Verdict | Explanation | Sources |
|---|---|---|---|
| LoRA freezes pre-trained weights during fine-tuning | **SUPPORTED** | All 3 sub-claims verified. F1 score of 91.3 cited. | transformer.pdf |
| LORA enables single-update multi-task adaptation | **INCONCLUSIVE** | 2/7 sub-claims supported. Insufficient evidence in corpus. | resnet.pdf |
| Performance plateaus at rank r=8 | **INCONCLUSIVE** | 1/2 sub-claims supported. Plateau evidence inconclusive. | resnet.pdf |

**Verdicts:**
- `SUPPORTED` — corpus evidence directly supports the claim
- `CONTRADICTED` — corpus evidence contradicts the claim
- `INCONCLUSIVE` — no or insufficient relevant evidence in the corpus
- `MOSTLY_SUPPORTED` — majority supported, some inconclusive

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Offline Ingestion (one-time)                      │
│                                                             │
│ PDFs → Chunker (500-char) → Embeddings → ChromaDB          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Pre-Retrieval                                     │
│                                                             │
│ Query → Semantic Cache (O(1) hit/miss)                     │
│            └→ Query Router → Expansion Ensemble           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Hybrid Retrieval                                  │
│                                                             │
│ ANN Search (ChromaDB) → RRF Fusion → Cross-Encoder Rerank  │
│                                           CRAG Grader        │
│                               Context ← [Relevant]           │
│                         Web Fallback ← [Irrelevant]         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Agentic Generation                                │
│                                                             │
│ Graded Context → Generator (llama3.2:3b) → Streaming UI    │
│                        ↕ Iterative Tools                    │
│             Self-Critique Loop ──→ Verified Output          │
│                              │                             │
│                        Telemetry → RAGAS Eval               │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1 — Ingestion
PDFs are parsed and chunked into 500-character segments with 100-char overlap. Each chunk is embedded with `all-MiniLM-L6-v2` and stored in ChromaDB with HNSW indexing.

### Phase 2 — Pre-Retrieval
Incoming queries check a semantic cache first (embedding nearest-match). On a miss, the query is expanded via a parallel ensemble: HyDE (hypothetical document), multi-query rephrasing, and step-back abstraction. The ensemble is submitted to retrieval.

### Phase 3 — Retrieval
Dense ANN search (ChromaDB) and sparse keyword search run in parallel and are fused via Reciprocal Rank Fusion. A cross-encoder re-ranks the fused results to top-5. A CRAG grader evaluates relevance — irrelevant results trigger a web search fallback.

### Phase 4 — Generation
The heavy LLM tier (Ollama `llama3.2:3b`) generates with iterative tool access for Python, SQL, and math. Output streams optimistically to the UI. An async self-critique step checks for hallucination before final delivery. Telemetry flows to RAGAS evaluation (Faithfulness, Recall).

---

## Project Structure

```
tcf-ai/
├── auto_corpus.py         # Main entry — orchestrates full pipeline
├── verify_v2.py          # Core RAG pipeline
├── ingest.py              # PDF ingestion → ChromaDB
├── download_arxiv.py      # arXiv downloader (rate-limited)
├── rag_verify.py          # CLI wrapper
├── rag_verify_tui.py     # Interactive TUI
├── setup.sh               # One-time setup
├── requirements.txt       # Python dependencies
├── CLAUDE.md             # Architecture docs (AI context)
├── README.md             # This file
└── corpus/               # PDF corpus
    └── chroma_db/        # Vector store (auto-created)
```

---

## Medical Seed Corpus

**24 topic areas, ~190 papers** across:

| Category | Topics |
|---|---|
| Medical Imaging | chest X-ray, brain MRI, histopathology, CT, OCT, dermoscopy |
| Clinical NLP | EHRs, NER, clinical report generation, deidentification |
| Medical LLMs | PubMedBERT, MedQA, GPT-4 in clinical settings |
| Diagnosis | cancer prognosis, sepsis prediction, diabetic retinopathy |
| Fine-tuning | LoRA, PEFT, instruction tuning for medical domain |
| Multi-modal | vision-language models, federated learning, self-supervised |
| Benchmarks | MIMIC, ChestX-ray14, RadGraph, i2b2 |

---

## Extending

**Different model** — edit `verify_v2.py`:
```python
llm = ChatOllama(model="llama3.1:8b", ...)
```
Then `ollama pull llama3.1:8b`.

**Better embeddings** — edit `verify_v2.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

**Add papers manually** — drop PDFs into `./corpus/` and rebuild:
```bash
python rag_verify.py --rebuild
```

---

## Troubleshooting

**"Ollama not running"** — start it first:
```bash
ollama serve
```

**arXiv download fails** — SSL cert issue on macOS. Run:
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

**All INCONCLUSIVE** — corpus lacks relevant papers. Either add papers to `./corpus/` manually or the auto-expansion will download relevant ones.

**Slow first run** — model and corpus download once. Subsequent runs are fast.

---

## Citation

Built by [aniketink](https://github.com/aniketink). MIT License.

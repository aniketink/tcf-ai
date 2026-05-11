#!/bin/bash
# RAG-Verify Setup Script — Run once to install dependencies and Ollama

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "================================================"
echo " RAG-Verify — Setup"
echo "================================================"

# 1. Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python 3.13 venv..."
    python3.13 -m venv "$VENV_DIR"
fi

echo "[*] Activating venv..."
source "$VENV_DIR/bin/activate"

# 2. Install dependencies
echo "[*] Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q
"$VENV_DIR/bin/pip" install langchain-ollama arxiv requests certifi -q

# 3. Install Ollama if not installed
if ! command -v ollama &> /dev/null; then
    echo "[*] Installing Ollama..."
    brew install ollama
fi

# 4. Pull the LLM
echo "[*] Pulling llama3.2:3b model (~1.9GB, may take a few minutes)..."
ollama pull llama3.2:3b

# 5. Build initial knowledge base
if [ -d "$PROJECT_DIR/corpus" ] && [ "$(ls -A "$PROJECT_DIR/corpus"/*.pdf 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[*] Building initial knowledge base..."
    "$VENV_DIR/bin/python" "$PROJECT_DIR/ingest.py" "$PROJECT_DIR/corpus"
fi

echo ""
echo "================================================"
echo " Setup complete!"
echo "================================================"
echo ""
echo "To verify a paper:"
echo "  source .venv/bin/activate"
echo "  ollama serve  (in one terminal)"
echo "  python auto_corpus.py your_paper.pdf  (in another)"
echo ""

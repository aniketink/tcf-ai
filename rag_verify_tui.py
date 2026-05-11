#!/usr/bin/env python3
"""
RAG-Verify TUI — Fully autonomous. Drop a PDF, get results.
Single workflow: auto-detect corpus -> auto-ingest -> verify -> show results.
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
INGEST_SCRIPT = SCRIPT_DIR / "ingest.py"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_v2.py"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_arxiv.py"


# ─── helpers ────────────────────────────────────────────────────────────────────

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
    return sorted(CORPUS_DIR.glob("*.pdf"))


def get_corpus_status():
    pdfs = get_corpus_pdfs()
    chunks = get_chunk_count()
    return {
        "pdfs": pdfs,
        "count": len(pdfs),
        "chunks": chunks,
        "needs_ingest": chunks == 0 and len(pdfs) > 0,
        "ready": chunks > 0 and len(pdfs) > 0,
    }


def ensure_kb(verbose=False):
    """Ensure knowledge base exists, rebuild if outdated."""
    status = get_corpus_status()

    if not status["pdfs"]:
        return False, "No PDFs in corpus. Add papers first."

    if status["chunks"] > 0 and status["needs_ingest"]:
        return False, "Knowledge base empty."

    if status["chunks"] > 0:
        return True, f"{status['chunks']:,} chunks"

    return True, None  # will ingest


def auto_ingest(progress=None, task=None):
    """Auto-ingest all corpus PDFs."""
    pdfs = get_corpus_pdfs()
    if not pdfs:
        return 0

    result = subprocess.run(
        [str(VENV_PYTHON), str(INGEST_SCRIPT), str(CORPUS_DIR)],
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    if progress and task is not None:
        progress.update(task, completed=100)

    return get_chunk_count()


def run_verify(pdf_path: Path, verbose: bool = False) -> str:
    """Run verify_v2.py and capture output."""
    cmd = [str(VENV_PYTHON), str(VERIFY_SCRIPT), str(pdf_path)]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(
        cmd,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    output = result.stdout.decode("utf-8", errors="replace")
    if result.returncode != 0 and not output:
        output = result.stderr.decode("utf-8", errors="replace")
    return output


# ─── screens ──────────────────────────────────────────────────────────────────

def render_home():
    status = get_corpus_status()
    header = Panel(
        "[bold cyan]RAG-Verify[/bold cyan]  —  Research Paper Fact-Checker  |  [dim]Fully autonomous[/dim]",
        style="on #0d1117",
        box=ROUNDED,
    )

    # corpus info
    if status["pdfs"]:
        t = Table(box=None, show_header=True, header_style="bold cyan")
        t.add_column("Corpus papers", style="cyan")
        t.add_column("Size", justify="right", style="dim")
        for p in status["pdfs"]:
            t.add_row(p.name, f"[dim]{p.stat().st_size / 1024 / 1024:.1f} MB[/dim]")
        corp_panel = Panel(t, title=f"[cyan]Corpus ({status['count']} papers)[/cyan]", border_style="cyan", box=ROUNDED)
    else:
        corp_panel = Panel("[yellow]No papers in corpus[/yellow]", title="[cyan]Corpus[/cyan]", border_style="yellow", box=ROUNDED)

    # KB info
    chunks = status["chunks"]
    if chunks > 0:
        kb_panel = Panel(f"[green]{chunks:,}[/green] [dim]chunks indexed[/dim]", title="[cyan]Knowledge Base[/cyan]", border_style="green", box=ROUNDED)
    else:
        kb_panel = Panel("[yellow]Not built[/yellow]", title="[cyan]Knowledge Base[/cyan]", border_style="yellow", box=ROUNDED)

    info_table = Table(box=None, show_header=False, padding=(0, 2))
    info_table.add_column("", ratio=1)
    info_table.add_column("", ratio=1)
    info_table.add_row(corp_panel, kb_panel)

    # instructions
    inst = [
        "[bold]How it works:[/bold]",
        "[dim]1.[/dim] [cyan]Put PDFs in ./corpus[/cyan]   →   papers for the knowledge base",
        "[dim]2.[/dim] [cyan]Drop your paper[/cyan]        →   verify it against the knowledge base",
        "[dim]3.[/dim] Results shown       →   SUPPORTED / CONTRADICTED / INCONCLUSIVE",
        "",
        "[dim]Drag & drop a PDF here, or type the path, then press Enter[/dim]",
    ]
    inst_panel = Panel("\n".join(inst), box=ROUNDED, border_style="dim")

    footer = Panel("[dim]Press [bold]Q[/bold] to quit | [bold]A[/bold] to add arXiv papers | [bold]R[/bold] to rebuild KB[/dim]", style="on #0d1117")

    console.print(header)
    console.print()
    console.print(info_table)
    console.print()
    console.print(inst_panel)
    console.print()
    console.print(footer)


def render_drop_zone(pdf_path: str) -> Table:
    t = Table(box=ROUNDED, show_header=False, padding=(1, 2))
    t.add_column("")
    t.add_row("[bold cyan]Verifying:[/bold cyan]", pdf_path)
    return t


def render_verification_progress(step: str, detail: str = ""):
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("step", style="cyan", width=20)
    t.add_column("detail", style="dim")
    t.add_row(f"[cyan]{step}[/cyan]", detail)
    return t


# ─── workflow ───────────────────────────────────────────────────────────────────

def workflow_verify(pdf: Path, verbose: bool = False):
    """The full autonomous verification workflow."""
    console.clear()

    start = datetime.now()

    # Step 0: header
    header = Panel(
        f"[bold cyan]Verifying:[/bold cyan] {pdf.name}",
        style="on #0d1117",
        box=ROUNDED,
    )
    console.print(header)
    console.print()

    # Step 1: check corpus
    status = get_corpus_status()
    if not status["pdfs"]:
        console.print("[red]No papers in corpus. Add PDFs to ./corpus first.[/red]")
        return

    # Step 2: check / build KB
    steps = Table(box=None, show_header=False, padding=(0, 3))
    steps.add_column("", width=4)
    steps.add_column("", width=40)
    steps.add_column("", style="dim")

    def add_step(num, label, detail, ok=None):
        icon = "[green]✓[/green]" if ok == True else ("[red]✗[/red]" if ok == False else "[cyan]▸[/cyan]")
        steps.add_row(icon, f"[bold]{label}[/bold]", detail)

    add_step("1", "Loading paper", pdf.name)

    if status["chunks"] == 0:
        add_step("2", "Building knowledge base", f"{status['count']} PDFs → ChromaDB")
        console.print(steps)
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("[cyan]Ingesting corpus...[/cyan]", total=100)
            chunks = auto_ingest(progress, ptask)
            steps = Table(box=None, show_header=False, padding=(0, 3))
            steps.add_column("", width=4)
            steps.add_column("", width=40)
            steps.add_column("", style="dim")
            add_step("1", "Loading paper", pdf.name)
            add_step("2", "Building knowledge base", f"[green]{chunks:,} chunks indexed[/green]", ok=True)
    else:
        add_step("2", "Knowledge base ready", f"[green]{status['chunks']:,} chunks[/green]", ok=True)
        console.print(steps)
        console.print()

    # Step 3: extract claims
    add_step("3", "Extracting claims", "LLM analyzing paper...")
    console.print(steps)
    console.print()

    # Step 4: run verification
    add_step("4", "Verifying claims", "Retrieval + reranking + verdict...")
    console.print(steps)
    console.print()

    console.print("[dim]Running verification (this may take a few minutes)...[/dim]\n")

    output = run_verify(pdf, verbose=verbose)

    console.print("\n[bold cyan]Results:[/bold cyan]\n")
    console.print(Syntax(output, "bash", theme="monokai", line_numbers=False))

    elapsed = (datetime.now() - start).total_seconds()
    console.print(f"\n[dim]Completed in {elapsed:.0f}s[/dim]")


def workflow_add_arxiv():
    console.clear()

    topics = [
        "LoRA low-rank adaptation",
        "retrieval augmented generation",
        "transformer fine-tuning",
        "RAG fact-checking",
        "parameter efficient fine-tuning",
    ]

    t = Table(title="Add Papers from arXiv", show_header=True, box=ROUNDED)
    t.add_column("#", style="dim", width=3)
    t.add_column("Topic", style="cyan")
    for i, topic in enumerate(topics, 1):
        t.add_row(str(i), topic)
    t.add_row("", "[dim]Custom topic...[/dim]")

    console.print(t)
    console.print()

    choice = Prompt.ask(
        "[cyan]Choose a topic (number)[/cyan] or type a custom topic",
        default="1"
    ).strip()

    if choice.isdigit() and 1 <= int(choice) <= len(topics):
        topic = topics[int(choice) - 1]
    else:
        topic = choice

    count = Prompt.ask("[cyan]How many papers?[/cyan]", default="5").strip()
    count_int = int(count) if count.isdigit() else 5

    console.print(f"\n[dim]Searching arXiv for: '{topic}'[/dim]\n")

    result = subprocess.run(
        [str(VENV_PYTHON), str(DOWNLOAD_SCRIPT),
         "--query", topic, "--max", str(count_int), "--corpus", str(CORPUS_DIR)],
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    console.print(Syntax(result.stdout.decode("utf-8", errors="replace"), "bash", theme="monokai", line_numbers=False))

    new_pdfs = get_corpus_pdfs()
    console.print(f"\n[green]Corpus now has {len(new_pdfs)} papers.[/green]")
    console.print("[dim]Press Enter to continue...[/dim]")


def workflow_rebuild_kb():
    console.clear()

    pdfs = get_corpus_pdfs()
    if not pdfs:
        console.print("[yellow]No PDFs in corpus to ingest.[/yellow]")
        return

    t = Table(title="Papers to ingest", show_header=True, box=ROUNDED)
    t.add_column("#", style="dim", width=3)
    t.add_column("Paper", style="cyan")
    t.add_column("Size", justify="right", style="dim")
    for i, p in enumerate(pdfs, 1):
        t.add_row(str(i), p.name, f"{p.stat().st_size / 1024 / 1024:.1f} MB")
    console.print(t)
    console.print()

    console.print("[dim]Rebuilding knowledge base...[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("[cyan]Ingesting...[/cyan]", total=100)
        chunks = auto_ingest(progress, ptask)

    console.print(f"\n[green]Done! {chunks:,} chunks indexed.[/green]")


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    console.clear()

    while True:
        console.clear()
        render_home()

        console.print()
        choice = Prompt.ask(
            "[bold cyan]PDF path or command:[/bold cyan]",
            default=""
        ).strip()

        if not choice:
            continue

        if choice.lower() == "q":
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        if choice.lower() == "a":
            workflow_add_arxiv()
            Prompt.ask("[dim]Press Enter to continue...[/dim]")
            continue

        if choice.lower() == "r":
            workflow_rebuild_kb()
            Prompt.ask("[dim]Press Enter to continue...[/dim]")
            continue

        # treat as PDF path
        pdf_path = Path(choice)
        if not pdf_path.exists():
            console.print(f"[red]File not found: {pdf_path}[/red]")
            console.print("[dim]Try: drag the PDF file here, or type the full path.[/dim]")
            console.print("[dim]Press Enter to continue...[/dim]")
            Prompt.ask("")
            continue

        if pdf_path.suffix.lower() != ".pdf":
            console.print(f"[red]Not a PDF: {pdf_path}[/red]")
            Prompt.ask("[dim]Press Enter to continue...[/dim]")
            continue

        workflow_verify(pdf_path, verbose=True)
        Prompt.ask("[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    main()

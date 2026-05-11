#!/usr/bin/env python3
"""Download papers from arXiv with rate limiting and retry."""

import time
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

REQUEST_DELAY = 4.0  # seconds between API requests (arXiv limit: ~1 req/3s)


def _parse_arxiv_ids(xml_text: str) -> list[str]:
    """Parse arXiv IDs from Atom feed XML."""
    ids = []
    # The feed uses default namespace or atom prefix
    try:
        root = ET.fromstring(xml_text)
        # Try atom namespace
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            id_elem = entry.find("a:id", ns)
            if id_elem is not None:
                text = id_elem.text or ""
                # format: http://arxiv.org/abs/2301.12345v2
                aid = text.rsplit("/abs/", 1)[-1]
                ids.append(aid)
    except ET.ParseError:
        # fallback: regex
        import re
        ids = re.findall(r'http://arxiv\.org/abs/([^<]+)', xml_text)
    return ids


def download_by_query(query: str, max_results: int = 10, output_dir: str = "./corpus") -> int:
    """Search arXiv API and download PDFs with rate limiting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    search_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            resp = requests.get(search_url, params=params, timeout=60, verify=False)
            if resp.status_code == 429:
                retry_count += 1
                wait = int(resp.headers.get("Retry-After", 60)) + random.uniform(0, 15)
                print(f"  [!] Rate limited. Waiting {wait:.0f}s... ({retry_count}/{max_retries})")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                retry_count += 1
                print(f"  [!] Server error {resp.status_code}. Retrying... ({retry_count}/{max_retries})")
                time.sleep(10)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            retry_count += 1
            print(f"  [!] Request failed: {e}. Retrying... ({retry_count}/{max_retries})")
            time.sleep(5)

    if retry_count >= max_retries:
        print(f"  [!] Failed to search arXiv after {max_retries} retries.")
        return 0

    entry_ids = _parse_arxiv_ids(resp.text)
    if not entry_ids:
        print(f"  [!] No results found for: '{query}'")
        return 0

    print(f"  Found {len(entry_ids)} results. Downloading PDFs...")
    downloaded = 0

    for i, aid in enumerate(entry_ids):
        out_path = output_path / f"{aid}.pdf"
        if out_path.exists():
            print(f"  [=] Already exists: {aid}")
            time.sleep(REQUEST_DELAY)
            continue

        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        try:
            print(f"  [↓] Downloading {aid}...")
            resp = requests.get(pdf_url, timeout=60, verify=False)
            if resp.status_code in (429, 503):
                print(f"  [!] Rate limited. Waiting 30s...")
                time.sleep(30)
                resp = requests.get(pdf_url, timeout=60, verify=False)
            resp.raise_for_status()

            with open(out_path, "wb") as f:
                f.write(resp.content)

            mb = len(resp.content) / 1024 / 1024
            print(f"  [+] {aid} ({mb:.1f} MB)")
            downloaded += 1
        except requests.exceptions.RequestException as e:
            print(f"  [-] Failed: {aid} ({e})")

        if i < len(entry_ids) - 1:
            time.sleep(REQUEST_DELAY)

    return downloaded


def download_by_arxiv_ids(ids: list[str], output_dir: str = "./corpus") -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i, aid in enumerate(ids):
        out_path = output_path / f"{aid}.pdf"
        if out_path.exists():
            print(f"  [=] Already exists: {aid}")
            time.sleep(REQUEST_DELAY)
            continue

        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        try:
            resp = requests.get(pdf_url, timeout=60, verify=False)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  [+] {aid}")
            downloaded += 1
        except requests.exceptions.RequestException as e:
            print(f"  [-] Failed: {aid} ({e})")

        if i < len(ids) - 1:
            time.sleep(REQUEST_DELAY)

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--ids", nargs="+", type=str)
    parser.add_argument("--max", "-n", type=int, default=10)
    parser.add_argument("--corpus", "-c", type=str, default=None)
    args = parser.parse_args()
    args.corpus = args.corpus or str(Path(__file__).parent / "corpus")

    if args.ids:
        n = download_by_arxiv_ids(args.ids, args.corpus)
        print(f"\nDownloaded {n}/{len(args.ids)} papers")
    elif args.query:
        n = download_by_query(args.query, max_results=args.max, output_dir=args.corpus)
        print(f"\nDownloaded {n}/{args.max} papers to '{args.corpus}'")
    else:
        parser.print_help()

#!/usr/bin/env python3
"""
crawl_to_md.py

Usage:
    python3 crawl_to_md.py <start_url> <output_dir>

Example:
    python3 crawl_to_md.py https://example.com ./site_md
"""

import os
import re
import sys
import base64
import urllib.parse
from pathlib import Path
from collections import deque

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as mdify

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch(url: str) -> str | None:
    """Return the page content or None on failure."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        print(f"[WARN] Failed to fetch {url}: {exc}")
        return None


def to_base64(img_url: str, base_url: str) -> str | None:
    """Download an image and return a data‑uri string."""
    try:
        full_url = urllib.parse.urljoin(base_url, img_url)
        r = requests.get(full_url, timeout=10)
        r.raise_for_status()
        mime = r.headers.get("Content-Type", "image/png")
        b64 = base64.b64encode(r.content).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as exc:
        print(f"[WARN] Could not embed image {img_url}: {exc}")
        return None


def process_main(html: str, base_url: str) -> str:
    """Extract <main>, embed images, and convert to Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        # Fallback to <article> or <body> if <main> is missing
        main = soup.find("article") or soup.find("body")
        if not main:
            return ""

    # Convert images to base64
    for img in main.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        data_uri = to_base64(src, base_url)
        if data_uri:
            img["src"] = data_uri

    # Convert the <main> content to Markdown
    return mdify(str(main), heading_style="ATX")


def url_to_path(url: str, base: str) -> Path:
    """Map a URL to a local file path inside the output dir."""
    parsed = urllib.parse.urlparse(url)
    rel = parsed.path.lstrip("/")
    if not rel or rel.endswith("/"):
        rel = os.path.join(rel, "index.html")
    # Replace .html/.htm with .md
    rel = re.sub(r"\.(html?|php)$", ".md", rel)
    return Path(base) / rel

# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

def crawl(start_url: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visited = set()
    queue = deque([start_url])

    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"[INFO] Crawling {url}")
        html = fetch(url)
        if not html:
            continue

        # Save Markdown
        md = process_main(html, url)
        if md:
            out_path = url_to_path(url, out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            print(f"[OK]  Saved {out_path}")

        # Find internal links to crawl
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Strip fragment identifiers and query strings for deduplication
            href = urllib.parse.urldefrag(href)[0]
            # Resolve relative URLs
            child = urllib.parse.urljoin(url, href)
            # Only follow same‑domain links
            if urllib.parse.urlparse(child).netloc == urllib.parse.urlparse(start_url).netloc:
                if child not in visited:
                    queue.append(child)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    start = sys.argv[1]
    out = sys.argv[2]
    crawl(start, out)

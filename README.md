# python-examples

Practical examples and prototypes for building a **crawl → Markdown → hierarchical vector database** pipeline designed for **recursive or hierarchical LLM systems**, **Retrieval-Augmented Generation (RAG)**, and agent memory.


## Recursive Language Models (RLM) and Hierarchical Context
This architecture is well-suited to recursive or hierarchical LLM systems and hierarchical document reasoning, and it directly supports the goals of minimising token usage, reducing hallucinations, and preventing context drift.

Hallucinations are still possible if:
- summaries are wrong
- metadata is weak
- the model ignores instructions

## Why this works for RLM-style systems

Recursive or hierarchical LLM systems operate by reasoning over structured summaries first, then selectively descending into more detailed context only when required. This pipeline enables that pattern naturally.

## Key properties that make this effective:

- **Hierarchical** file structure
URL-mirrored directories create an implicit document tree (site → section → page → chunk). This allows agents to reason top-down instead of loading flat context.

- **Chunk-level** semantic boundaries
~500-word chunks preserve semantic coherence, enabling recursive expansion without injecting unrelated context.

- **Stable document identities**
Deterministic paths and chunk IDs prevent context drift across recursive calls and agent iterations.

But only if:
- chunking logic is stable
- content order doesn’t change
- re-ingestion uses the same rules

**Metadata-first retrieval**
Path, title, section, and tags allow an RLM to filter before embedding similarity, dramatically reducing token load.

- chromaDB itself does not enforce “metadata-first” — your query logic must do this

# Summary 
This repository contains three complementary scripts:

| Step | File | Description |
|------|------|-------------|
| Step 1 | `crawl_to_md.py` | crawl a website and convert it into structured Markdown |
| Step 2 | `create_summary_anchors.py`| generate `_summary.md` files for every folder containing Markdown files, creating folder-level anchors for RLM-style retrieval|
| Step 3 | `ingest_md_to_chroma.py` | ingest Markdown files into ChromaDB with embeddings |
| Step 4 | `query_chroma.py` | example call |

---

# Prerequisites

### Using Homebrew (recommended on macOS)
```bash
brew install python
```

### Install required packages for crawl_to_md.py
```bash
pip3 install requests beautifulsoup4 markdownify
```
### Install required for ingest_md_to_chroma.py
```bash
pip3 install chromadb sentence-transformers tqdm markdown
```
### Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

# crawl_to_md.py

An end‑to‑end crawler that:

1. Traverses internal links from a start URL.
2. Extracts the **main content** – it first looks for a `<main>` tag, then falls back to `<article>` or `<body>` if `<main>` is missing.
3. Converts the extracted HTML to Markdown using `markdownify`.
4. Mirrors the site’s URL structure in the output directory.

### Running the crawler
```
python3 crawl_to_md.py https://example.com ./site_md
```

### Example output structure:
```
site_md/
├── _summary.md
├── index.md
├── about.md
├── blog/
│ ├── _summary.md
│ ├── post1.md
│ └── post2.md
```

# ingest_md_to_chroma.py

This script ingests a directory of Markdown files (typically produced by `crawl_to_md.py`) into **ChromaDB** for semantic search, retrieval-augmented generation (RAG), and agent memory.

## How it works

| Step | What happens | Why it matters |
|-----|-------------|----------------|
| **Walk** | `root.rglob("*.md")` finds every Markdown file | Guarantees full coverage |
| **Read** | `read_text()` loads UTF-8 content | Handles all files safely |
| **Chunk** | Splits content into ~500-word chunks | Keeps vectors small and semantically coherent |
| **Embed** | Uses `sentence-transformers` to produce 384-dim vectors | High-quality semantic embeddings |
| **Metadata** | Stores path, title, section, chunk index | Enables filtering and context reconstruction |
| **Add** | `collection.add()` writes vectors to disk | Persistence across restarts |



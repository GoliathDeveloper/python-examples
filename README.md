# python-examples
examples and prototypes

# Using Homebrew (recommended)
brew install python

# Install required packages for crawl_to_md.py
pip3 install requests beautifulsoup4 markdownify

# Install required for ingest_md_to_chroma.py
pip install chromadb sentence-transformers tqdm markdown

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# crawl_to_md.py
A self‑contained Python script that:

1. Crawls every page reachable from a given start URL (depth‑first, no external domains).
2. Extracts the content inside the <main> tag of each page.
3. Converts that HTML fragment to Markdown (using markdownify).
4. Embeds any images found in the <main> tag as Base‑64 data URIs so the LLM can read them.
5. Writes each page’s Markdown to a file named after the page’s relative path (e.g., index.md, about.md, blog/post1.md).

# Running the crawler script
python3 crawl_to_md.py https://example.com ./site_md


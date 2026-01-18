#!/usr/bin/env python3
"""
ingest_md_to_chroma.py

Walks the markdown tree, chunks, embeds, and stores in ChromaDB.
"""

import os
import pathlib
import uuid
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import markdown

# ---------- Configuration ----------
ROOT_DIR = pathlib.Path("< path to folder that contains markdown >")
CHUNK_SIZE = 500  # words
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = pathlib.Path("./chromadb")  # persistence dir

# ---------- Helpers ----------
def read_markdown(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Split by words; keep headings as separate chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
    return chunks

def extract_title(text: str) -> str:
    """Return the first markdown heading or file name."""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""

def walk_markdown_files(root: pathlib.Path) -> List[pathlib.Path]:
    return list(root.rglob("*.md"))

# ---------- Main ----------
def main():
    # Create Chroma client & collection
    client = chromadb.Client(path=CHROMA_DIR)
    collection = client.create_collection(name="< insert collection name >")

    # Load embedding model
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Process files
    files = walk_markdown_files(ROOT_DIR)
    for file_path in tqdm(files, desc="Processing files"):
        rel_path = file_path.relative_to(ROOT_DIR).as_posix()
        content = read_markdown(file_path)
        title = extract_title(content) or file_path.stem

        # Chunking
        chunks = split_into_chunks(content)

        # Prepare vectors & metadata
        ids = []
        embeddings = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            ids.append(f"{rel_path}::chunk-{idx}")
            embeddings.append(embed_fn(chunk))
            metadatas.append(
                {
                    "path": rel_path,
                    "title": title,
                    "section": file_path.parent.name,
                    "chunk_index": idx,
                }
            )

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("\nIngestion complete.")
    print(f"Collection size: {collection.count()} vectors.")
    print(f"ChromaDB stored at: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
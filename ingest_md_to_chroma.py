#!/usr/bin/env python3
"""
ingest_md_to_chroma.py

Walks the markdown tree, chunks, embeds, and stores in ChromaDB with support for
Recursive Language Model (RLM) workflows via folder-level _summary.md anchors and
hierarchical metadata for agent retrieval contracts.
"""

import os
import pathlib
import uuid
from typing import List, Tuple, Dict, Optional

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
    """Walk all markdown files, prioritizing _summary.md files per folder."""
    return sorted(
        list(root.rglob("*.md")),
        key=lambda p: (
            p.parent,
            p.name == "_summary.md"  # Summary files sort first within folder
        ),
    )

def get_folder_hierarchy(file_path: pathlib.Path, root: pathlib.Path) -> List[str]:
    """Return a list of ancestor folder names from root to the file's parent."""
    hierarchy = []
    current = file_path.parent
    while current != root and current != current.parent:
        hierarchy.append(current.name)
        current = current.parent
    hierarchy.reverse()
    return hierarchy

def is_summary_file(file_path: pathlib.Path) -> bool:
    """Check if this is a folder-level summary anchor."""
    return file_path.name == "_summary.md"

def get_folder_contents_summary(folder: pathlib.Path) -> str:
    """Build a summary of folder contents for agent retrieval contract."""
    contents = []
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            contents.append(f"- {child.name}/ (folder)")
        elif child.suffix == ".md" and child.name != "_summary.md":
            contents.append(f"- {child.stem} (document)")
    return "\n".join(contents) if contents else "(Empty folder)"

# ---------- Main ----------
def main():
    # Create Chroma client & collection
    client = chromadb.Client(path=CHROMA_DIR)
    collection = client.create_collection(name="< insert collection name >")

    # Load embedding model
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Process files (summaries first for hierarchical context)
    files = walk_markdown_files(ROOT_DIR)
    for file_path in tqdm(files, desc="Processing files"):
        rel_path = file_path.relative_to(ROOT_DIR).as_posix()
        content = read_markdown(file_path)
        title = extract_title(content) or file_path.stem

        # Determine if this is a summary anchor
        is_summary = is_summary_file(file_path)
        
        # Build hierarchical metadata for RLM agent retrieval contract
        hierarchy_meta = get_folder_hierarchy(file_path, ROOT_DIR)
        
        # If this is a summary, include folder contents for agent reasoning
        folder_contents = ""
        if is_summary:
            folder_contents = get_folder_contents_summary(file_path.parent)

        # Chunking
        chunks = split_into_chunks(content)

        # Prepare vectors & metadata
        ids = []
        embeddings = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk-{idx}"
            ids.append(chunk_id)
            embeddings.append(embed_fn(chunk))
            
            # Metadata for hierarchical retrieval
            metadata = {
                "path": rel_path,
                "title": title,
                "section": file_path.parent.name,
                "chunk_index": idx,
                "type": "summary" if is_summary else "content",
                "is_summary": "true" if is_summary else "false",
                "depth": len(get_folder_hierarchy(file_path, ROOT_DIR)),
                "parent_folders": get_folder_hierarchy(file_path, ROOT_DIR),
            }
            
            # Add hierarchy metadata (parent folder anchors)
            metadata.update(hierarchy_meta)
            
            # Add folder contents for summaries (agent retrieval contract)
            if is_summary and folder_contents:
                metadata["folder_contents"] = folder_contents
            
            metadatas.append(metadata)

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("\nIngestion complete.")
    print(f"Collection size: {collection.count()} vectors.")
    print(f"ChromaDB stored at: {CHROMA_DIR.resolve()}")
    print(f"\nSupports RLM workflows with:")
    print(f"  - Folder-level _summary.md anchors")
    print(f"  - Hierarchical metadata for agent retrieval contracts")
    print(f"  - Parent-child folder relationships")


if __name__ == "__main__":
    main()
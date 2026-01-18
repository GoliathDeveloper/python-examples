import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client(path="./chromadb")
collection = client.get_collection(name="< collection name >")

# Simple text query
query_text = "< Insert question >?"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
query_vec = embed_fn(query_text)

# First pass: summaries only (folder-level anchors for RLM retrieval contract)
summary_results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={"type": "summary"}
)

# Second pass: detailed content
content_results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"type": "content"}
)

print(f"\nQuery: {query_text}")
print("=" * 80)
print("\n[SUMMARIES] Folder-level anchors (agent retrieval contract):")
print("-" * 80)

for i, (doc_id, metadata) in enumerate(zip(summary_results["ids"][0], summary_results["metadatas"][0]), 1):
    print(f"\n{i}. {metadata['title']}")
    print(f"   Path: {metadata['path']}")
    if "folder_contents" in metadata:
        print(f"   Contents: {metadata['folder_contents']}")

print("\n" + "=" * 80)
print("\n[CONTENT] Detailed chunks:")
print("-" * 80)

for i, (doc_id, metadata) in enumerate(zip(content_results["ids"][0], content_results["metadatas"][0]), 1):
    print(f"\n{i}. {metadata['title']}")
    print(f"   Path: {metadata['path']} (chunk {metadata['chunk_index']})")
    print(f"   Section: {metadata['section']}")
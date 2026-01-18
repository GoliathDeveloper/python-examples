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

results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"section": "explanation"}  # optional filter
)

for match in results["ids"][0]:
    print(f"Score: {match['score']:.3f}")
    print(f"Path: {match['metadata']['path']}")
    print(f"Chunk: {match['metadata']['chunk_index']}")
    print(f"Snippet: {match['metadata']['snippet'][:200]}...")
    print("-" * 80)
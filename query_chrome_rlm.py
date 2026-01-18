import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional

def query_with_hierarchy(
    query_text: str,
    n_results: int = 5,
    retrieve_summaries_first: bool = True,
    filter_section: Optional[str] = None,
) -> None:
    """
    Query ChromaDB with RLM-style hierarchical retrieval.
    
    Args:
        query_text: The user query
        n_results: Number of chunks to retrieve
        retrieve_summaries_first: If True, prioritize _summary.md anchors (retrieval contract)
        filter_section: Optional folder/section to constrain retrieval
    """
    client = chromadb.Client(path="./chromadb")
    collection = client.get_collection(name="< collection name >")
    
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    query_vec = embed_fn(query_text)
    
    # Build filter for hierarchical retrieval
    where_filter = None
    if filter_section:
        where_filter = {"section": filter_section}
    
    # Step 1: Retrieve with optional summaries-first bias
    n_to_retrieve = n_results * 2 if retrieve_summaries_first else n_results
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_to_retrieve,
        where=where_filter
    )
    
    # Step 2: If summaries-first, separate summaries and content
    if retrieve_summaries_first:
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Separate by type
        summaries = []
        content = []
        for mid, meta, dist in zip(ids, metadatas, distances):
            if meta.get("type") == "summary":
                summaries.append((mid, meta, dist))
            else:
                content.append((mid, meta, dist))

        # Sort each list by distance
        summaries.sort(key=lambda x: x[2])
        content.sort(key=lambda x: x[2])

        # Combine, taking up to n_results
        combined = summaries[:n_results]
        if len(combined) < n_results:
            combined.extend(content[:n_results - len(combined)])

        # Display results with hierarchical context
        print(f"\nQuery: {query_text}")
        print("Results (prioritizing folder-level summaries for agent retrieval contract):")
        print("=" * 80)

        for idx, (match_id, metadata, distance) in enumerate(combined, 1):
            is_summary = metadata.get("type") == "summary"
            summary_badge = "[SUMMARY]" if is_summary else "[CONTENT]"

            print(f"\n{idx}. {summary_badge} {metadata.get('title', 'Untitled')}")
            print(f"   Path: {metadata.get('path')}")
            print(f"   Section: {metadata.get('section')}")
            print(f"   Depth: {metadata.get('depth', 'N/A')}")
            print(f"   Distance: {distance:.3f}")

            # Show parent anchors (hierarchy)
            parent_folders = metadata.get("parent_folders", [])
            if parent_folders:
                print("   Parent anchors:")
                for pf in parent_folders:
                    print(f"     - {pf}")

            # Show folder contents for summaries (agent retrieval contract)
            if "folder_contents" in metadata:
                print("   Folder contents (retrieval contract):")
                for line in metadata["folder_contents"].split("\n"):
                    print(f"     {line}")

            print(f"   Chunk: {metadata.get('chunk_index')}")
            print("-" * 80)
    else:
        # Fallback: simple flat retrieval
        print(f"\nQuery: {query_text}")
        print("Results:")
        print("=" * 80)
        
        for idx, (match_id, metadata, distance) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0]),
            1
        ):
            print(f"\n{idx}. {metadata.get('title', 'Untitled')}")
            print(f"   Path: {metadata.get('path')}")
            print(f"   Section: {metadata.get('section')}")
            print(f"   Distance: {distance:.3f}")
            print(f"   Chunk: {metadata.get('chunk_index')}")
            print("-" * 80)


if __name__ == "__main__":
    # Example: RLM-style hierarchical query
    query_text = "< Insert question >?"
    
    # Query with folder summaries prioritized (retrieval contract)
    query_with_hierarchy(
        query_text,
        n_results=5,
        retrieve_summaries_first=True,
        filter_section=None  # Optional: filter to a specific section
    )
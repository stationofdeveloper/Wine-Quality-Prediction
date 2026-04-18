"""
📚 RAG Engine — Wine Quality Prediction
=========================================
Builds a local vector store from wine knowledge base files
and retrieves relevant context to augment LLM responses.

Dependencies (all free):
    pip install chromadb sentence-transformers langchain langchain-community
"""

import os
import glob
from pathlib import Path
from typing import Optional

# ─── lazy imports (only loaded when needed) ────────────────────────────────────
_chroma_client = None
_collection    = None
_embedder      = None

KNOWLEDGE_DIR = Path(__file__).parent / "wine_knowledge"
CHROMA_DIR    = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "wine_knowledge"


def _get_embedder():
    """Load sentence-transformers model once (cached)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # Small, fast, free model — no API key needed
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    """Return (or create) the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    import chromadb
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        _collection = _chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        _collection = _chroma_client.create_collection(COLLECTION_NAME)
        _build_index()

    return _collection


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _build_index():
    """Index all .txt files in wine_knowledge/ into ChromaDB."""
    if not KNOWLEDGE_DIR.exists():
        print("⚠️  wine_knowledge/ folder not found — RAG disabled.")
        return

    embedder   = _get_embedder()
    collection = _collection

    all_chunks = []
    all_ids    = []
    all_metas  = []

    for txt_file in KNOWLEDGE_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{txt_file.stem}_{i}")
            all_metas.append({"source": txt_file.name})

    if not all_chunks:
        print("⚠️  No .txt files found in wine_knowledge/")
        return

    # Embed and add in batches
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        embeddings   = embedder.encode(batch_chunks).tolist()
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            ids=all_ids[i:i + batch_size],
            metadatas=all_metas[i:i + batch_size],
        )

    print(f"✅  RAG index built: {len(all_chunks)} chunks from "
          f"{len(list(KNOWLEDGE_DIR.glob('*.txt')))} files")


def retrieve(query: str, top_k: int = 4) -> str:
    """
    Retrieve the most relevant wine knowledge chunks for a query.

    Returns a single string of concatenated chunks to pass to the LLM.
    """
    try:
        embedder   = _get_embedder()
        collection = _get_collection()

        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )
        chunks = results["documents"][0] if results["documents"] else []
        return "\n\n---\n\n".join(chunks) if chunks else ""

    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""


def rebuild_index():
    """Force a full rebuild of the ChromaDB index (call after updating knowledge files)."""
    global _chroma_client, _collection

    import chromadb
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _collection = _chroma_client.create_collection(COLLECTION_NAME)
    _build_index()


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RAG engine...")
    ctx = retrieve("What does high volatile acidity taste like?")
    print("\nRetrieved context:\n")
    print(ctx[:600], "...")
"""ChromaDB vector store wrapper for semantic search across all DIKW layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wisdom.logging_config import get_logger

logger = get_logger("storage.vector")

# Collection names
EXPERIENCES_COLLECTION = "experiences"
KNOWLEDGE_COLLECTION = "knowledge"
WISDOM_COLLECTION = "wisdom"


class VectorStore:
    """ChromaDB wrapper managing three collections for experiences, knowledge, and wisdom."""

    def __init__(self, persist_dir: Path):
        import chromadb

        self.persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        # Use ChromaDB's built-in default embedding function (ONNX MiniLM-L6-v2)
        self.experiences = self.client.get_or_create_collection(
            name=EXPERIENCES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.knowledge = self.client.get_or_create_collection(
            name=KNOWLEDGE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.wisdom = self.client.get_or_create_collection(
            name=WISDOM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store initialized at %s", persist_dir)

    def _get_collection(self, layer: str):
        if layer == "experience":
            return self.experiences
        elif layer == "knowledge":
            return self.knowledge
        elif layer == "wisdom":
            return self.wisdom
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def add(
        self,
        layer: str,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a document embedding."""
        collection = self._get_collection(layer)
        meta = metadata or {}
        # ChromaDB metadata values must be str, int, float, or bool
        clean_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = ",".join(str(x) for x in v)
            else:
                clean_meta[k] = str(v)

        collection.upsert(
            ids=[id],
            documents=[text],
            metadatas=[clean_meta],
        )

    def search(
        self,
        layer: str,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Semantic search returning ids, distances, and metadata."""
        collection = self._get_collection(layer)
        if collection.count() == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        items = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)

            for i, doc_id in enumerate(ids):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1.0 - (distances[i] / 2.0)
                items.append({
                    "id": doc_id,
                    "similarity": similarity,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "document": documents[i] if i < len(documents) else "",
                })
        return items

    def delete(self, layer: str, id: str) -> None:
        """Remove a document from a collection."""
        collection = self._get_collection(layer)
        try:
            collection.delete(ids=[id])
        except Exception:
            pass  # Silently ignore if not found

    def count(self, layer: str) -> int:
        return self._get_collection(layer).count()

    def clear(self, layer: str) -> None:
        """Remove all documents from a collection."""
        collection = self._get_collection(layer)
        if collection.count() > 0:
            # Get all ids and delete them
            all_data = collection.get()
            if all_data["ids"]:
                collection.delete(ids=all_data["ids"])

    def warmup(self) -> None:
        """Pre-warm the embedding model by embedding a test string."""
        try:
            import chromadb.utils.embedding_functions as ef
            fn = ef.DefaultEmbeddingFunction()
            fn(["warmup test"])
            logger.info("Embedding model warmed up")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

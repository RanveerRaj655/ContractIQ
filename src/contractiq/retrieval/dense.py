"""
retrieval/dense.py
------------------
Dense retrieval via sentence-transformers embeddings + FAISS IndexFlatIP.

Vectors are L2-normalized before indexing, so inner product = cosine similarity.
The index is persisted to disk (data/index/) with a metadata sidecar so we skip
re-embedding on subsequent runs unless the corpus or model has changed.
"""

import hashlib
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from contractiq.chunking import Chunk

logger = logging.getLogger(__name__)


def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Lightweight hash over (doc_id, start, end) to detect corpus changes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.doc_id}:{c.start}:{c.end}\n".encode())
    return h.hexdigest()


class DenseRetriever:
    """Embeds chunks with a sentence-transformer model, indexes with FAISS
    IndexFlatIP (cosine via normalized vectors), and persists to disk."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_dir: Path | str = "data/index",
        batch_size: int = 256,
    ):
        self.chunks = chunks
        self._model_name = model_name
        self._index_dir = Path(index_dir)
        self._batch_size = batch_size

        self._model = SentenceTransformer(model_name)
        self._index = self._load_or_build()

    # ── Persistence helpers ──────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self._index_dir / "faiss.index"

    @property
    def _meta_path(self) -> Path:
        return self._index_dir / "meta.json"

    def _current_meta(self) -> dict:
        return {
            "model_name": self._model_name,
            "num_chunks": len(self.chunks),
            "corpus_hash": _corpus_fingerprint(self.chunks),
        }

    def _cache_is_valid(self) -> bool:
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        try:
            stored = json.loads(self._meta_path.read_text(encoding="utf-8"))
            current = self._current_meta()
            return (
                stored.get("model_name") == current["model_name"]
                and stored.get("num_chunks") == current["num_chunks"]
                and stored.get("corpus_hash") == current["corpus_hash"]
            )
        except (json.JSONDecodeError, KeyError):
            return False

    # ── Build / load ─────────────────────────────────────────────────────

    def _load_or_build(self) -> faiss.Index:
        if self._cache_is_valid():
            logger.info("Loading cached FAISS index from %s", self._index_dir)
            return faiss.read_index(str(self._index_path))

        logger.info(
            "Building FAISS index: %d chunks with %s ...",
            len(self.chunks),
            self._model_name,
        )
        texts = [c.text for c in self.chunks]
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-norm so IP = cosine
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Persist
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps(self._current_meta(), indent=2),
            encoding="utf-8",
        )
        logger.info("FAISS index saved to %s", self._index_dir)
        return index

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Encode query, search FAISS index, return top-k (Chunk, score) pairs."""
        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        q_vec = np.asarray(q_vec, dtype=np.float32)
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results when k > n
                continue
            results.append((self.chunks[idx], float(score)))
        return results

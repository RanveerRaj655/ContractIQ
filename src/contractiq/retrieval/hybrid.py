"""
retrieval/hybrid.py
-------------------
Hybrid retriever combining BM25 (sparse) and Dense (semantic) results via
Reciprocal Rank Fusion (RRF).

RRF is a rank-based fusion method that doesn't require score normalisation
across heterogeneous retrieval systems (BM25 scores are unbounded, cosine
similarity is [-1, 1]).  For each document d:

    RRF(d) = Σ  1 / (k + rank_i(d))

where the sum is over each retriever i and rank_i(d) is the 1-based rank of
d in retriever i's result list.  k is a smoothing constant (default 60, from
the original Cormack et al. 2009 paper).
"""

from contractiq.chunking import Chunk
from contractiq.retrieval.bm25 import BM25Retriever
from contractiq.retrieval.dense import DenseRetriever


def _chunk_key(chunk: Chunk) -> tuple[str, int, int]:
    """Identity key for matching chunks across retrievers."""
    return (chunk.doc_id, chunk.start, chunk.end)


class HybridRetriever:
    """Fuses BM25 and Dense retriever results with Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Retrieve from both BM25 and Dense, fuse with RRF, return top-k."""
        bm25_results = self.bm25.search(query, k=k)
        dense_results = self.dense.search(query, k=k)

        # Accumulate RRF scores
        rrf_scores: dict[tuple[str, int, int], float] = {}
        chunk_lookup: dict[tuple[str, int, int], Chunk] = {}

        for rank, (chunk, _score) in enumerate(bm25_results, start=1):
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_lookup[key] = chunk

        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_lookup[key] = chunk

        # Sort by fused score descending, take top-k
        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(chunk_lookup[key], score) for key, score in ranked]

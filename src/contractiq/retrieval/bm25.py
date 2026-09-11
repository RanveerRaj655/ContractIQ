"""
retrieval/bm25.py
-----------------
BM25 sparse retrieval baseline. Pure CPU, no embeddings/GPU needed - this is
what "search the docs" looked like before dense retrieval, and it's still a
surprisingly strong baseline for legal text (lots of exact terminology like
"Governing Law", "Indemnification" that benefits from lexical matching).

This is intentionally kept separate from any dense-retrieval module so the
two can be combined later (hybrid = BM25 + dense + Reciprocal Rank Fusion)
without entangling the code.
"""

import re

from rank_bm25 import BM25Okapi

from contractiq.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        # argsort descending, take top k
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in ranked_idx]

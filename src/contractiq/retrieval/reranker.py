"""
retrieval/reranker.py
---------------------
Cross-encoder reranking. 
Takes a set of candidate chunks (usually retrieved by a fast, recall-oriented
first stage like BM25 or Hybrid) and scores each (Query, Chunk) pair using a
cross-encoder model.

This is computationally expensive (which is why we only do it for the top-k
candidates) but provides much higher precision because the self-attention layers
can see the query and document tokens interacting simultaneously, unlike dense
retrieval which encodes them independently.
"""

from sentence_transformers import CrossEncoder

from contractiq.chunking import Chunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Rerank a list of candidate chunks for a given query.
        
        Args:
            query: The user's query string.
            candidates: A list of (Chunk, initial_score) tuples from the first stage.
            top_k: Number of chunks to return after reranking.
            
        Returns:
            A list of (Chunk, cross_encoder_score) tuples, sorted descending.
        """
        if not candidates:
            return []

        # Create (query, document) pairs for the cross-encoder
        pairs = [[query, chunk.text] for chunk, _score in candidates]
        
        # Predict scores
        scores = self._model.predict(pairs)
        
        # Zip chunks with new scores and sort
        scored_candidates = [
            (chunk, float(score)) 
            for (chunk, _), score in zip(candidates, scores)
        ]
        
        # Sort descending by the new cross-encoder score and take top-k
        ranked = sorted(scored_candidates, key=lambda kv: kv[1], reverse=True)[:top_k]
        
        return ranked

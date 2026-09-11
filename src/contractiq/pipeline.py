"""
pipeline.py
-----------
The unified retrieval pipeline. Wraps BM25, Dense, Hybrid, and Reranking 
into a single interface. The active strategy is controlled by config.py.
"""


from contractiq.chunking import Chunk
from contractiq.config import settings
from contractiq.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
)


class RetrievalPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        # Optionally allow passing pre-built instances to avoid redundant initialization during evaluation
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.chunks = chunks
        
        # Initialize retrievers lazily or use provided ones
        self.bm25 = bm25_retriever or BM25Retriever(chunks)
        self.dense = dense_retriever or DenseRetriever(
            chunks,
            model_name=settings.embedding_model,
            index_dir=settings.faiss_index_path,
            batch_size=settings.embedding_batch_size,
        )
        self.hybrid = hybrid_retriever or HybridRetriever(
            self.bm25, self.dense, rrf_k=settings.rrf_k
        )
        self.reranker = reranker or CrossEncoderReranker(
            model_name=settings.reranker_model
        )

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Retrieve top-k chunks using the active strategy defined in config.
        """
        strategy = settings.active_retrieval_strategy

        if strategy == "bm25":
            return self.bm25.search(query, k=k)
        elif strategy == "dense":
            return self.dense.search(query, k=k)
        elif strategy == "hybrid":
            return self.hybrid.search(query, k=k)
        elif strategy == "hybrid_rerank":
            # First stage: high recall, fetch more candidates
            candidates = self.hybrid.search(query, k=settings.rerank_fetch_k)
            # Second stage: high precision reranking
            return self.reranker.rerank(query, candidates, top_k=k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # Make the pipeline act like a standard retriever component
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return self.retrieve(query, k=k)

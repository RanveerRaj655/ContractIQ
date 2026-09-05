"""
retrieval — BM25, dense (FAISS), and hybrid (RRF) retrieval strategies.

Import any retriever directly from this package:
    from contractiq.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
"""

from contractiq.retrieval.bm25 import BM25Retriever
from contractiq.retrieval.dense import DenseRetriever
from contractiq.retrieval.hybrid import HybridRetriever
from contractiq.retrieval.reranker import CrossEncoderReranker

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "CrossEncoderReranker"]

"""
config.py
---------
Single source of truth for every tunable parameter in the ContractIQ
pipeline.  Import `settings` from anywhere; override via environment
variables or a .env file (pydantic-settings handles both automatically).

Usage:
    from contractiq.config import settings
    chunks = naive_fixed_chunk(doc_id, text, chunk_size=settings.chunk_size)
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# Project root is two levels up from this file: src/contractiq/config.py -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """All tunable parameters live here. Nothing should be a magic number elsewhere."""

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    corpus_dir: Path = _PROJECT_ROOT / "data" / "corpus"
    benchmark_mini: Path = _PROJECT_ROOT / "data" / "benchmarks" / "mini.json"
    benchmark_full: Path = _PROJECT_ROOT / "data" / "benchmarks" / "full.json"
    eval_results_dir: Path = _PROJECT_ROOT / "eval_results"

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    structure_target_size: int = 1000
    structure_max_size: int = 1600

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Dense retrieval ───────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    faiss_index_path: Path = _PROJECT_ROOT / "data" / "index"
    embedding_batch_size: int = 256

    # ── Hybrid retrieval (RRF) ────────────────────────────────────────────
    rrf_k: int = 60

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20

    # ── Pipeline ─────────────────────────────────────────────────────────
    active_retrieval_strategy: str = "hybrid_rerank"  # "bm25", "dense", "hybrid", "hybrid_rerank"

    # ── LLM / Generation ─────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.0

    # ── Eval thresholds ──────────────────────────────────────────────────
    f1_pass_threshold: float = 0.05  # minimum F1 to consider a query "answered"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

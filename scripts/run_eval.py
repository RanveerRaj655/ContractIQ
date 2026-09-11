"""
run_eval.py
-----------
Full comparison matrix: 4 retrieval configurations evaluated on the mini
benchmark with character-level precision/recall/F1.

Configs:
  1. naive_fixed    + BM25
  2. structure_aware + BM25
  3. structure_aware + Dense  (FAISS + bge-small-en-v1.5)
  4. structure_aware + Hybrid (BM25 + Dense via RRF)

Results are printed as a formatted table and saved to
eval_results/retrieval_comparison.json.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import Chunk, naive_fixed_chunk, structure_aware_chunk
from contractiq.config import settings
from contractiq.eval import RetrievedSpan, f1, precision_recall
from contractiq.retrieval.bm25 import BM25Retriever
from contractiq.retrieval.dense import DenseRetriever
from contractiq.retrieval.hybrid import HybridRetriever
from contractiq.retrieval.reranker import CrossEncoderReranker

CORPUS_DIR = settings.corpus_dir
BENCH_PATH = settings.benchmark_mini
TOP_K = settings.top_k


def load_benchmark() -> tuple[list[dict], set[str]]:
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    tests = bench["tests"]
    doc_ids = sorted({snip["file_path"] for t in tests for snip in t["snippets"]})
    return tests, doc_ids


def chunk_corpus(chunker_fn, doc_ids: set[str]) -> list[Chunk]:
    all_chunks = []
    for doc_id in doc_ids:
        text = (CORPUS_DIR / doc_id).read_text(encoding="utf-8")
        all_chunks.extend(chunker_fn(doc_id, text))
    return all_chunks


def evaluate(retriever, tests: list[dict], k: int = TOP_K):
    precisions, recalls, f1s = [], [], []
    per_category = defaultdict(lambda: {"p": [], "r": []})

    for test in tests:
        query = test["query"]
        results = retriever.search(query, k=k)
        retrieved_spans = [RetrievedSpan(c.doc_id, c.start, c.end) for c, _ in results]
        p, r = precision_recall(retrieved_spans, test["snippets"])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1(p, r))
        per_category[test["category"]]["p"].append(p)
        per_category[test["category"]]["r"].append(r)

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    return avg_p, avg_r, avg_f1, per_category


def main():
    tests, doc_ids = load_benchmark()
    print(f"Mini benchmark: {len(tests)} queries over {len(doc_ids)} contracts\n")

    # ── Chunk with both strategies ───────────────────────────────────────
    print("Chunking corpus ...")
    naive_chunks = chunk_corpus(naive_fixed_chunk, doc_ids)
    struct_chunks = chunk_corpus(structure_aware_chunk, doc_ids)
    print(f"  naive_fixed:     {len(naive_chunks)} chunks")
    print(f"  structure_aware: {len(struct_chunks)} chunks\n")

    # ── Build retrievers ─────────────────────────────────────────────────
    print("Building BM25 retrievers ...")
    bm25_naive = BM25Retriever(naive_chunks)
    bm25_struct = BM25Retriever(struct_chunks)

    print("Building Dense retriever (structure_aware chunks) ...")
    dense_struct = DenseRetriever(
        struct_chunks,
        model_name=settings.embedding_model,
        index_dir=settings.faiss_index_path,
        batch_size=settings.embedding_batch_size,
    )

    print("Building Hybrid retriever (BM25 + Dense, RRF) ...")
    hybrid_struct = HybridRetriever(
        bm25_struct, dense_struct, rrf_k=settings.rrf_k
    )

    print("Building Cross-Encoder Reranker ...\n")
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    # Helper to wrap the reranker for the eval loop
    class RerankedRetriever:
        def search(self, query: str, k: int = TOP_K):
            candidates = hybrid_struct.search(query, k=settings.rerank_fetch_k)
            return reranker.rerank(query, candidates, top_k=k)

    hybrid_rerank_struct = RerankedRetriever()

    # ── Define evaluation configs ────────────────────────────────────────
    configs = [
        ("naive_fixed + BM25",           bm25_naive),
        ("structure_aware + BM25",       bm25_struct),
        ("structure_aware + Dense",      dense_struct),
        ("structure_aware + Hybrid",     hybrid_struct),
        ("structure_aware + Hybrid + Rerank", hybrid_rerank_struct),
    ]

    results_summary = {}

    # ── Evaluate each config ─────────────────────────────────────────────
    for config_name, retriever in configs:
        t0 = time.time()
        avg_p, avg_r, avg_f1, _per_cat = evaluate(retriever, tests)
        eval_time = time.time() - t0

        print(f"=== {config_name} (top-{TOP_K}) ===")
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f1:.4f}")
        print(f"  eval time: {eval_time:.2f}s")
        print()

        results_summary[config_name] = {
            "precision": round(avg_p, 6),
            "recall": round(avg_r, 6),
            "f1": round(avg_f1, 6),
        }

    # ── Summary table ────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'Config':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for name, nums in results_summary.items():
        print(f"{name:<30} {nums['precision']:>10.4f} {nums['recall']:>10.4f} {nums['f1']:>10.4f}")
    print("=" * 65)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = settings.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

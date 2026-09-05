"""
run_baseline_eval.py
---------------------
Our first real, measured result. Loads the mini benchmark, builds a BM25
index over the corpus using two different chunking strategies, and reports
character-level precision/recall/F1 for each - isolating chunking as the
only variable, since retrieval method (BM25) is held constant.

This produces the "before / after" table that anchors every later claim
in the project (hybrid retrieval, re-ranking, etc. all get compared against
this baseline).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import naive_fixed_chunk, structure_aware_chunk
from contractiq.retrieval_bm25 import BM25Retriever
from contractiq.eval import precision_recall, f1, RetrievedSpan
from contractiq.config import settings

CORPUS_DIR = settings.corpus_dir
BENCH_PATH = settings.benchmark_mini
TOP_K = settings.top_k


def build_index(chunker_fn, doc_ids: set[str]):
    all_chunks = []
    for doc_id in doc_ids:
        text = (CORPUS_DIR / doc_id).read_text(encoding="utf-8")
        all_chunks.extend(chunker_fn(doc_id, text))
    return BM25Retriever(all_chunks)


def evaluate(retriever: BM25Retriever, tests: list[dict], k: int = TOP_K):
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
    with open(BENCH_PATH) as f:
        bench = json.load(f)
    tests = bench["tests"]
    doc_ids = {snip["file_path"] for t in tests for snip in t["snippets"]}
    print(f"Mini benchmark: {len(tests)} queries over {len(doc_ids)} contracts\n")

    results_summary = {}

    for name, chunker_fn in [("naive_fixed", naive_fixed_chunk), ("structure_aware", structure_aware_chunk)]:
        t0 = time.time()
        retriever = build_index(chunker_fn, doc_ids)
        build_time = time.time() - t0

        t0 = time.time()
        avg_p, avg_r, avg_f1, per_cat = evaluate(retriever, tests)
        eval_time = time.time() - t0

        print(f"=== {name} (top-{TOP_K}) ===")
        print(f"  chunks indexed: {len(retriever.chunks)}")
        print(f"  index build time: {build_time:.2f}s | eval time: {eval_time:.2f}s")
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f1:.4f}")
        print()

        results_summary[name] = {"precision": avg_p, "recall": avg_r, "f1": avg_f1, "num_chunks": len(retriever.chunks)}

    out_path = settings.eval_results_dir / "baseline_bm25_chunking_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

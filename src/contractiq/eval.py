"""
eval.py
-------
Character-level retrieval precision/recall, computed the same way as the
LegalBench-RAG paper: for each query, treat the ground-truth answer spans
and the retrieved chunk spans (grouped by document) as sets of character
positions, and measure their overlap.

  precision = overlap_chars / retrieved_chars
  recall    = overlap_chars / ground_truth_chars

We use interval arithmetic (merge + two-pointer sweep) instead of building
per-character boolean arrays, since contracts can be 300K+ characters long.
"""

from dataclasses import dataclass


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ivs = sorted(intervals)
    merged = [ivs[0]]
    for s, e in ivs[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlapping or touching
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def total_length(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in intervals)


def overlap_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Both lists are assumed already merged (non-overlapping, sorted)."""
    i, j, total = 0, 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass
class RetrievedSpan:
    doc_id: str
    start: int
    end: int


def precision_recall(
    retrieved: list[RetrievedSpan],
    ground_truth: list[dict],  # [{"file_path": ..., "span": [s, e]}, ...]
) -> tuple[float, float]:
    """Returns (precision, recall) for a single query."""

    # Group into per-doc interval lists
    ret_by_doc: dict[str, list[tuple[int, int]]] = {}
    for r in retrieved:
        ret_by_doc.setdefault(r.doc_id, []).append((r.start, r.end))

    gt_by_doc: dict[str, list[tuple[int, int]]] = {}
    for g in ground_truth:
        gt_by_doc.setdefault(g["file_path"], []).append(tuple(g["span"]))

    total_overlap = 0
    total_retrieved = 0
    total_gt = 0

    all_docs = set(ret_by_doc) | set(gt_by_doc)
    for doc in all_docs:
        r_ivs = merge_intervals(ret_by_doc.get(doc, []))
        g_ivs = merge_intervals(gt_by_doc.get(doc, []))
        total_overlap += overlap_length(r_ivs, g_ivs)
        total_retrieved += total_length(r_ivs)
        total_gt += total_length(g_ivs)

    precision = total_overlap / total_retrieved if total_retrieved > 0 else 0.0
    recall = total_overlap / total_gt if total_gt > 0 else 0.0
    return precision, recall


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    # Smoke test with a hand-built example
    gt = [{"file_path": "a.txt", "span": [100, 200]}]
    retrieved = [RetrievedSpan("a.txt", 150, 250)]  # 50% overlap with the 100-char gt span
    p, r = precision_recall(retrieved, gt)
    print(f"precision={p:.3f} recall={r:.3f} f1={f1(p, r):.3f}")
    assert abs(r - 0.5) < 1e-9, "recall should be 50 overlapping chars / 100 gt chars"
    assert abs(p - 0.5) < 1e-9, "precision should be 50 overlapping chars / 100 retrieved chars"
    print("OK")

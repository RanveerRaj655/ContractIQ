"""
test_eval.py
-------------
Hand-computed test cases for the character-level precision/recall evaluation.
These test the interval arithmetic in eval.py against known-correct results,
covering the critical edge cases for RAG evaluation.
"""

from contractiq.eval import (
    RetrievedSpan,
    f1,
    merge_intervals,
    overlap_length,
    precision_recall,
    total_length,
)

# ── Unit tests for interval helpers ──────────────────────────────────────

class TestMergeIntervals:
    def test_non_overlapping(self):
        assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]

    def test_overlapping(self):
        assert merge_intervals([(0, 15), (10, 25)]) == [(0, 25)]

    def test_touching(self):
        assert merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]

    def test_empty(self):
        assert merge_intervals([]) == []

    def test_single(self):
        assert merge_intervals([(5, 10)]) == [(5, 10)]

    def test_unsorted_input(self):
        assert merge_intervals([(20, 30), (0, 10), (5, 25)]) == [(0, 30)]


class TestOverlapLength:
    def test_partial_overlap(self):
        a = [(0, 100)]
        b = [(50, 150)]
        assert overlap_length(a, b) == 50

    def test_no_overlap(self):
        a = [(0, 10)]
        b = [(20, 30)]
        assert overlap_length(a, b) == 0

    def test_full_containment(self):
        a = [(0, 100)]
        b = [(10, 50)]
        assert overlap_length(a, b) == 40

    def test_identical(self):
        a = [(10, 20)]
        b = [(10, 20)]
        assert overlap_length(a, b) == 10


class TestTotalLength:
    def test_basic(self):
        assert total_length([(0, 10), (20, 30)]) == 20

    def test_empty(self):
        assert total_length([]) == 0


# ── precision_recall: hand-computed cases ────────────────────────────────

class TestPrecisionRecall:
    def test_50_percent_overlap(self):
        """Retrieved [150, 250) overlaps ground truth [100, 200) by exactly 50 chars.
        precision = 50 / 100 = 0.5 (50 overlap out of 100 retrieved)
        recall    = 50 / 100 = 0.5 (50 overlap out of 100 ground truth)"""
        gt = [{"file_path": "a.txt", "span": [100, 200]}]
        retrieved = [RetrievedSpan("a.txt", 150, 250)]
        p, r = precision_recall(retrieved, gt)
        assert abs(p - 0.5) < 1e-9
        assert abs(r - 0.5) < 1e-9

    def test_zero_overlap(self):
        """Retrieved span doesn't touch the ground truth at all.
        precision = 0, recall = 0."""
        gt = [{"file_path": "a.txt", "span": [100, 200]}]
        retrieved = [RetrievedSpan("a.txt", 300, 400)]
        p, r = precision_recall(retrieved, gt)
        assert p == 0.0
        assert r == 0.0

    def test_100_percent_overlap(self):
        """Retrieved span exactly matches ground truth.
        precision = 1.0, recall = 1.0."""
        gt = [{"file_path": "a.txt", "span": [100, 200]}]
        retrieved = [RetrievedSpan("a.txt", 100, 200)]
        p, r = precision_recall(retrieved, gt)
        assert abs(p - 1.0) < 1e-9
        assert abs(r - 1.0) < 1e-9

    def test_retrieved_contains_gt(self):
        """Retrieved is a superset of ground truth.
        overlap = 100 (entire GT span)
        precision = 100 / 500 = 0.2
        recall    = 100 / 100 = 1.0"""
        gt = [{"file_path": "a.txt", "span": [200, 300]}]
        retrieved = [RetrievedSpan("a.txt", 100, 600)]
        p, r = precision_recall(retrieved, gt)
        assert abs(p - 0.2) < 1e-9
        assert abs(r - 1.0) < 1e-9

    def test_wrong_doc_no_overlap(self):
        """Retrieved span covers the right char range but in the WRONG document.
        Should count as zero overlap — doc_id grouping is essential."""
        gt = [{"file_path": "contract_A.txt", "span": [100, 200]}]
        retrieved = [RetrievedSpan("contract_B.txt", 100, 200)]
        p, r = precision_recall(retrieved, gt)
        # precision = 0/100 = 0 (no overlap with any gt in contract_B)
        # recall = 0/100 = 0 (no overlap with gt in contract_A)
        assert p == 0.0
        assert r == 0.0

    def test_multiple_gt_spans_same_doc(self):
        """Two ground truth spans, retrieved covers one fully and misses the other.
        GT: [100,200) and [300,400) -> total_gt = 200
        Retrieved: [100,200) -> overlap = 100
        precision = 100/100 = 1.0
        recall = 100/200 = 0.5"""
        gt = [
            {"file_path": "a.txt", "span": [100, 200]},
            {"file_path": "a.txt", "span": [300, 400]},
        ]
        retrieved = [RetrievedSpan("a.txt", 100, 200)]
        p, r = precision_recall(retrieved, gt)
        assert abs(p - 1.0) < 1e-9
        assert abs(r - 0.5) < 1e-9

    def test_no_retrieved_spans(self):
        """Nothing retrieved => precision and recall are both 0."""
        gt = [{"file_path": "a.txt", "span": [100, 200]}]
        p, r = precision_recall([], gt)
        assert p == 0.0
        assert r == 0.0

    def test_no_gt_spans(self):
        """No ground truth => recall = 0, precision = 0 (nothing to match)."""
        retrieved = [RetrievedSpan("a.txt", 100, 200)]
        p, r = precision_recall(retrieved, [])
        assert p == 0.0
        assert r == 0.0


# ── f1 ───────────────────────────────────────────────────────────────────

class TestF1:
    def test_perfect(self):
        assert abs(f1(1.0, 1.0) - 1.0) < 1e-9

    def test_zero(self):
        assert f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        """f1(0.5, 0.5) = 2*0.5*0.5/(0.5+0.5) = 0.5"""
        assert abs(f1(0.5, 0.5) - 0.5) < 1e-9

    def test_asymmetric(self):
        """f1(1.0, 0.5) = 2*1*0.5/1.5 = 2/3"""
        assert abs(f1(1.0, 0.5) - 2/3) < 1e-9

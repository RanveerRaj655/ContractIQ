# ContractIQ — Starter Package

This is pre-processed data + validated code from the planning session, ready to drop into
your Antigravity project so you don't have to redo the data prep step.

## What's in here

```
data/
  corpus/            510 CUAD contracts as plain .txt files
  benchmarks/
    full.json        6,702 ground-truth queries (all 510 contracts, all 41 clause categories)
    mini.json        532 ground-truth queries (40 contracts, stratified to cover all 41 categories)
src/contractiq/
  chunking.py        naive_fixed_chunk() + structure_aware_chunk() — both tested
  eval.py            character-level precision/recall/F1 (LegalBench-RAG methodology) — tested
  retrieval_bm25.py  BM25 baseline retriever
scripts/
  prepare_data.py    the script that generated data/ from raw CUAD (reference only, already run)
  run_baseline_eval.py   runs the baseline comparison, writes to eval_results/
eval_results/
  baseline_bm25_chunking_comparison.json   the Day-1 baseline numbers already computed
requirements.txt
```

## Ground truth schema (used by both benchmark files)

```json
{
  "tests": [
    {
      "query": "Highlight the parts (if any) of this contract related to \"Governing Law\"...",
      "category": "Governing Law",
      "contract_title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
      "snippets": [
        {"file_path": "LIMEENERGYCO_....txt", "span": [1234, 1289]}
      ]
    }
  ]
}
```

`span` is a `[start, end)` character index range into the corresponding file in `data/corpus/`.
A query can have multiple snippets (multiple answer locations in the same or different documents).

## Already-validated baseline (Day 1, BM25 only)

| Chunking | Precision | Recall | F1 |
|---|---|---|---|
| naive_fixed | 0.0109 | 0.0810 | 0.0171 |
| structure_aware | 0.0105 | 0.0850 | 0.0170 |

Low absolute numbers are expected — CUAD answer spans are short phrases inside larger chunks,
which caps precision hard. This is the same pattern reported in the LegalBench-RAG paper. The
point of these numbers is to be your **baseline to beat**, not an end result.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/run_baseline_eval.py
```

## Source / attribution

Built from CUAD v1 (Hendrycks et al., 2021), Creative Commons Attribution 4.0.
Original: https://github.com/TheAtticusProject/cuad
Benchmark methodology mirrors LegalBench-RAG (Pipitone & Houir Alami, 2024): https://arxiv.org/abs/2408.10343

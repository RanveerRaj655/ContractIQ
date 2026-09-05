"""
prepare_data.py
----------------
Converts the raw CUAD v1 SQuAD-style JSON into:
  1. data/corpus/*.txt          -> one plain-text file per contract
  2. data/benchmarks/full.json  -> every answerable query, LegalBench-RAG-style schema
  3. data/benchmarks/mini.json  -> a stratified subset (one query per category per a
                                   sample of contracts) for fast iteration

Ground truth schema (mirrors LegalBench-RAG so our eval code / reported numbers are
directly comparable to the published benchmark):

{
  "tests": [
    {
      "query": "Highlight the parts ... related to \"Governing Law\" ...",
      "category": "Governing Law",
      "snippets": [
        {"file_path": "LIMEENERGYCO_...txt", "span": [123, 456]}
      ]
    },
    ...
  ]
}
"""

import json
import random
import re
from pathlib import Path

RAW_JSON = Path("/home/claude/contractiq/data/raw/extracted/CUADv1.json")
CORPUS_DIR = Path("/home/claude/contractiq/data/corpus")
BENCH_DIR = Path("/home/claude/contractiq/data/benchmarks")

random.seed(42)


def safe_filename(title: str) -> str:
    """Contract titles can contain slashes/odd chars; make them filesystem-safe."""
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)
    return name[:150] + ".txt"


def extract_category(question: str) -> str:
    """CUAD questions are templated: '...related to "Category Name". Details: ...'"""
    m = re.search(r'"([^"]+)"', question)
    return m.group(1) if m else "Unknown"


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_JSON) as f:
        data = json.load(f)

    full_tests = []
    contracts_meta = []  # for building the mini stratified sample

    for contract in data["data"]:
        title = contract["title"]
        fname = safe_filename(title)
        paragraph = contract["paragraphs"][0]
        context = paragraph["context"]

        # Write corpus file
        (CORPUS_DIR / fname).write_text(context, encoding="utf-8")

        contract_categories_present = []

        for qa in paragraph["qas"]:
            if qa["is_impossible"]:
                continue
            category = extract_category(qa["question"])
            snippets = [
                {"file_path": fname, "span": [ans["answer_start"], ans["answer_start"] + len(ans["text"])]}
                for ans in qa["answers"]
            ]
            test_case = {
                "query": qa["question"],
                "category": category,
                "contract_title": title,
                "snippets": snippets,
            }
            full_tests.append(test_case)
            contract_categories_present.append(category)

        contracts_meta.append({"fname": fname, "categories": contract_categories_present})

    # ---- Full benchmark ----
    full_bench = {"tests": full_tests}
    with open(BENCH_DIR / "full.json", "w") as f:
        json.dump(full_bench, f, indent=2)

    # ---- Mini benchmark: stratified sample ----
    # Sample ~35 contracts, but prioritize ones covering diverse / rarer categories
    # so the mini set still has decent coverage of all 41 clause types.
    all_categories = sorted({t["category"] for t in full_tests})
    contracts_by_fname = {c["fname"]: c for c in contracts_meta}

    chosen_fnames = set()
    # Greedy set-cover: keep adding the contract that covers the most
    # not-yet-covered categories, until all categories are covered or we
    # hit our contract budget.
    covered = set()
    remaining_meta = list(contracts_meta)
    budget = 40
    while remaining_meta and len(covered) < len(all_categories) and len(chosen_fnames) < budget:
        remaining_meta.sort(key=lambda c: len(set(c["categories"]) - covered), reverse=True)
        best = remaining_meta.pop(0)
        new_cov = set(best["categories"]) - covered
        if not new_cov and len(chosen_fnames) > 0:
            break
        covered |= set(best["categories"])
        chosen_fnames.add(best["fname"])

    # Top up to a nice round number with random additional contracts for volume
    extra_pool = [c["fname"] for c in contracts_meta if c["fname"] not in chosen_fnames]
    random.shuffle(extra_pool)
    while len(chosen_fnames) < budget and extra_pool:
        chosen_fnames.add(extra_pool.pop())

    mini_tests = [t for t in full_tests if t["snippets"][0]["file_path"] in chosen_fnames]
    mini_bench = {"tests": mini_tests}
    with open(BENCH_DIR / "mini.json", "w") as f:
        json.dump(mini_bench, f, indent=2)

    # ---- Report ----
    print(f"Contracts written to corpus: {len(contracts_meta)}")
    print(f"Full benchmark queries: {len(full_tests)}")
    print(f"Mini benchmark: {len(chosen_fnames)} contracts, {len(mini_tests)} queries")
    print(f"Categories covered in mini: {len({t['category'] for t in mini_tests})}/{len(all_categories)}")


if __name__ == "__main__":
    main()

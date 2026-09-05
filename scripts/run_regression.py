"""
run_regression.py
-----------------
Runs the end-to-end RAG pipeline over the questions in tests/regression_questions.json.
Outputs a markdown file with side-by-side comparisons of the expected vs actual answers.
"""

import json
import sys
from pathlib import Path

# Add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline
from contractiq.generation import LLMClient


def load_corpus() -> list:
    print(f"Loading 40 benchmark contracts from {settings.corpus_dir}...")
    with open(settings.benchmark_mini, encoding="utf-8") as f:
        bench = json.load(f)
    doc_ids = sorted({snip["file_path"] for t in bench["tests"] for snip in t["snippets"]})
    corpus_files = [settings.corpus_dir / doc_id for doc_id in doc_ids]

    all_chunks = []
    for filepath in corpus_files:
        text = filepath.read_text(encoding="utf-8")
        all_chunks.extend(structure_aware_chunk(filepath.name, text))
    return all_chunks


def main():
    test_file = Path("tests/regression_questions.json")
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        sys.exit(1)

    with open(test_file, encoding="utf-8") as f:
        questions = json.load(f)

    all_chunks = load_corpus()

    print(f"Initializing RetrievalPipeline (strategy: {settings.active_retrieval_strategy})...")
    pipeline = RetrievalPipeline(all_chunks)
    
    print("Initializing LLMClient...")
    llm = LLMClient()

    results_md = "# Regression Test Results\n\n"

    for i, q in enumerate(questions, 1):
        query = q["query"]
        expected = q["expected_answer"]
        q_type = q["type"]
        contract = q["contract"]

        print(f"\nProcessing [{i}/{len(questions)}] ({q_type}): {query[:50]}...")
        
        # We append the target contract to the query for the retriever so it knows which document to search in
        targeted_query = f"{query} Document: {contract}"
        
        retrieved_chunks = pipeline.retrieve(targeted_query, k=settings.top_k)
        
        if not retrieved_chunks:
            actual = "No chunks retrieved."
        else:
            actual = llm.generate_answer(query, retrieved_chunks)

        results_md += f"## Question {i} ({q_type.capitalize()})\n"
        results_md += f"**Target Document**: `{contract}`\n\n"
        results_md += f"**Query**: {query}\n\n"
        results_md += "| Expected | Actual |\n"
        results_md += "|----------|--------|\n"
        # Clean up newlines for the markdown table
        expected_clean = expected.replace("\n", " ")
        actual_clean = actual.replace("\n", "<br>")
        results_md += f"| {expected_clean} | {actual_clean} |\n\n"

    out_dir = settings.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regression_results.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(results_md)
        
    print(f"\nDone! Results written to {out_path}")


if __name__ == "__main__":
    main()

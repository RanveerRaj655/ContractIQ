"""
run_regression.py
-----------------
Runs the end-to-end RAG pipeline with guardrails over regression_questions.json.
Outputs a markdown file for visual comparison and a structured JSONL for observability.
"""

import json
import sys
import time
from pathlib import Path

# Add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline
from contractiq.generation import LLMClient
from contractiq.guardrails import scan_input, should_abstain, check_hallucination


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
    pipeline = RetrievalPipeline(all_chunks)
    llm = LLMClient()

    out_dir = settings.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "regression_results.md"
    out_jsonl = out_dir / "query_log.jsonl"
    
    # Open jsonl for appending observability logs
    log_file = open(out_jsonl, "w", encoding="utf-8")
    results_md = "# Regression Test Results (Guarded Pipeline)\n\n"

    for i, q in enumerate(questions, 1):
        start_time = time.time()
        
        query = q["query"]
        expected = q["expected_answer"]
        q_type = q["type"]
        contract = q["contract"]

        print(f"\nProcessing [{i}/{len(questions)}] ({q_type}): {query[:50]}...")
        
        # 1. Input Filter
        input_flags = scan_input(query)
        
        # We append the target contract to the query for the retriever
        targeted_query = f"{query} Document: {contract}"
        
        # 2. Retrieve
        retrieved_chunks = pipeline.retrieve(targeted_query, k=settings.top_k)
        
        # 3. Abstention Check
        top_scores = [score for chunk, score in retrieved_chunks]
        top_score = max(top_scores) if top_scores else 0.0
        abstained = should_abstain(top_scores)
        
        actual = ""
        hallucination_verdict = "N/A"
        hallucination_reason = ""
        
        if abstained:
            actual = f"Insufficient context (Score {top_score:.4f} < Threshold {settings.abstention_threshold}). Abstained."
        elif not retrieved_chunks:
            actual = "No chunks retrieved."
        else:
            # 4. Generate
            actual = llm.generate_answer(query, retrieved_chunks)
            # 5. Hallucination Check
            hallucination_result = check_hallucination(llm, query, actual, retrieved_chunks)
            hallucination_verdict = hallucination_result["verdict"]
            hallucination_reason = hallucination_result["reason"]

        latency_ms = int((time.time() - start_time) * 1000)

        # Build observability log
        log_entry = {
            "query_index": i,
            "query_type": q_type,
            "query": query,
            "input_flags": input_flags,
            "top_retrieval_score": top_score,
            "abstained": abstained,
            "latency_ms": latency_ms,
            "hallucination_verdict": hallucination_verdict,
            "hallucination_reason": hallucination_reason,
            "answer": actual
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        results_md += f"## Question {i} ({q_type.capitalize()})\n"
        results_md += f"**Target Document**: `{contract}`\n"
        results_md += f"**Abstained**: {abstained} (Score: {top_score:.4f})\n"
        results_md += f"**Hallucination Verdict**: {hallucination_verdict}\n\n"
        results_md += f"**Query**: {query}\n\n"
        results_md += "| Expected | Actual |\n"
        results_md += "|----------|--------|\n"
        expected_clean = expected.replace("\n", " ")
        actual_clean = actual.replace("\n", "<br>")
        results_md += f"| {expected_clean} | {actual_clean} |\n\n"

    log_file.close()
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(results_md)
        
    print(f"\nDone! Observability logs written to {out_jsonl}")
    print(f"Results written to {out_md}")


if __name__ == "__main__":
    main()

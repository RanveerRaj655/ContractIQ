"""
calibrate_threshold.py
----------------------
Runs retrieval on the benchmark queries, calculates recall, and logs the 
top retrieval score for each query to help determine a good abstention threshold.
"""

import json
import sys
from pathlib import Path

# Add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline
from contractiq.eval import precision_recall, RetrievedSpan


def main():
    print(f"Loading 40 benchmark contracts from {settings.corpus_dir}...")
    with open(settings.benchmark_mini, encoding="utf-8") as f:
        bench = json.load(f)
        
    tests = bench["tests"]
    doc_ids = sorted({snip["file_path"] for t in tests for snip in t["snippets"]})
    corpus_files = [settings.corpus_dir / doc_id for doc_id in doc_ids]

    all_chunks = []
    for filepath in corpus_files:
        text = filepath.read_text(encoding="utf-8")
        all_chunks.extend(structure_aware_chunk(filepath.name, text))

    print(f"Initializing RetrievalPipeline (strategy: {settings.active_retrieval_strategy})...")
    pipeline = RetrievalPipeline(all_chunks)

    results_data = []

    for i, t in enumerate(tests, 1):
        query = t["query"]
        ground_truth_snippets = t["snippets"]
        
        # We append the target document ID from the first snippet to simulate our targeted queries
        target_doc = ground_truth_snippets[0]["file_path"] if ground_truth_snippets else ""
        targeted_query = f"{query} Document: {target_doc}"

        retrieved_chunks_scores = pipeline.retrieve(targeted_query, k=settings.top_k)
        
        retrieved_spans = [
            RetrievedSpan(c.doc_id, c.start, c.end) for c, score in retrieved_chunks_scores
        ]
        
        # Calculate recall
        precision, recall = precision_recall(retrieved_spans, ground_truth_snippets)
        
        # Get top score
        top_score = retrieved_chunks_scores[0][1] if retrieved_chunks_scores else 0.0
        
        results_data.append({
            "query_index": i,
            "recall": recall,
            "top_score": top_score
        })
        
        print(f"Query {i}/{len(tests)} | Recall: {recall:.2f} | Top Score: {top_score:.4f}", end="\r")

    print("\n\n--- Calibration Results ---")
    
    # Bucket by successful retrieval (recall > 0.3) vs failed retrieval (recall = 0)
    successful_scores = [r["top_score"] for r in results_data if r["recall"] > 0.3]
    failed_scores = [r["top_score"] for r in results_data if r["recall"] == 0.0]
    
    if successful_scores:
        print(f"Successful Retrievals (Recall > 0.3): {len(successful_scores)}")
        print(f"  Min Score: {min(successful_scores):.4f}")
        print(f"  Avg Score: {sum(successful_scores)/len(successful_scores):.4f}")
        print(f"  Max Score: {max(successful_scores):.4f}")
        
    if failed_scores:
        print(f"Failed Retrievals (Recall == 0.0): {len(failed_scores)}")
        print(f"  Min Score: {min(failed_scores):.4f}")
        print(f"  Avg Score: {sum(failed_scores)/len(failed_scores):.4f}")
        print(f"  Max Score: {max(failed_scores):.4f}")

    print("\nRecommended Threshold: Look for a value that separates the min score of successful retrievals from the max score of failed retrievals.")

if __name__ == "__main__":
    main()

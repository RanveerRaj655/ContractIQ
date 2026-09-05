"""
ask.py
------
CLI script for end-to-end RAG (Retrieval + Generation).

Usage:
  python scripts/ask.py "What is the governing law?"
"""

import argparse
import sys
import json
from pathlib import Path

# Add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline
from contractiq.generation import LLMClient


def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG: Ask a question about the contracts.")
    parser.add_argument("query", type=str, help="The query string to test")
    args = parser.parse_args()

    print(f"Loading 40 benchmark contracts from {settings.corpus_dir}...")
    with open(settings.benchmark_mini, encoding="utf-8") as f:
        bench = json.load(f)
    doc_ids = sorted({snip["file_path"] for t in bench["tests"] for snip in t["snippets"]})
    corpus_files = [settings.corpus_dir / doc_id for doc_id in doc_ids]

    if not corpus_files:
        print("Error: No text files found.")
        sys.exit(1)

    all_chunks = []
    for filepath in corpus_files:
        text = filepath.read_text(encoding="utf-8")
        all_chunks.extend(structure_aware_chunk(filepath.name, text))

    print(f"Initializing RetrievalPipeline (strategy: {settings.active_retrieval_strategy})...")
    pipeline = RetrievalPipeline(all_chunks)

    print(f"Retrieving chunks for query: '{args.query}'...")
    results = pipeline.retrieve(args.query, k=settings.top_k)
    
    if not results:
        print("No chunks retrieved.")
        return

    print("Generating answer...")
    llm = LLMClient()
    answer = llm.generate_answer(args.query, results)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES USED:")
    print("=" * 60)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"[Chunk {i}] Document: {chunk.doc_id} (Score: {score:.4f})")

if __name__ == "__main__":
    main()

"""
debug_query.py
--------------
CLI tool to test queries manually through the active retrieval pipeline.
Useful for sanity checking results by hand.

Usage:
  python scripts/debug_query.py "What is the governing law?"
"""

import argparse
import sys
from pathlib import Path

# Add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline


def main():
    parser = argparse.ArgumentParser(description="Debug a query through the retrieval pipeline.")
    parser.add_argument("query", type=str, help="The query string to test")
    parser.add_argument("--k", type=int, default=settings.top_k, help="Number of chunks to return")
    args = parser.parse_args()

    print(f"Loading corpus from {settings.corpus_dir}...")
    corpus_files = list(settings.corpus_dir.glob("*.txt"))
    if not corpus_files:
        print("Error: No text files found in the corpus directory.")
        sys.exit(1)

    print(f"Chunking {len(corpus_files)} documents...")
    all_chunks = []
    for filepath in corpus_files:
        text = filepath.read_text(encoding="utf-8")
        all_chunks.extend(structure_aware_chunk(filepath.name, text))

    print(f"Initializing RetrievalPipeline (strategy: {settings.active_retrieval_strategy})...")
    pipeline = RetrievalPipeline(all_chunks)

    print(f"\nSearching for: '{args.query}'\n" + "=" * 50)
    
    results = pipeline.retrieve(args.query, k=args.k)
    
    if not results:
        print("No results found.")
        return

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\n[Rank {rank}] Score: {score:.4f} | Document: {chunk.doc_id} | Chars: {chunk.start}-{chunk.end}")
        print("-" * 50)
        print(chunk.text.strip())
        print("-" * 50)

if __name__ == "__main__":
    main()

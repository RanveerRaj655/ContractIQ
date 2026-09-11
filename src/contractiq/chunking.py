"""
chunking.py
-----------
Two chunking strategies, both returning chunks that carry their original
character offsets in the source document. That offset tracking is essential:
it's what lets us compute character-level precision/recall against the CUAD
ground-truth spans later (same methodology as LegalBench-RAG).

1. naive_fixed_chunk   - fixed-size sliding window (the "everyone's tutorial" baseline)
2. structure_aware_chunk - splits on legal-document structure (numbered clauses,
                           section headers, blank-line paragraph breaks) and
                           merges tiny fragments up to a target size.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str          # corpus filename
    text: str
    start: int            # char offset in original document
    end: int              # char offset in original document (exclusive)


def naive_fixed_chunk(doc_id: str, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Baseline: fixed-size sliding window over raw characters. No respect for
    sentence/clause boundaries. This is what most RAG tutorials ship with."""
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk_size, n)
        chunks.append(Chunk(doc_id=doc_id, text=text[pos:end], start=pos, end=end))
        if end == n:
            break
        pos += step
    return chunks


# Legal contracts are heavily structured: numbered sections ("1.", "1.1", "Section 3"),
# ALL-CAPS headers, and blank-line paragraph breaks. We split on these boundaries first,
# then greedily merge adjacent small pieces up to a target chunk size so we don't end up
# with hundreds of tiny 1-sentence chunks (bad for retrieval recall).
_SECTION_BREAK_RE = re.compile(
    r"""
    (?:\n\s*\n)                                   # blank-line paragraph break
    | (?:\n(?=\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s+))) # numbered clause like "1.2 " at line start
    | (?:\n(?=\s*(?:SECTION|Section|ARTICLE|Article)\s+[\dIVXLC]+)) # section/article headers
    """,
    re.VERBOSE,
)


def structure_aware_chunk(doc_id: str, text: str, target_size: int = 1000, max_size: int = 1600) -> list[Chunk]:
    """Split on legal-document structural boundaries, then merge small adjacent
    pieces up to ~target_size so chunks stay retrieval-friendly, and hard-cap
    at max_size so we never blow past what fits comfortably in a prompt."""

    # Find split points
    boundaries = [0]
    for m in _SECTION_BREAK_RE.finditer(text):
        boundaries.append(m.start())
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    # Raw structural segments (may be tiny, e.g. a single header line)
    raw_segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg_text = text[s:e]
        if seg_text.strip():
            raw_segments.append((s, e))

    if not raw_segments:
        return naive_fixed_chunk(doc_id, text, chunk_size=target_size, overlap=0)

    # Greedily merge adjacent segments up to target_size, never exceeding max_size
    merged: list[Chunk] = []
    cur_start, cur_end = raw_segments[0]
    for s, e in raw_segments[1:]:
        candidate_len = e - cur_start
        if candidate_len <= max_size:
            cur_end = e
        else:
            merged.append(Chunk(doc_id=doc_id, text=text[cur_start:cur_end], start=cur_start, end=cur_end))
            cur_start, cur_end = s, e
    merged.append(Chunk(doc_id=doc_id, text=text[cur_start:cur_end], start=cur_start, end=cur_end))

    # If a single segment is itself bigger than max_size (rare: giant unbroken
    # paragraph), fall back to fixed-window splitting just for that piece.
    final: list[Chunk] = []
    for c in merged:
        if len(c.text) > max_size:
            sub = naive_fixed_chunk(doc_id, c.text, chunk_size=target_size, overlap=100)
            for sc in sub:
                final.append(Chunk(doc_id=doc_id, text=sc.text, start=c.start + sc.start, end=c.start + sc.end))
        else:
            final.append(c)

    return final


if __name__ == "__main__":
    # Quick smoke test against a real contract
    from pathlib import Path

    sample = next(Path("/home/claude/contractiq/data/corpus").glob("*.txt"))
    text = sample.read_text()

    naive = naive_fixed_chunk(sample.name, text)
    structured = structure_aware_chunk(sample.name, text)

    print(f"Document: {sample.name} ({len(text)} chars)")
    print(f"Naive fixed chunking:      {len(naive)} chunks, avg size {sum(len(c.text) for c in naive)//len(naive)}")
    print(f"Structure-aware chunking:  {len(structured)} chunks, avg size {sum(len(c.text) for c in structured)//len(structured)}")

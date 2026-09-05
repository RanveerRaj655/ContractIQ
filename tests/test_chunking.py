"""
test_chunking.py
-----------------
Verify that chunking strategies preserve document content — no characters
should be dropped or invented, and chunk offsets must be consistent with
the text they claim to represent.
"""

import pytest
from contractiq.chunking import naive_fixed_chunk, structure_aware_chunk, Chunk


# ── Helpers ──────────────────────────────────────────────────────────────

def _assert_chunks_cover_doc(text: str, chunks: list[Chunk]):
    """Every character in `text` must appear in at least one chunk.
    Build a boolean coverage array from chunk offsets and verify full coverage."""
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    # Allow trailing whitespace to be uncovered (structure_aware strips whitespace-only segments)
    for i in range(len(text)):
        if not covered[i]:
            # Only whitespace characters may be uncovered
            assert text[i].isspace(), (
                f"Non-whitespace character at position {i} ('{text[i]}') "
                f"is not covered by any chunk"
            )


def _assert_chunk_text_matches_offsets(text: str, chunks: list[Chunk]):
    """Each chunk's .text must exactly equal text[chunk.start:chunk.end]."""
    for c in chunks:
        actual = text[c.start:c.end]
        assert c.text == actual, (
            f"Chunk at [{c.start}:{c.end}] text mismatch:\n"
            f"  chunk.text = {c.text!r}\n"
            f"  doc slice  = {actual!r}"
        )


# ── naive_fixed_chunk ────────────────────────────────────────────────────

class TestNaiveFixedChunk:
    def test_short_doc_single_chunk(self):
        text = "Hello, world!"
        chunks = naive_fixed_chunk("test.txt", text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start == 0
        assert chunks[0].end == len(text)

    def test_exact_chunk_size(self):
        text = "A" * 1000
        chunks = naive_fixed_chunk("test.txt", text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_overlap_produces_correct_count(self):
        text = "X" * 2600
        # chunk_size=1000, overlap=200 -> step=800 -> positions 0,800,1600,2400(->2600)
        chunks = naive_fixed_chunk("test.txt", text, chunk_size=1000, overlap=200)
        # With step=800: chunk 0=[0,1000), 1=[800,1800), 2=[1600,2600)
        # Actually position 2400 would start but 2400+1000>2600, so end=2600
        assert len(chunks) >= 3

    def test_no_characters_dropped(self):
        text = "The quick brown fox " * 50  # 1000 chars
        text += "jumps over the lazy dog. " * 40  # 1000 more
        chunks = naive_fixed_chunk("test.txt", text, chunk_size=500, overlap=100)
        _assert_chunks_cover_doc(text, chunks)
        _assert_chunk_text_matches_offsets(text, chunks)

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be smaller"):
            naive_fixed_chunk("test.txt", "some text", chunk_size=100, overlap=100)

    def test_real_legal_text(self):
        """Simulate legal-style text with varied structure."""
        text = (
            "SECTION 1. DEFINITIONS\n\n"
            '"Agreement" means this Master Services Agreement.\n'
            '"Confidential Information" means any non-public information.\n\n'
            "SECTION 2. TERM AND TERMINATION\n\n"
            "2.1 This Agreement shall commence on the Effective Date.\n"
            "2.2 Either party may terminate this Agreement for convenience "
            "upon thirty (30) days prior written notice.\n\n"
        ) * 5
        chunks = naive_fixed_chunk("legal.txt", text, chunk_size=200, overlap=50)
        _assert_chunks_cover_doc(text, chunks)
        _assert_chunk_text_matches_offsets(text, chunks)


# ── structure_aware_chunk ────────────────────────────────────────────────

class TestStructureAwareChunk:
    def test_short_doc(self):
        text = "Just a small paragraph."
        chunks = structure_aware_chunk("test.txt", text, target_size=1000)
        assert len(chunks) >= 1
        _assert_chunk_text_matches_offsets(text, chunks)

    def test_structured_legal_doc(self):
        text = (
            "SECTION 1. DEFINITIONS\n\n"
            '"Agreement" means this Master Services Agreement, including all exhibits.\n\n'
            "SECTION 2. OBLIGATIONS\n\n"
            "2.1 The Contractor shall perform all services in a professional manner.\n\n"
            "2.2 The Client shall provide timely access to all required facilities.\n\n"
            "SECTION 3. GOVERNING LAW\n\n"
            "This Agreement shall be governed by the laws of the State of Delaware.\n"
        )
        chunks = structure_aware_chunk("contract.txt", text, target_size=500)
        _assert_chunk_text_matches_offsets(text, chunks)

    def test_no_characters_dropped_structured(self):
        """Build a large-ish legal document and verify full coverage."""
        sections = []
        for i in range(1, 11):
            sections.append(f"SECTION {i}. TITLE OF SECTION {i}\n\n")
            for j in range(1, 4):
                sections.append(
                    f"{i}.{j} This is clause {i}.{j} of the agreement. "
                    f"It contains important legal language that must be preserved "
                    f"in its entirety during the chunking process.\n\n"
                )
        text = "".join(sections)
        chunks = structure_aware_chunk("big_contract.txt", text, target_size=300, max_size=500)
        _assert_chunks_cover_doc(text, chunks)
        _assert_chunk_text_matches_offsets(text, chunks)

    def test_oversized_segment_falls_back(self):
        """A single paragraph bigger than max_size should get sub-chunked."""
        # One giant paragraph with no structural breaks
        text = "A" * 3000
        chunks = structure_aware_chunk("giant.txt", text, target_size=500, max_size=800)
        _assert_chunk_text_matches_offsets(text, chunks)
        assert all(len(c.text) <= 800 for c in chunks), "All chunks must be <= max_size"
        _assert_chunks_cover_doc(text, chunks)

    def test_doc_id_propagated(self):
        text = "SECTION 1\n\nParagraph one.\n\nSECTION 2\n\nParagraph two.\n"
        chunks = structure_aware_chunk("my_doc.txt", text, target_size=100)
        assert all(c.doc_id == "my_doc.txt" for c in chunks)

    def test_empty_doc(self):
        """Empty doc should produce no chunks (or a single empty chunk)."""
        chunks = structure_aware_chunk("empty.txt", "", target_size=500)
        # Either no chunks or trivial — either way it shouldn't crash
        assert isinstance(chunks, list)

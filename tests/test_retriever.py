"""
Tests for the retriever module (sentence segmentation, chunking, ranking).
"""

import pytest
from retriever import (
    Sentence,
    ExtractedChunk,
    segment_sentences,
    create_chunks,
    rank_chunks_by_salience,
    retrieve,
)
from config import RetrieverConfig


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
1. The appellant filed a petition before the Hon'ble High Court under Article 226 of the Constitution of India.

2. The respondent, State of Maharashtra, opposed the petition on the ground that the appellant had an alternative remedy under Section 64 of the Act of 2013.

3. The Court held that the acquisition was for a bonafide public purpose. The Social Impact Assessment study was found to be substantially compliant with the provisions of Sections 4 and 5 of the Act.

4. However, the Court observed that the SIA report did not contain any assessment of alternative sites as mandated by Section 4(2)(e) of the Act. No public hearing was conducted as required by Section 5.

5. In the result, the appeal is allowed in part. The notification under Section 11(1) is set aside. The matter is remanded to the State Government with a direction to conduct a fresh SIA study within six months.
""".strip()


# ────────────────────────────────────────────────────────────
# Sentence Segmentation
# ────────────────────────────────────────────────────────────

class TestSentenceSegmentation:
    def test_basic_segmentation(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        assert len(sentences) > 0
        assert all(isinstance(s, Sentence) for s in sentences)

    def test_sentences_have_offsets(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        for s in sentences:
            assert s.start_char >= 0
            assert s.end_char > s.start_char
            assert s.text.strip()

    def test_sentences_are_ordered(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        for i in range(1, len(sentences)):
            assert sentences[i].index > sentences[i - 1].index

    def test_legal_abbreviations_not_split(self):
        """'Hon'ble', 'v.', etc. should not cause spurious sentence breaks."""
        text = "The Hon'ble Mr. Justice delivered the judgment in XYZ v. State of UP."
        sentences = segment_sentences(text)
        # Should ideally be 1 sentence (or at most 2), not 4+
        assert len(sentences) <= 2


# ────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        chunks = create_chunks(sentences, max_chunk_tokens=50, overlap_sentences=1)
        assert len(chunks) > 0
        assert all(isinstance(c, ExtractedChunk) for c in chunks)

    def test_chunk_ids_are_sequential(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        chunks = create_chunks(sentences, max_chunk_tokens=50)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_respects_token_limit(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        limit = 50
        chunks = create_chunks(sentences, max_chunk_tokens=limit, overlap_sentences=1)
        for c in chunks:
            # Chunks may exceed the budget due to overlap sentences and
            # single long sentences that can't be split further.
            assert c.token_count <= limit * 3, (
                f"Chunk {c.chunk_id} has {c.token_count} tokens "
                f"(limit={limit}, 3x tolerance={limit*3})"
            )

    def test_single_chunk_for_short_text(self):
        sentences = segment_sentences("The court allowed the appeal.")
        chunks = create_chunks(sentences, max_chunk_tokens=1000)
        assert len(chunks) == 1

    def test_chunk_offsets(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        chunks = create_chunks(sentences, max_chunk_tokens=100)
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char > c.start_char


# ────────────────────────────────────────────────────────────
# Salience Ranking
# ────────────────────────────────────────────────────────────

class TestSalienceRanking:
    def test_ranking_returns_scored_chunks(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        chunks = create_chunks(sentences, max_chunk_tokens=50)
        ranked = rank_chunks_by_salience(chunks, top_k=3)
        assert len(ranked) <= 3
        assert all(c.salience_score > 0 for c in ranked)

    def test_ranking_with_top_k(self):
        sentences = segment_sentences(SAMPLE_TEXT)
        chunks = create_chunks(sentences, max_chunk_tokens=30)
        ranked = rank_chunks_by_salience(chunks, top_k=2)
        assert len(ranked) == min(2, len(chunks))

    def test_empty_input(self):
        ranked = rank_chunks_by_salience([])
        assert ranked == []


# ────────────────────────────────────────────────────────────
# Full Retrieve Pipeline
# ────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_full_retrieval(self):
        config = RetrieverConfig(
            max_chunk_tokens=60,
            top_k_chunks=5,
        )
        chunks = retrieve(SAMPLE_TEXT, config)
        assert len(chunks) > 0
        assert len(chunks) <= 5
        # Chunks should be re-sorted by document order
        offsets = [c.start_char for c in chunks]
        assert offsets == sorted(offsets)

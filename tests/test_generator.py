"""
Tests for the generator module (prompt construction, response parsing).

These tests focus on the offline/parseable parts. API calls to Gemini
are tested only if GOOGLE_API_KEY is set (integration tests).
"""

import json
import pytest
from generator import (
    GroundedSentence,
    GenerationResult,
    _build_prompt,
    _extract_json,
    _parse_generation_response,
)
from retriever import ExtractedChunk


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

def _make_chunks(n: int = 3) -> list[ExtractedChunk]:
    return [
        ExtractedChunk(
            chunk_id=i,
            text=f"This is the text of chunk {i}. It contains legal content.",
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        for i in range(n)
    ]


# ────────────────────────────────────────────────────────────
# Prompt Construction
# ────────────────────────────────────────────────────────────

class TestPromptConstruction:
    def test_prompt_contains_all_chunk_ids(self):
        chunks = _make_chunks(3)
        prompt = _build_prompt(chunks, target_sentences=5)
        for c in chunks:
            assert f"[CHUNK-{c.chunk_id}]" in prompt

    def test_prompt_contains_chunk_text(self):
        chunks = _make_chunks(2)
        prompt = _build_prompt(chunks)
        for c in chunks:
            assert c.text in prompt

    def test_prompt_mentions_target_sentences(self):
        prompt = _build_prompt(_make_chunks(1), target_sentences=10)
        assert "10" in prompt


# ────────────────────────────────────────────────────────────
# JSON Extraction
# ────────────────────────────────────────────────────────────

class TestJsonExtraction:
    def test_plain_json_array(self):
        raw = '[{"sentence": "Test.", "source_chunks": [0], "key_spans": ["test"]}]'
        result = _extract_json(raw)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"sentence": "Test.", "source_chunks": [0], "key_spans": []}]\n```'
        result = _extract_json(raw)
        assert isinstance(result, list)

    def test_json_with_surrounding_text(self):
        raw = 'Here is the summary:\n[{"sentence": "Test."}]\nDone.'
        result = _extract_json(raw)
        assert isinstance(result, list)

    def test_single_object(self):
        raw = '{"sentence": "Test.", "source_chunks": [1]}'
        result = _extract_json(raw)
        assert isinstance(result, dict)

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, Exception)):
            _extract_json("This is not JSON at all.")


# ────────────────────────────────────────────────────────────
# Response Parsing
# ────────────────────────────────────────────────────────────

class TestResponseParsing:
    def test_parse_valid_response(self):
        raw = json.dumps([
            {
                "sentence": "The court allowed the appeal.",
                "source_chunks": [0, 2],
                "key_spans": ["appeal is allowed"],
            },
            {
                "sentence": "The SIA study was non-compliant.",
                "source_chunks": [1],
                "key_spans": ["SIA study", "non-compliant"],
            },
        ])
        sentences = _parse_generation_response(raw)
        assert len(sentences) == 2
        assert sentences[0].source_chunk_ids == [0, 2]
        assert sentences[1].key_spans == ["SIA study", "non-compliant"]

    def test_parse_single_object_response(self):
        raw = json.dumps({
            "sentence": "Single sentence.",
            "source_chunks": [0],
            "key_spans": ["sentence"],
        })
        sentences = _parse_generation_response(raw)
        assert len(sentences) == 1

    def test_grounded_sentence_to_dict(self):
        gs = GroundedSentence(
            sentence="Test.", source_chunk_ids=[1], key_spans=["test"]
        )
        d = gs.to_dict()
        assert d["sentence"] == "Test."
        assert d["source_chunks"] == [1]


# ────────────────────────────────────────────────────────────
# GenerationResult
# ────────────────────────────────────────────────────────────

class TestGenerationResult:
    def test_summary_text(self):
        gs1 = GroundedSentence(sentence="First.", source_chunk_ids=[0])
        gs2 = GroundedSentence(sentence="Second.", source_chunk_ids=[1])
        result = GenerationResult(grounded_sentences=[gs1, gs2])
        assert result.summary_text == "First. Second."

    def test_to_dict(self):
        gs = GroundedSentence(sentence="Test.", source_chunk_ids=[0])
        result = GenerationResult(grounded_sentences=[gs], model_used="gemini-test")
        d = result.to_dict()
        assert "summary" in d
        assert d["model"] == "gemini-test"

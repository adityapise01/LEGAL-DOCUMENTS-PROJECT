"""
Tests for the verifier module (NLI checking, span matching, verdicts).

NLI model tests are marked with `@pytest.mark.slow` so they can be
skipped in quick CI runs (``pytest -m "not slow"``).
"""

import pytest
from verifier import (
    Verdict,
    VerificationResult,
    VerificationReport,
    _fuzzy_ratio,
    _best_span_match,
)
from generator import GroundedSentence


# ────────────────────────────────────────────────────────────
# Fuzzy Matching
# ────────────────────────────────────────────────────────────

class TestFuzzyMatching:
    def test_exact_match(self):
        assert _fuzzy_ratio("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert _fuzzy_ratio("Hello World", "hello world") == 1.0

    def test_partial_match(self):
        ratio = _fuzzy_ratio("the appeal is allowed", "the appeal was allowed")
        assert 0.8 < ratio < 1.0

    def test_no_match(self):
        ratio = _fuzzy_ratio("completely different text", "xyz abc 123")
        assert ratio < 0.5


class TestBestSpanMatch:
    def test_exact_substring(self):
        source = "The court held that the appeal is allowed in part."
        span = "appeal is allowed"
        score = _best_span_match(span, source)
        assert score == 1.0

    def test_fuzzy_substring(self):
        source = "The court held that the appeal was allowed in part."
        span = "appeal is allowed"
        score = _best_span_match(span, source)
        assert score > 0.7

    def test_no_match(self):
        source = "The court held that the petition is dismissed."
        span = "completely unrelated text about something else"
        score = _best_span_match(span, source)
        assert score < 0.5


# ────────────────────────────────────────────────────────────
# Verdict Data Models
# ────────────────────────────────────────────────────────────

class TestVerdict:
    def test_verdict_values(self):
        assert Verdict.PASS.value == "PASS"
        assert Verdict.WARN.value == "WARN"
        assert Verdict.FAIL.value == "FAIL"
        assert Verdict.DROPPED.value == "DROPPED"


class TestVerificationResult:
    def test_to_dict(self):
        gs = GroundedSentence(
            sentence="Test sentence.",
            source_chunk_ids=[0, 1],
            key_spans=["test"],
        )
        vr = VerificationResult(
            sentence=gs,
            verdict=Verdict.PASS,
            entailment_score=0.95,
            nli_label="ENTAILMENT",
        )
        d = vr.to_dict()
        assert d["verdict"] == "PASS"
        assert d["entailment_score"] == 0.95


class TestVerificationReport:
    def _make_report(self, verdicts: list[Verdict]) -> VerificationReport:
        results = []
        for v in verdicts:
            gs = GroundedSentence(sentence=f"Sentence {v.value}.")
            results.append(VerificationResult(sentence=gs, verdict=v))
        return VerificationReport(results=results)

    def test_pass_rate_all_pass(self):
        report = self._make_report([Verdict.PASS, Verdict.PASS, Verdict.PASS])
        assert report.pass_rate == 1.0

    def test_pass_rate_mixed(self):
        report = self._make_report([Verdict.PASS, Verdict.WARN, Verdict.FAIL])
        # PASS + WARN = 2 out of 3
        assert abs(report.pass_rate - 2 / 3) < 0.01

    def test_pass_rate_all_fail(self):
        report = self._make_report([Verdict.FAIL, Verdict.DROPPED])
        assert report.pass_rate == 0.0

    def test_verified_sentences_excludes_failed(self):
        report = self._make_report([Verdict.PASS, Verdict.FAIL, Verdict.WARN])
        # Should include PASS and WARN, not FAIL
        assert len(report.verified_sentences) == 2

    def test_verified_summary(self):
        report = self._make_report([Verdict.PASS, Verdict.PASS])
        summary = report.verified_summary
        assert "Sentence PASS" in summary

    def test_to_dict(self):
        report = self._make_report([Verdict.PASS])
        d = report.to_dict()
        assert d["total_sentences"] == 1
        assert d["passed"] == 1
        assert "verified_summary" in d

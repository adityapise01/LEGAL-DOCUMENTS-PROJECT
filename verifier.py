"""
Phase 3 — Post-Hoc Verification & Regeneration.

Checks every generated summary sentence against its cited source spans
using two complementary strategies:

  1. **NLI Entailment** — DeBERTa-v3-large-mnli classifies whether the
     source chunks logically entail the summary sentence.
  2. **Span Anchor Verification** — Fuzzy string matching confirms that
     the quoted key_spans actually exist in the cited source text.

Sentences that fail verification enter a **reject-and-regenerate loop**,
where the generator is re-invoked with explicit feedback.  After
*max_regen_attempts*, still-failing sentences are dropped and flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from config import VerifierConfig, GeneratorConfig
from generator import GroundedSentence, regenerate_sentence
from retriever import ExtractedChunk

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    DROPPED = "DROPPED"


@dataclass
class VerificationResult:
    """Verification outcome for a single grounded sentence."""

    sentence: GroundedSentence
    verdict: Verdict
    entailment_score: float = 0.0
    nli_label: str = ""  # ENTAILMENT / NEUTRAL / CONTRADICTION
    span_match_scores: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    regen_attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence.sentence,
            "verdict": self.verdict.value,
            "entailment_score": round(self.entailment_score, 4),
            "nli_label": self.nli_label,
            "span_match_scores": [round(s, 4) for s in self.span_match_scores],
            "reasons": self.reasons,
            "source_chunks": self.sentence.source_chunk_ids,
            "key_spans": self.sentence.key_spans,
            "regen_attempts": self.regen_attempts,
        }


@dataclass
class VerificationReport:
    """Aggregate verification report for the entire summary."""

    results: list[VerificationResult]

    @property
    def passed(self) -> list[VerificationResult]:
        return [r for r in self.results if r.verdict == Verdict.PASS]

    @property
    def warned(self) -> list[VerificationResult]:
        return [r for r in self.results if r.verdict == Verdict.WARN]

    @property
    def failed(self) -> list[VerificationResult]:
        return [r for r in self.results if r.verdict in (Verdict.FAIL, Verdict.DROPPED)]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        ok = len([r for r in self.results if r.verdict in (Verdict.PASS, Verdict.WARN)])
        return ok / len(self.results)

    @property
    def verified_sentences(self) -> list[GroundedSentence]:
        """Return only sentences that passed or warned (not failed/dropped)."""
        return [
            r.sentence for r in self.results
            if r.verdict in (Verdict.PASS, Verdict.WARN)
        ]

    @property
    def verified_summary(self) -> str:
        return " ".join(gs.sentence for gs in self.verified_sentences)

    def to_dict(self) -> dict:
        return {
            "total_sentences": len(self.results),
            "passed": len(self.passed),
            "warned": len(self.warned),
            "failed": len(self.failed),
            "pass_rate": round(self.pass_rate, 4),
            "verified_summary": self.verified_summary,
            "details": [r.to_dict() for r in self.results],
        }


# ────────────────────────────────────────────────────────────
# NLI Entailment Checker
# ────────────────────────────────────────────────────────────

class NLIChecker:
    """
    Wraps a HuggingFace NLI model (default: DeBERTa-v3-large-mnli) to
    check whether a premise entails a hypothesis.
    """

    _LABEL_MAP = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading NLI model '%s' on %s…", model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._model.eval()
        self._torch = torch

    def check(
        self, premise: str, hypothesis: str, max_premise_tokens: int = 512
    ) -> tuple[str, float]:
        """
        Run NLI inference.

        Args:
            premise: The source text (concatenated chunks).
            hypothesis: The summary sentence to verify.
            max_premise_tokens: Truncate the premise to this many tokens.

        Returns:
            (label, entailment_probability) where label is one of
            ENTAILMENT, NEUTRAL, CONTRADICTION.
        """
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=max_premise_tokens + 128,  # hypothesis headroom
        ).to(self._device)

        with self._torch.no_grad():
            logits = self._model(**inputs).logits

        probs = self._torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = self._LABEL_MAP.get(pred_idx, "UNKNOWN")
        entailment_prob = float(probs[2])  # index 2 = ENTAILMENT

        return label, entailment_prob


# ────────────────────────────────────────────────────────────
# Span Anchor Verification (fuzzy matching)
# ────────────────────────────────────────────────────────────

def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Compute a normalised Levenshtein similarity ratio between two strings.
    Uses python-Levenshtein if available, else falls back to difflib.
    """
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()

    if a_lower == b_lower:
        return 1.0

    try:
        from Levenshtein import ratio
        return ratio(a_lower, b_lower)
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a_lower, b_lower).ratio()


def _best_span_match(span: str, source_text: str) -> float:
    """
    Find the best fuzzy match for *span* within *source_text*.

    Instead of comparing against the entire text (expensive), we slide
    a window of similar length and compute the ratio.
    """
    span_lower = span.lower().strip()
    source_lower = source_text.lower()

    # Exact substring check first
    if span_lower in source_lower:
        return 1.0

    # Sliding window fuzzy match
    span_len = len(span_lower)
    best = 0.0
    step = max(1, span_len // 4)

    for start in range(0, len(source_lower) - span_len + 1, step):
        window = source_lower[start : start + span_len]
        r = _fuzzy_ratio(span_lower, window)
        if r > best:
            best = r
            if best >= 0.98:
                break  # good enough

    return best


# ────────────────────────────────────────────────────────────
# Core Verification Logic
# ────────────────────────────────────────────────────────────

def _verify_single(
    gs: GroundedSentence,
    chunk_map: dict[int, ExtractedChunk],
    nli_checker: NLIChecker,
    config: VerifierConfig,
) -> VerificationResult:
    """Verify a single grounded sentence."""
    reasons: list[str] = []
    span_scores: list[float] = []

    # ── 1. Build premise from cited chunks ──
    cited_chunks = [chunk_map[cid] for cid in gs.source_chunk_ids if cid in chunk_map]
    if not cited_chunks:
        return VerificationResult(
            sentence=gs,
            verdict=Verdict.FAIL,
            reasons=["No valid source chunks cited."],
        )

    premise = " ".join(c.text for c in cited_chunks)

    # ── 2. NLI entailment check ──
    nli_label, entailment_prob = nli_checker.check(
        premise, gs.sentence, config.max_premise_tokens
    )

    if nli_label == "CONTRADICTION":
        reasons.append(f"NLI: CONTRADICTION (entailment_prob={entailment_prob:.3f})")
    elif entailment_prob < config.entailment_threshold:
        reasons.append(
            f"NLI: entailment below threshold "
            f"({entailment_prob:.3f} < {config.entailment_threshold})"
        )

    # ── 3. Span anchor verification ──
    source_text = " ".join(c.text for c in cited_chunks)
    for span in gs.key_spans:
        score = _best_span_match(span, source_text)
        span_scores.append(score)
        if score < config.fuzzy_match_threshold:
            reasons.append(
                f"Phantom citation: span '{span[:60]}…' not found in source "
                f"(best match ratio={score:.3f})"
            )

    # ── 4. Determine verdict ──
    if reasons:
        # Check if it's just a fuzzy warning
        all_spans_ok = all(s >= config.fuzzy_match_threshold for s in span_scores)
        nli_ok = entailment_prob >= config.entailment_threshold and nli_label != "CONTRADICTION"

        if nli_ok and all_spans_ok:
            # Might be a soft warning for fuzzy matches
            has_fuzzy_warn = any(
                config.fuzzy_match_threshold <= s < config.fuzzy_warn_threshold
                for s in span_scores
            )
            verdict = Verdict.WARN if has_fuzzy_warn else Verdict.PASS
            reasons = [r for r in reasons if "Phantom" not in r]  # clear phantom reasons
        else:
            verdict = Verdict.FAIL
    else:
        verdict = Verdict.PASS

    return VerificationResult(
        sentence=gs,
        verdict=verdict,
        entailment_score=entailment_prob,
        nli_label=nli_label,
        span_match_scores=span_scores,
        reasons=reasons,
    )


# ────────────────────────────────────────────────────────────
# Public API — Full Verification with Regeneration Loop
# ────────────────────────────────────────────────────────────

def verify(
    grounded_sentences: list[GroundedSentence],
    chunks: list[ExtractedChunk],
    verifier_config: Optional[VerifierConfig] = None,
    generator_config: Optional[GeneratorConfig] = None,
) -> VerificationReport:
    """
    Verify all grounded sentences and regenerate those that fail.

    Args:
        grounded_sentences: Output from the generator.
        chunks: The extracted chunks used for generation.
        verifier_config: Verifier configuration.
        generator_config: Generator config (needed for regeneration calls).

    Returns:
        VerificationReport with per-sentence verdicts and a verified summary.
    """
    if verifier_config is None:
        verifier_config = VerifierConfig()
    if generator_config is None:
        generator_config = GeneratorConfig()

    # Build chunk lookup
    chunk_map = {c.chunk_id: c for c in chunks}

    # Initialise NLI model
    nli_checker = NLIChecker(verifier_config.nli_model)

    results: list[VerificationResult] = []

    for gs in grounded_sentences:
        logger.info("Verifying: '%s'", gs.sentence[:60])

        result = _verify_single(gs, chunk_map, nli_checker, verifier_config)

        # ── Regeneration loop for FAIL verdicts ──
        if result.verdict == Verdict.FAIL:
            relevant_chunks = [
                chunk_map[cid] for cid in gs.source_chunk_ids if cid in chunk_map
            ]
            if not relevant_chunks:
                relevant_chunks = chunks[:5]  # fallback to top chunks

            for attempt in range(verifier_config.max_regen_attempts):
                logger.info(
                    "  Regeneration attempt %d/%d for: '%s'",
                    attempt + 1, verifier_config.max_regen_attempts,
                    gs.sentence[:50],
                )

                reason_text = "; ".join(result.reasons)
                new_gs = regenerate_sentence(
                    failed_sentence=gs.sentence,
                    reason=reason_text,
                    relevant_chunks=relevant_chunks,
                    config=generator_config,
                )

                # Re-verify the regenerated sentence
                new_result = _verify_single(
                    new_gs, chunk_map, nli_checker, verifier_config
                )
                new_result.regen_attempts = attempt + 1

                if new_result.verdict != Verdict.FAIL:
                    result = new_result
                    break
                else:
                    result = new_result
                    gs = new_gs  # update for next regen attempt

            # If still failing after all attempts, mark as DROPPED
            if result.verdict == Verdict.FAIL:
                result.verdict = Verdict.DROPPED
                result.reasons.append(
                    f"Dropped after {verifier_config.max_regen_attempts} "
                    f"regeneration attempts."
                )
                logger.warning(
                    "  DROPPED sentence after %d regen attempts: '%s'",
                    verifier_config.max_regen_attempts, gs.sentence[:60],
                )

        results.append(result)

    report = VerificationReport(results=results)
    logger.info(
        "Verification complete: %d passed, %d warned, %d failed/dropped (%.1f%% pass rate).",
        len(report.passed), len(report.warned), len(report.failed),
        report.pass_rate * 100,
    )
    return report

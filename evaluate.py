"""
Evaluation & Metrics Framework.

Computes four families of metrics:

  1. **ROUGE** (rouge-score)       — n-gram overlap adequacy
  2. **BERTScore** (bert-score)    — semantic overlap adequacy
  3. **NLI Faithfulness**          — fraction of sentences entailed by source
  4. **Human Evaluation / IAA**    — Cohen's κ and Fleiss' κ from annotator CSV

All public functions accept plain strings or lists and return
JSON-serialisable dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import EvaluationConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 1. ROUGE Scores
# ────────────────────────────────────────────────────────────

def compute_rouge(
    hypothesis: str,
    reference: str,
    rouge_types: Optional[list[str]] = None,
) -> dict:
    """
    Compute ROUGE scores between a hypothesis and reference summary.

    Args:
        hypothesis: Generated summary text.
        reference: Gold-standard reference summary.
        rouge_types: List of ROUGE variants (default: rouge1, rouge2, rougeL).

    Returns:
        Dict mapping each ROUGE type to its precision, recall, and F1.
    """
    from rouge_score import rouge_scorer

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    result = {}
    for rtype in rouge_types:
        s = scores[rtype]
        result[rtype] = {
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "f1": round(s.fmeasure, 4),
        }

    logger.info("ROUGE scores: %s", {k: v["f1"] for k, v in result.items()})
    return result


# ────────────────────────────────────────────────────────────
# 2. BERTScore
# ────────────────────────────────────────────────────────────

def compute_bertscore(
    hypothesis: str | list[str],
    reference: str | list[str],
    model_type: str = "roberta-large",
    rescale: bool = True,
) -> dict:
    """
    Compute BERTScore between hypothesis and reference texts.

    Args:
        hypothesis: Generated summary (string or list of strings).
        reference: Reference summary (string or list of strings).
        model_type: BERTScore backbone model.
        rescale: Whether to rescale with baselines.

    Returns:
        Dict with precision, recall, F1 (averaged if multiple inputs).
    """
    from bert_score import score as bert_score_fn

    # Ensure list input
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if isinstance(reference, str):
        reference = [reference]

    P, R, F1 = bert_score_fn(
        hypothesis,
        reference,
        model_type=model_type,
        rescale_with_baseline=rescale,
        lang="en",
        verbose=False,
    )

    result = {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4),
    }

    logger.info("BERTScore: P=%.4f, R=%.4f, F1=%.4f", result["precision"], result["recall"], result["f1"])
    return result


# ────────────────────────────────────────────────────────────
# 3. NLI Faithfulness Score
# ────────────────────────────────────────────────────────────

def compute_faithfulness(
    summary_sentences: list[str],
    source_texts: list[str],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    max_premise_tokens: int = 512,
) -> dict:
    """
    Compute an NLI-based faithfulness score: for each summary sentence,
    check whether the corresponding source text entails it.

    Args:
        summary_sentences: List of summary sentences.
        source_texts: List of source texts (one per sentence, typically the
                      concatenated cited chunks for that sentence).
        nli_model: HuggingFace NLI model checkpoint.
        max_premise_tokens: Truncation limit for the premise.

    Returns:
        Dict with mean_entailment_prob, pct_entailed, pct_contradicted,
        and per-sentence details.
    """
    if len(summary_sentences) != len(source_texts):
        raise ValueError(
            f"Mismatched lengths: {len(summary_sentences)} sentences "
            f"vs {len(source_texts)} source texts."
        )

    # Lazy import to avoid loading NLI model at module level
    from verifier import NLIChecker

    checker = NLIChecker(nli_model)

    entailment_probs: list[float] = []
    labels: list[str] = []
    per_sentence: list[dict] = []

    for sent, source in zip(summary_sentences, source_texts):
        label, prob = checker.check(source, sent, max_premise_tokens)
        entailment_probs.append(prob)
        labels.append(label)
        per_sentence.append({
            "sentence": sent[:100],
            "label": label,
            "entailment_prob": round(prob, 4),
        })

    n = len(labels)
    result = {
        "mean_entailment_prob": round(float(np.mean(entailment_probs)), 4),
        "pct_entailed": round(labels.count("ENTAILMENT") / n * 100, 2) if n else 0,
        "pct_neutral": round(labels.count("NEUTRAL") / n * 100, 2) if n else 0,
        "pct_contradicted": round(labels.count("CONTRADICTION") / n * 100, 2) if n else 0,
        "num_sentences": n,
        "per_sentence": per_sentence,
    }

    logger.info(
        "Faithfulness: %.1f%% entailed, %.1f%% contradicted, mean_prob=%.4f",
        result["pct_entailed"], result["pct_contradicted"], result["mean_entailment_prob"],
    )
    return result


# ────────────────────────────────────────────────────────────
# 4. Human Evaluation — Inter-Annotator Agreement
# ────────────────────────────────────────────────────────────

def _interpret_kappa(kappa: float, labels: dict[str, tuple[float, float]]) -> str:
    """Return a human-readable interpretation of a kappa value."""
    for label, (lo, hi) in labels.items():
        if lo <= kappa < hi:
            return label
    return "near-perfect" if kappa >= 0.80 else "undefined"


def _cohens_kappa(ann1: list, ann2: list) -> float:
    """
    Compute Cohen's Kappa for two annotators.

    Works for ordinal/categorical ratings by treating them as
    categorical labels.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(ann1, ann2))


def _fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for multiple annotators.

    Args:
        rating_matrix: (n_subjects × n_categories) matrix where each entry
                       is the number of raters who assigned that category
                       to that subject.

    Returns:
        Fleiss' kappa statistic.
    """
    n_subjects, n_categories = rating_matrix.shape
    n_raters = int(rating_matrix[0].sum())

    if n_raters < 2:
        return 0.0

    # Proportion in each category
    p_j = rating_matrix.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (
        (rating_matrix ** 2).sum(axis=1) - n_raters
    ) / (n_raters * (n_raters - 1))

    P_bar = float(P_i.mean())
    P_e = float((p_j ** 2).sum())

    if abs(1 - P_e) < 1e-10:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(kappa)


def compute_inter_annotator_agreement(
    annotations_path: str | Path,
    score_column: str = "faithfulness_score",
    config: Optional[EvaluationConfig] = None,
) -> dict:
    """
    Compute inter-annotator agreement from a CSV of human annotations.

    Expected CSV columns:
        summary_id, annotator_id, faithfulness_score, adequacy_score, fluency_score

    The function computes Cohen's Kappa (for exactly 2 annotators) or
    Fleiss' Kappa (for 2+ annotators) on the specified score column.

    Args:
        annotations_path: Path to the CSV file.
        score_column: Which score column to compute agreement on.
        config: Evaluation configuration.

    Returns:
        Dict with kappa value, interpretation, annotator count, and
        per-metric breakdown if multiple score columns are present.
    """
    if config is None:
        config = EvaluationConfig()

    df = pd.read_csv(annotations_path)

    required = {"summary_id", "annotator_id", score_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in annotations CSV: {missing}")

    annotators = sorted(df["annotator_id"].unique())
    n_annotators = len(annotators)

    # Compute all available score columns
    score_cols = [c for c in ["faithfulness_score", "adequacy_score", "fluency_score"]
                  if c in df.columns]

    per_metric: dict[str, dict] = {}

    for col in score_cols:
        if n_annotators == 2:
            # Cohen's Kappa
            pivot = df.pivot(index="summary_id", columns="annotator_id", values=col)
            pivot = pivot.dropna()
            ann1 = pivot.iloc[:, 0].astype(int).tolist()
            ann2 = pivot.iloc[:, 1].astype(int).tolist()
            kappa = _cohens_kappa(ann1, ann2)
            method = "Cohen's Kappa"
        else:
            # Fleiss' Kappa
            categories = sorted(df[col].dropna().unique())
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            summaries = sorted(df["summary_id"].unique())

            matrix = np.zeros((len(summaries), len(categories)), dtype=float)
            for s_idx, sid in enumerate(summaries):
                subset = df[df["summary_id"] == sid]
                for _, row in subset.iterrows():
                    val = row[col]
                    if pd.notna(val) and val in cat_to_idx:
                        matrix[s_idx, cat_to_idx[val]] += 1

            kappa = _fleiss_kappa(matrix)
            method = "Fleiss' Kappa"

        interpretation = _interpret_kappa(kappa, config.kappa_labels)
        per_metric[col] = {
            "kappa": round(kappa, 4),
            "method": method,
            "interpretation": interpretation,
        }

    primary = per_metric.get(score_column, {})

    result = {
        "primary_metric": score_column,
        "kappa": primary.get("kappa", 0.0),
        "method": primary.get("method", ""),
        "interpretation": primary.get("interpretation", ""),
        "n_annotators": n_annotators,
        "n_summaries": int(df["summary_id"].nunique()),
        "per_metric": per_metric,
    }

    logger.info(
        "IAA (%s): κ=%.4f (%s) — %d annotators, %d summaries.",
        result["method"], result["kappa"], result["interpretation"],
        n_annotators, result["n_summaries"],
    )
    return result


# ────────────────────────────────────────────────────────────
# Convenience: Run All Metrics
# ────────────────────────────────────────────────────────────

def evaluate_summary(
    hypothesis: str,
    reference: str,
    summary_sentences: Optional[list[str]] = None,
    source_texts: Optional[list[str]] = None,
    config: Optional[EvaluationConfig] = None,
) -> dict:
    """
    Run all automated metrics on a single summary.

    Args:
        hypothesis: The generated summary text.
        reference: The gold reference summary.
        summary_sentences: Individual sentences (for faithfulness). If None,
                           faithfulness is skipped.
        source_texts: Source texts corresponding to each sentence.
        config: Evaluation configuration.

    Returns:
        Dict with ROUGE, BERTScore, and optionally faithfulness results.
    """
    if config is None:
        config = EvaluationConfig()

    results: dict = {"rouge": {}, "bertscore": {}, "faithfulness": None}

    # ROUGE
    results["rouge"] = compute_rouge(hypothesis, reference, config.rouge_types)

    # BERTScore
    results["bertscore"] = compute_bertscore(
        hypothesis, reference,
        model_type=config.bertscore_model,
        rescale=config.bertscore_rescale,
    )

    # Faithfulness (optional)
    if summary_sentences and source_texts:
        results["faithfulness"] = compute_faithfulness(
            summary_sentences, source_texts
        )

    return results

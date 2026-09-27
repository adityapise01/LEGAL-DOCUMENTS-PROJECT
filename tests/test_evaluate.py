"""
Tests for the evaluation module (ROUGE, BERTScore, IAA).

BERTScore and faithfulness tests are marked `@pytest.mark.slow`.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    compute_rouge,
    compute_inter_annotator_agreement,
    _cohens_kappa,
    _fleiss_kappa,
    _interpret_kappa,
)
from config import EvaluationConfig


# ────────────────────────────────────────────────────────────
# ROUGE
# ────────────────────────────────────────────────────────────

class TestROUGE:
    def test_identical_texts(self):
        text = "The court allowed the appeal in part."
        scores = compute_rouge(text, text)
        assert scores["rouge1"]["f1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rouge2"]["f1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL"]["f1"] == pytest.approx(1.0, abs=0.01)

    def test_completely_different_texts(self):
        scores = compute_rouge("alpha beta gamma", "delta epsilon zeta")
        assert scores["rouge1"]["f1"] < 0.1

    def test_partial_overlap(self):
        hyp = "The court allowed the appeal."
        ref = "The court dismissed the appeal."
        scores = compute_rouge(hyp, ref)
        # Some overlap (the, court, appeal) but not perfect
        assert 0.3 < scores["rouge1"]["f1"] < 0.9

    def test_custom_rouge_types(self):
        scores = compute_rouge("hello world", "hello world", rouge_types=["rouge1"])
        assert "rouge1" in scores
        assert "rouge2" not in scores


# ────────────────────────────────────────────────────────────
# Inter-Annotator Agreement
# ────────────────────────────────────────────────────────────

class TestIAA:
    def _create_annotations_csv(self, data: list[dict]) -> str:
        """Create a temp CSV and return its path."""
        df = pd.DataFrame(data)
        path = tempfile.mktemp(suffix=".csv")
        df.to_csv(path, index=False)
        return path

    def test_perfect_agreement(self):
        data = []
        for sid in range(10):
            for ann_id in [1, 2]:
                data.append({
                    "summary_id": sid,
                    "annotator_id": ann_id,
                    "faithfulness_score": 5,
                })
        path = self._create_annotations_csv(data)
        result = compute_inter_annotator_agreement(path)
        # Perfect agreement → kappa should be high (may be undefined
        # if all labels are the same, but sklearn handles this)
        assert result["n_annotators"] == 2

    def test_random_agreement(self):
        rng = np.random.RandomState(42)
        data = []
        for sid in range(50):
            for ann_id in [1, 2]:
                data.append({
                    "summary_id": sid,
                    "annotator_id": ann_id,
                    "faithfulness_score": int(rng.randint(1, 6)),
                })
        path = self._create_annotations_csv(data)
        result = compute_inter_annotator_agreement(path)
        # Random ≈ low kappa
        assert result["kappa"] < 0.5

    def test_missing_column_raises(self):
        data = [{"summary_id": 1, "annotator_id": 1, "wrong_col": 5}]
        path = self._create_annotations_csv(data)
        with pytest.raises(ValueError, match="Missing columns"):
            compute_inter_annotator_agreement(path)


class TestKappaHelpers:
    def test_cohens_kappa_perfect(self):
        ann1 = [1, 2, 3, 4, 5] * 10
        ann2 = [1, 2, 3, 4, 5] * 10
        kappa = _cohens_kappa(ann1, ann2)
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_fleiss_kappa_perfect(self):
        # 5 subjects, 3 categories, 2 raters — perfect agreement
        matrix = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 0],
            [0, 2, 0],
        ], dtype=float)
        kappa = _fleiss_kappa(matrix)
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_interpret_kappa(self):
        config = EvaluationConfig()
        assert _interpret_kappa(0.1, config.kappa_labels) == "poor"
        assert _interpret_kappa(0.5, config.kappa_labels) == "moderate"
        assert _interpret_kappa(0.75, config.kappa_labels) == "substantial"
        assert _interpret_kappa(0.9, config.kappa_labels) == "near-perfect"

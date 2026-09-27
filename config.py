"""
Centralized configuration for the Legal Judgment Summarization Pipeline.

All tuneable parameters — model identifiers, chunking hyperparameters,
verification thresholds, and file paths — live here so every downstream
module imports a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrieverConfig:
    """Controls extractive selection and hierarchical chunking."""

    # spaCy model for sentence segmentation
    spacy_model: str = "en_core_web_sm"

    # Sentence-transformer model for semantic similarity in TextRank
    embedding_model: str = "all-MiniLM-L6-v2"

    # Maximum number of tokens per chunk (approx; measured in whitespace tokens)
    max_chunk_tokens: int = 1024

    # Number of overlapping sentences between adjacent chunks
    chunk_overlap_sentences: int = 2

    # How many top-ranked chunks to keep after salience ranking
    top_k_chunks: int = 15

    # TextRank damping factor (PageRank parameter)
    textrank_damping: float = 0.85

    # Minimum cosine similarity to create an edge in the TextRank graph
    similarity_threshold: float = 0.1


@dataclass
class GeneratorConfig:
    """Controls the constrained rewriting / span-grounding phase."""

    # Groq model identifier
    model_name: str = "qwen/qwen3.8-27b"

    # Groq API key — read from environment
    api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))

    # Temperature for generation (lower = more faithful)
    temperature: float = 0.2

    # Maximum output tokens for the summary
    max_output_tokens: int = 800

    # Number of retries if JSON parsing fails
    max_parse_retries: int = 2

    # Target summary length guidance (in sentences)
    target_summary_sentences: int = 15


@dataclass
class VerifierConfig:
    """Controls post-hoc NLI verification and the regeneration loop."""

    # HuggingFace NLI model checkpoint (freely available, no auth needed)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Entailment probability threshold — sentences below this are flagged
    entailment_threshold: float = 0.7

    # Maximum number of regeneration attempts for failed sentences
    max_regen_attempts: int = 3

    # Minimum Levenshtein ratio for fuzzy span matching
    fuzzy_match_threshold: float = 0.80

    # Ratio above which a fuzzy match emits a WARN instead of FAIL
    fuzzy_warn_threshold: float = 0.95

    # Maximum premise length (in tokens) sent to the NLI model
    max_premise_tokens: int = 512


@dataclass
class EvaluationConfig:
    """Controls the evaluation and metrics framework."""

    # BERTScore backbone model
    bertscore_model: str = "roberta-large"

    # Whether to rescale BERTScore with baselines
    bertscore_rescale: bool = True

    # ROUGE variants to compute
    rouge_types: list[str] = field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL"]
    )

    # Kappa interpretation thresholds
    kappa_labels: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "poor": (0.0, 0.20),
            "fair": (0.20, 0.40),
            "moderate": (0.40, 0.60),
            "substantial": (0.60, 0.80),
            "near-perfect": (0.80, 1.00),
        }
    )


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Project paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [self.raw_data_dir, self.processed_data_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

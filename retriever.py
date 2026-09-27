"""
Phase 1 — Extractive Selection & Hierarchical Chunking.

Implements a three-stage pipeline:
  1. Sentence segmentation  (spaCy with legal-abbreviation rules)
  2. Structural chunking     (paragraph-aware, with configurable overlap)
  3. Salience ranking        (TextRank over sentence-transformer embeddings)

The output is a ranked list of `ExtractedChunk` objects, each carrying its
source-text character offsets so that downstream phases can cite exact spans.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from config import RetrieverConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    """A single sentence with its position in the source document."""

    text: str
    start_char: int
    end_char: int
    index: int  # ordinal position in the document


@dataclass
class ExtractedChunk:
    """A chunk of text selected from the source, carrying provenance metadata."""

    chunk_id: int
    text: str
    start_char: int
    end_char: int
    sentences: list[Sentence] = field(default_factory=list)
    salience_score: float = 0.0

    @property
    def token_count(self) -> int:
        return len(self.text.split())


# ────────────────────────────────────────────────────────────
# Stage 1 — Sentence Segmentation
# ────────────────────────────────────────────────────────────

# Legal abbreviations that should NOT trigger a sentence break
_LEGAL_ABBREVS = {
    "v.", "vs.", "art.", "sect.", "sec.", "no.", "nos.",
    "hon'ble", "shri", "smt.", "dr.", "mr.", "mrs.", "ms.",
    "govt.", "pvt.", "ltd.", "ors.", "anr.", "etc.", "viz.",
    "i.e.", "e.g.", "ibid.", "supra.", "cf.", "para.",
    "u/s", "s.", "cl.", "r.", "o.", "pt.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
    "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    "sc.", "scc.", "air.", "bom.", "del.", "mad.", "cal.",
}


def _load_spacy(model_name: str):
    """Load a spaCy model, downloading it if necessary."""
    import spacy

    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    except OSError:
        logger.info("Downloading spaCy model '%s'…", model_name)
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    return nlp


def segment_sentences(text: str, spacy_model: str = "en_core_web_sm") -> list[Sentence]:
    """
    Split *text* into sentences using spaCy's dependency parser, with
    custom post-processing for legal abbreviations.

    Returns:
        Ordered list of Sentence objects with character offsets.
    """
    nlp = _load_spacy(spacy_model)

    # Increase max length for long judgments
    nlp.max_length = max(nlp.max_length, len(text) + 1000)

    doc = nlp(text)
    raw_sents: list[tuple[int, int, str]] = []

    for sent in doc.sents:
        raw_sents.append((sent.start_char, sent.end_char, sent.text.strip()))

    # ── Merge broken sentences caused by legal abbreviations ──
    merged: list[tuple[int, int, str]] = []
    i = 0
    while i < len(raw_sents):
        start, end, txt = raw_sents[i]
        # If this "sentence" ends with a known abbreviation, merge with the next
        while i + 1 < len(raw_sents):
            last_token = txt.rstrip().split()[-1].lower() if txt.strip() else ""
            if last_token in _LEGAL_ABBREVS or len(txt.strip()) < 15:
                i += 1
                _, end, next_txt = raw_sents[i]
                txt = txt + " " + next_txt
            else:
                break
        merged.append((start, end, txt))
        i += 1

    sentences = [
        Sentence(text=txt, start_char=s, end_char=e, index=idx)
        for idx, (s, e, txt) in enumerate(merged)
        if txt.strip()  # skip empty
    ]

    logger.info("Segmented document into %d sentences.", len(sentences))
    return sentences


# ────────────────────────────────────────────────────────────
# Stage 2 — Structural Chunking
# ────────────────────────────────────────────────────────────

def _detect_section_breaks(text: str) -> set[int]:
    """
    Return character offsets where we detect major structural breaks
    (numbered paragraphs, headings, double newlines).
    """
    breaks: set[int] = set()
    # Numbered paragraph pattern: start-of-line digit(s) followed by a period
    for m in re.finditer(r'(?m)^\s*\d{1,3}\.\s', text):
        breaks.add(m.start())
    # Double newline
    for m in re.finditer(r'\n\s*\n', text):
        breaks.add(m.start())
    return breaks


def create_chunks(
    sentences: list[Sentence],
    max_chunk_tokens: int = 1024,
    overlap_sentences: int = 2,
    source_text: str = "",
) -> list[ExtractedChunk]:
    """
    Group sentences into coherent chunks respecting paragraph boundaries
    and a maximum token budget, with configurable overlap.

    Args:
        sentences: Ordered sentences from `segment_sentences`.
        max_chunk_tokens: Approximate maximum whitespace-token count per chunk.
        overlap_sentences: Number of sentences to repeat across adjacent chunks.
        source_text: Original document text (used for structural break detection).

    Returns:
        Ordered list of ExtractedChunk objects.
    """
    section_breaks = _detect_section_breaks(source_text) if source_text else set()

    chunks: list[ExtractedChunk] = []
    current_sents: list[Sentence] = []
    current_tokens = 0

    def _flush():
        if not current_sents:
            return
        chunk_text = " ".join(s.text for s in current_sents)
        chunks.append(
            ExtractedChunk(
                chunk_id=len(chunks),
                text=chunk_text,
                start_char=current_sents[0].start_char,
                end_char=current_sents[-1].end_char,
                sentences=list(current_sents),
            )
        )

    for sent in sentences:
        sent_tokens = len(sent.text.split())

        # Check for structural break at this sentence's position
        is_break = any(
            abs(sent.start_char - b) < 5 for b in section_breaks
        )

        # Flush if we'd exceed the budget or we hit a structural break
        if (current_tokens + sent_tokens > max_chunk_tokens and current_sents) or (
            is_break and current_tokens > max_chunk_tokens * 0.3
        ):
            _flush()
            # Keep overlap
            overlap = current_sents[-overlap_sentences:] if overlap_sentences else []
            current_sents = list(overlap)
            current_tokens = sum(len(s.text.split()) for s in current_sents)

        current_sents.append(sent)
        current_tokens += sent_tokens

    _flush()  # final chunk

    logger.info(
        "Created %d chunks (max_tokens=%d, overlap=%d).",
        len(chunks), max_chunk_tokens, overlap_sentences,
    )
    return chunks


# ────────────────────────────────────────────────────────────
# Stage 3 — Salience Ranking (TextRank + Semantic Similarity)
# ────────────────────────────────────────────────────────────

def _compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Compute sentence embeddings using a sentence-transformer model."""
    import os
    from sentence_transformers import SentenceTransformer

    # Try loading from local cache first to avoid unnecessary network calls.
    # Fall back to online download only if local cache is missing.
    try:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("Loaded sentence-transformer from local cache.")
    except Exception:
        logger.info("Local cache miss — downloading sentence-transformer model…")
        model = SentenceTransformer(model_name, local_files_only=False)

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def rank_chunks_by_salience(
    chunks: list[ExtractedChunk],
    embedding_model: str = "all-MiniLM-L6-v2",
    damping: float = 0.85,
    similarity_threshold: float = 0.1,
    top_k: Optional[int] = None,
) -> list[ExtractedChunk]:
    """
    Rank chunks by importance using TextRank with semantic similarity edges.

    Args:
        chunks: Chunks from `create_chunks`.
        embedding_model: Sentence-transformer model for similarity.
        damping: PageRank damping factor.
        similarity_threshold: Minimum cosine similarity to add an edge.
        top_k: Number of top chunks to return (None = return all, ranked).

    Returns:
        Chunks sorted by descending salience score (trimmed to top_k if set).
    """
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    embeddings = _compute_embeddings(texts, embedding_model)

    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # Build similarity matrix
    sim_matrix = normed @ normed.T

    # Build graph
    G = nx.Graph()
    n = len(chunks)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i, j]))

    # PageRank
    try:
        scores = nx.pagerank(G, alpha=damping, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge; using uniform scores.")
        scores = {i: 1.0 / n for i in range(n)}

    for i, chunk in enumerate(chunks):
        chunk.salience_score = scores.get(i, 0.0)

    ranked = sorted(chunks, key=lambda c: c.salience_score, reverse=True)

    if top_k is not None:
        ranked = ranked[:top_k]

    logger.info(
        "Ranked %d chunks; top salience=%.4f, bottom=%.4f.",
        len(ranked),
        ranked[0].salience_score if ranked else 0,
        ranked[-1].salience_score if ranked else 0,
    )
    return ranked


# ────────────────────────────────────────────────────────────
# Public API — Full Retrieval Pipeline
# ────────────────────────────────────────────────────────────

def retrieve(
    text: str,
    config: Optional[RetrieverConfig] = None,
) -> list[ExtractedChunk]:
    """
    Run the full extractive retrieval pipeline on a document.

    Args:
        text: Raw judgment text.
        config: Retriever configuration (uses defaults if None).

    Returns:
        Top-K extracted chunks ranked by salience, each carrying
        character-offset provenance for downstream citation.
    """
    if config is None:
        config = RetrieverConfig()

    # Stage 1: Segment
    sentences = segment_sentences(text, config.spacy_model)

    # Stage 2: Chunk
    chunks = create_chunks(
        sentences,
        max_chunk_tokens=config.max_chunk_tokens,
        overlap_sentences=config.chunk_overlap_sentences,
        source_text=text,
    )

    # Stage 3: Rank
    ranked = rank_chunks_by_salience(
        chunks,
        embedding_model=config.embedding_model,
        damping=config.textrank_damping,
        similarity_threshold=config.similarity_threshold,
        top_k=config.top_k_chunks,
    )

    # Re-sort by document order for coherent summary generation
    ranked.sort(key=lambda c: c.start_char)

    return ranked

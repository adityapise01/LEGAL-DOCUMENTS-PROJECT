"""
Phase 2 — Constrained Rewriting & Span Grounding.

Takes the extracted chunks from the retriever and produces a structured
summary where **every sentence carries a citation** linking it to the
source chunks that support it.

Uses Google Gemini with structured JSON output to enforce the grounding
constraint at generation time.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config import GeneratorConfig
from retriever import ExtractedChunk

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────

@dataclass
class GroundedSentence:
    """A single summary sentence with its grounding citations."""

    sentence: str
    source_chunk_ids: list[int] = field(default_factory=list)
    key_spans: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "source_chunks": self.source_chunk_ids,
            "key_spans": self.key_spans,
        }


@dataclass
class GenerationResult:
    """Full output of the generation phase."""

    grounded_sentences: list[GroundedSentence]
    raw_response: str = ""
    model_used: str = ""

    @property
    def summary_text(self) -> str:
        return " ".join(gs.sentence for gs in self.grounded_sentences)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary_text,
            "grounded_sentences": [gs.to_dict() for gs in self.grounded_sentences],
            "model": self.model_used,
        }


# ────────────────────────────────────────────────────────────
# Prompt Construction
# ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a legal summarisation assistant specialising in Indian court judgments.
Your task is to produce a faithful, concise summary of the judgment based ONLY on the source chunks provided below.

STRICT RULES:
1. Every sentence in your summary MUST be directly supported by one or more of the provided source chunks.
2. You MUST cite the supporting chunk(s) for each sentence using their CHUNK-ID numbers.
3. You MUST quote the exact key phrase(s) from the source that support each sentence.
4. DO NOT add any information, claims, or legal interpretations not present in the source chunks.
5. PRESERVE all qualifying conditions, exact holdings, party names, section numbers, and dates exactly as they appear in the source.
6. DO NOT drop qualifying language (e.g., "subject to", "provided that", "to the extent that").
7. If two chunks contain contradictory information, note the contradiction rather than resolving it.

OUTPUT FORMAT:
Return a JSON array where each element has exactly these fields:
- "sentence": The summary sentence (string)
- "source_chunks": Array of CHUNK-ID numbers that support this sentence (array of integers)
- "key_spans": Array of exact quoted phrases from the source chunks (array of strings)

Example:
[
  {
    "sentence": "The Supreme Court allowed the appeal in part.",
    "source_chunks": [5, 7],
    "key_spans": ["this appeal is allowed in part"]
  }
]"""


def _build_prompt(
    chunks: list[ExtractedChunk],
    target_sentences: int = 15,
) -> str:
    """
    Build the user prompt with numbered source chunks.

    Args:
        chunks: Extracted chunks (should be in document order).
        target_sentences: Suggested number of summary sentences.

    Returns:
        Formatted prompt string.
    """
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(
            f"[CHUNK-{chunk.chunk_id}]\n{chunk.text}"
        )

    chunks_section = "\n\n".join(chunk_texts)

    return (
        f"Below are the source chunks extracted from an Indian legal judgment. "
        f"Summarise the judgment in approximately {target_sentences} sentences, "
        f"strictly following the rules and output format described in your instructions.\n\n"
        f"--- SOURCE CHUNKS ---\n\n{chunks_section}\n\n"
        f"--- END OF SOURCE CHUNKS ---\n\n"
        f"Now produce the JSON summary array."
    )


def _build_regen_prompt(
    failed_sentence: str,
    reason: str,
    chunks: list[ExtractedChunk],
) -> str:
    """Build a targeted prompt for regenerating a single failed sentence."""
    chunk_texts = "\n\n".join(
        f"[CHUNK-{c.chunk_id}]\n{c.text}" for c in chunks
    )

    return (
        f"A previously generated summary sentence was REJECTED by the verification system.\n\n"
        f"REJECTED SENTENCE: \"{failed_sentence}\"\n"
        f"REJECTION REASON: {reason}\n\n"
        f"SOURCE CHUNKS (use ONLY these):\n\n{chunk_texts}\n\n"
        f"Rewrite this single sentence so that it is FULLY supported by the source chunks above. "
        f"Return a JSON object with exactly these fields:\n"
        f'- "sentence": the rewritten sentence\n'
        f'- "source_chunks": array of CHUNK-ID numbers\n'
        f'- "key_spans": array of exact quoted phrases from the source\n'
    )


# ────────────────────────────────────────────────────────────
# API Interaction
# ────────────────────────────────────────────────────────────

def _init_groq(config: GeneratorConfig):
    """Initialise and return the Groq client."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "The 'groq' package is required. "
            "Install with: pip install groq"
        )

    if not config.api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Export it as an environment variable or "
            "pass it via GeneratorConfig.api_key."
        )

    client = Groq(api_key=config.api_key)
    return client


def _call_groq(client, config: GeneratorConfig, prompt: str) -> str:
    """Send a prompt to Groq and return the raw text response."""
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# ────────────────────────────────────────────────────────────
# Response Parsing
# ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> list | dict:
    """
    Parse JSON from the model response, handling markdown code fences
    and other common wrappers.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON array/object in the text
        match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _parse_generation_response(raw: str) -> list[GroundedSentence]:
    """Parse the raw JSON response into GroundedSentence objects."""
    data = _extract_json(raw)

    # Handle both array and single-object responses
    if isinstance(data, dict):
        data = [data]

    sentences: list[GroundedSentence] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sentences.append(
            GroundedSentence(
                sentence=str(item.get("sentence", "")),
                source_chunk_ids=[
                    int(x) for x in item.get("source_chunks", [])
                ],
                key_spans=[str(s) for s in item.get("key_spans", [])],
            )
        )

    return sentences


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def generate(
    chunks: list[ExtractedChunk],
    config: Optional[GeneratorConfig] = None,
) -> GenerationResult:
    """
    Generate a grounded summary from extracted chunks.

    Each output sentence carries citation pointers to the source chunks
    and exact quoted spans that support it.

    Args:
        chunks: Extracted chunks from the retriever (in document order).
        config: Generator configuration (uses defaults if None).

    Returns:
        GenerationResult with grounded sentences.
    """
    if config is None:
        config = GeneratorConfig()

    client = _init_groq(config)
    prompt = _build_prompt(chunks, config.target_summary_sentences)

    last_error: Exception | None = None
    raw_response = ""

    for attempt in range(1 + config.max_parse_retries):
        try:
            raw_response = _call_groq(client, config, prompt)
            grounded = _parse_generation_response(raw_response)

            if not grounded:
                raise ValueError("No sentences parsed from model response.")

            logger.info(
                "Generated %d grounded sentences (attempt %d).",
                len(grounded), attempt + 1,
            )
            return GenerationResult(
                grounded_sentences=grounded,
                raw_response=raw_response,
                model_used=config.model_name,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            logger.warning(
                "Parse attempt %d failed: %s. Retrying…", attempt + 1, e
            )

    raise RuntimeError(
        f"Failed to parse generation response after {1 + config.max_parse_retries} "
        f"attempts. Last error: {last_error}\nRaw response:\n{raw_response[:500]}"
    )


def regenerate_sentence(
    failed_sentence: str,
    reason: str,
    relevant_chunks: list[ExtractedChunk],
    config: Optional[GeneratorConfig] = None,
) -> GroundedSentence:
    """
    Regenerate a single sentence that failed verification.

    Args:
        failed_sentence: The sentence that was rejected.
        reason: Human-readable rejection reason.
        relevant_chunks: The source chunks to ground the rewrite against.
        config: Generator configuration.

    Returns:
        A new GroundedSentence that replaces the failed one.

    Raises:
        RuntimeError: If parsing fails after all retries.
    """
    if config is None:
        config = GeneratorConfig()

    client = _init_groq(config)
    prompt = _build_regen_prompt(failed_sentence, reason, relevant_chunks)

    for attempt in range(1 + config.max_parse_retries):
        try:
            raw = _call_groq(client, config, prompt)
            data = _extract_json(raw)
            if isinstance(data, list):
                data = data[0] if data else {}

            gs = GroundedSentence(
                sentence=str(data.get("sentence", "")),
                source_chunk_ids=[int(x) for x in data.get("source_chunks", [])],
                key_spans=[str(s) for s in data.get("key_spans", [])],
            )
            logger.info("Regenerated sentence (attempt %d).", attempt + 1)
            return gs

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                "Regen parse attempt %d failed: %s", attempt + 1, e
            )

    # Fallback: return the original sentence marked as unregenerable
    logger.error("Regeneration failed for: %s", failed_sentence[:80])
    return GroundedSentence(
        sentence=failed_sentence,
        source_chunk_ids=[],
        key_spans=[],
    )

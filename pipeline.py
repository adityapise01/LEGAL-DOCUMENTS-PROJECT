"""
End-to-End Pipeline Orchestrator.

Ties all modules together into a single workflow:

    Document → Retriever → Generator → Verifier → (Evaluator) → Output JSON

Provides both a programmatic API (`run_pipeline`) and a CLI interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import PipelineConfig
from data_loader import Document, load_from_directory, load_ildc, load_sample_judgment
from retriever import ExtractedChunk, retrieve
from generator import GenerationResult, generate
from verifier import VerificationReport, verify
from evaluate import evaluate_summary

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Pipeline Result
# ────────────────────────────────────────────────────────────

class PipelineResult:
    """Aggregates all outputs from a single pipeline run."""

    def __init__(
        self,
        document: Document,
        chunks: list[ExtractedChunk],
        generation: GenerationResult,
        verification: VerificationReport,
        evaluation: Optional[dict] = None,
        elapsed_seconds: float = 0.0,
    ):
        self.document = document
        self.chunks = chunks
        self.generation = generation
        self.verification = verification
        self.evaluation = evaluation
        self.elapsed_seconds = elapsed_seconds

    def to_dict(self) -> dict:
        return {
            "doc_id": self.document.doc_id,
            "input_chars": self.document.char_count,
            "input_approx_tokens": self.document.approx_token_count,
            "num_chunks_extracted": len(self.chunks),
            "generation": self.generation.to_dict(),
            "verification": self.verification.to_dict(),
            "evaluation": self.evaluation,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def save(self, path: str | Path) -> None:
        """Save the full result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved pipeline result to %s", path)


# ────────────────────────────────────────────────────────────
# Core Pipeline
# ────────────────────────────────────────────────────────────

def run_pipeline(
    document: Document,
    config: Optional[PipelineConfig] = None,
    run_evaluation: bool = True,
) -> PipelineResult:
    """
    Run the full faithfulness-constrained summarisation pipeline on a
    single document.

    Phases:
      1. Extractive selection & chunking (retriever)
      2. Constrained rewriting with span grounding (generator)
      3. Post-hoc NLI verification & regeneration (verifier)
      4. Evaluation against reference summary (if available)

    Args:
        document: The input legal document.
        config: Full pipeline configuration.
        run_evaluation: Whether to run ROUGE/BERTScore/faithfulness metrics.

    Returns:
        PipelineResult with all outputs.
    """
    if config is None:
        config = PipelineConfig()

    config.ensure_dirs()
    start_time = time.time()

    # ── Phase 1: Retrieve ──
    logger.info("=" * 60)
    logger.info("PHASE 1: Extractive Selection & Chunking")
    logger.info("=" * 60)
    chunks = retrieve(document.text, config.retriever)
    logger.info("Extracted %d chunks from %d-char document.", len(chunks), document.char_count)

    # ── Phase 2: Generate ──
    logger.info("=" * 60)
    logger.info("PHASE 2: Constrained Rewriting & Span Grounding")
    logger.info("=" * 60)
    generation = generate(chunks, config.generator)
    logger.info("Generated %d grounded sentences.", len(generation.grounded_sentences))

    # ── Phase 3: Verify ──
    logger.info("=" * 60)
    logger.info("PHASE 3: Post-Hoc Verification & Regeneration")
    logger.info("=" * 60)
    verification = verify(
        generation.grounded_sentences,
        chunks,
        config.verifier,
        config.generator,
    )
    logger.info(
        "Verification: %d passed, %d warned, %d failed. Pass rate: %.1f%%",
        len(verification.passed),
        len(verification.warned),
        len(verification.failed),
        verification.pass_rate * 100,
    )

    # ── Phase 4: Evaluate (optional) ──
    evaluation = None
    if run_evaluation and document.reference_summary:
        logger.info("=" * 60)
        logger.info("PHASE 4: Evaluation")
        logger.info("=" * 60)

        # Build per-sentence source texts for faithfulness eval
        chunk_map = {c.chunk_id: c for c in chunks}
        summary_sents = [gs.sentence for gs in verification.verified_sentences]
        source_texts = []
        for gs in verification.verified_sentences:
            cited = [chunk_map[cid] for cid in gs.source_chunk_ids if cid in chunk_map]
            source_texts.append(" ".join(c.text for c in cited) if cited else "")

        evaluation = evaluate_summary(
            hypothesis=verification.verified_summary,
            reference=document.reference_summary,
            summary_sentences=summary_sents,
            source_texts=source_texts,
            config=config.evaluation,
        )
        logger.info("Evaluation complete.")

    elapsed = time.time() - start_time

    result = PipelineResult(
        document=document,
        chunks=chunks,
        generation=generation,
        verification=verification,
        evaluation=evaluation,
        elapsed_seconds=elapsed,
    )

    # Auto-save output
    output_path = config.output_dir / f"{document.doc_id}_result.json"
    result.save(output_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("Verified summary (%d sentences):", len(verification.verified_sentences))
    logger.info(verification.verified_summary)
    logger.info("=" * 60)

    return result


# ────────────────────────────────────────────────────────────
# Batch Processing
# ────────────────────────────────────────────────────────────

def run_batch(
    documents: list[Document],
    config: Optional[PipelineConfig] = None,
    run_evaluation: bool = True,
) -> list[PipelineResult]:
    """Run the pipeline on multiple documents sequentially."""
    results = []
    for i, doc in enumerate(documents):
        logger.info(
            "\n" + "━" * 60 +
            f"\nProcessing document {i+1}/{len(documents)}: {doc.doc_id}"
            + "\n" + "━" * 60
        )
        try:
            result = run_pipeline(doc, config, run_evaluation)
            results.append(result)
        except Exception as e:
            logger.error("Failed to process %s: %s", doc.doc_id, e)
    return results


# ────────────────────────────────────────────────────────────
# CLI Interface
# ────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithfulness-Constrained Legal Judgment Summarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarise the built-in sample judgment
  python pipeline.py --sample

  # Summarise a local text file
  python pipeline.py --input judgment.txt --output summary.json

  # Process a directory of judgments
  python pipeline.py --input-dir ./data/raw/ --evaluate

  # Load from ILDC dataset (first 5 documents)
  python pipeline.py --dataset ildc --num-docs 5 --evaluate
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sample", action="store_true",
        help="Use the built-in sample judgment for testing.",
    )
    input_group.add_argument(
        "--input", type=str,
        help="Path to a single judgment text file.",
    )
    input_group.add_argument(
        "--input-dir", type=str,
        help="Path to a directory of judgment text files.",
    )
    input_group.add_argument(
        "--dataset", type=str, choices=["ildc"],
        help="Load documents from a HuggingFace dataset.",
    )

    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: outputs/<doc_id>_result.json).",
    )
    parser.add_argument(
        "--num-docs", type=int, default=None,
        help="Maximum number of documents to process (for datasets/directories).",
    )
    parser.add_argument(
        "--evaluate", action="store_true", default=True,
        help="Run evaluation metrics (default: True).",
    )
    parser.add_argument(
        "--no-evaluate", action="store_true",
        help="Skip evaluation metrics.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override: number of top chunks to extract.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    config = PipelineConfig()
    if args.top_k:
        config.retriever.top_k_chunks = args.top_k

    # Load documents
    if args.sample:
        documents = load_sample_judgment()
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        documents = [Document(doc_id=Path(args.input).stem, text=text)]
    elif args.input_dir:
        documents = load_from_directory(args.input_dir)
        if args.num_docs:
            documents = documents[:args.num_docs]
    elif args.dataset == "ildc":
        documents = load_ildc(max_docs=args.num_docs)
    else:
        parser.error("No valid input source specified.")
        return

    run_eval = args.evaluate and not args.no_evaluate

    # Run
    if len(documents) == 1:
        result = run_pipeline(documents[0], config, run_eval)
        if args.output:
            result.save(args.output)
        # Print summary to stdout
        print("\n" + "=" * 60)
        print("VERIFIED SUMMARY")
        print("=" * 60)
        print(result.verification.verified_summary)
        print(f"\nPass rate: {result.verification.pass_rate:.1%}")
        if result.evaluation:
            rouge = result.evaluation.get("rouge", {})
            print(f"ROUGE-L F1: {rouge.get('rougeL', {}).get('f1', 'N/A')}")
            bs = result.evaluation.get("bertscore", {})
            print(f"BERTScore F1: {bs.get('f1', 'N/A')}")
            faith = result.evaluation.get("faithfulness", {})
            if faith:
                print(f"Faithfulness (% entailed): {faith.get('pct_entailed', 'N/A')}%")
    else:
        results = run_batch(documents, config, run_eval)
        print(f"\nProcessed {len(results)}/{len(documents)} documents successfully.")


if __name__ == "__main__":
    main()

# Faithfulness-Constrained Legal Judgment Summarization Pipeline

An end-to-end Python pipeline that produces summaries of Indian legal judgments where **every claim is traceable to a span in the source document**, and that traceability is computationally verified.

## The Problem

Abstractive summarisation models are fluent by construction but truthful only incidentally. In a legal setting, a summary that attributes a holding the court did not make, or drops a qualifying condition, is **actively dangerous**. This pipeline addresses this by:

1. **Extracting** the most salient spans via hierarchical chunking and TextRank
2. **Generating** grounded summaries where every sentence cites its source
3. **Verifying** each citation via NLI entailment and fuzzy span matching
4. **Rejecting & regenerating** any sentence that fails verification

## Architecture

```
Document ──► Retriever ──► Generator ──► Verifier ──► Evaluated Summary
              (spaCy +      (Gemini +     (DeBERTa     (ROUGE +
              TextRank)      JSON)         NLI)         BERTScore +
                                                        Faithfulness)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Your API Key

```bash
# Linux/Mac
export GROQ_API_KEY="your-groq-api-key"

# Windows PowerShell
$env:GROQ_API_KEY = "your-groq-api-key"
```

### 3. Run on the Built-in Sample

```bash
python pipeline.py --sample
```

### 4. Run on Your Own Document

```bash
python pipeline.py --input path/to/judgment.txt
```

### 5. Batch Process a Directory

```bash
python pipeline.py --input-dir ./data/raw/ --evaluate
```

## Modules

| Module | Purpose |
|:---|:---|
| `config.py` | Centralized configuration (models, thresholds, paths) |
| `data_loader.py` | Dataset ingestion (ILDC, local files, built-in sample) |
| `retriever.py` | Phase 1: Sentence segmentation → Chunking → TextRank ranking |
| `generator.py` | Phase 2: Gemini-powered constrained rewriting with span citations |
| `verifier.py` | Phase 3: NLI entailment verification + reject/regenerate loop |
| `evaluate.py` | ROUGE, BERTScore, NLI faithfulness, Inter-annotator agreement |
| `pipeline.py` | End-to-end orchestrator with CLI |

## Output Format

The pipeline produces a JSON file with:

```json
{
  "doc_id": "sample_sc_judgment_001",
  "generation": {
    "summary": "The full summary text...",
    "grounded_sentences": [
      {
        "sentence": "The Supreme Court allowed the appeal in part.",
        "source_chunks": [5, 7],
        "key_spans": ["this appeal is allowed in part"]
      }
    ]
  },
  "verification": {
    "total_sentences": 12,
    "passed": 10,
    "warned": 1,
    "failed": 1,
    "pass_rate": 0.9167,
    "verified_summary": "Only verified sentences..."
  },
  "evaluation": {
    "rouge": {"rouge1": {"f1": 0.45}, "rouge2": {"f1": 0.22}, "rougeL": {"f1": 0.38}},
    "bertscore": {"f1": 0.71},
    "faithfulness": {"pct_entailed": 91.67, "mean_entailment_prob": 0.87}
  }
}
```

## Human Evaluation Protocol

Prepare a CSV with columns: `summary_id, annotator_id, faithfulness_score, adequacy_score, fluency_score`

```python
from evaluate import compute_inter_annotator_agreement
result = compute_inter_annotator_agreement("annotations.csv")
print(f"Cohen's Kappa: {result['kappa']:.4f} ({result['interpretation']})")
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Skip slow tests (no model downloads)
python -m pytest tests/ -v -m "not slow"
```

## Tech Stack

- **Extractive**: spaCy, sentence-transformers, networkx (TextRank)
- **Generative**: Groq API (Llama 3.1 70B — fast inference)
- **Verification**: DeBERTa-v3-large-mnli (NLI entailment)
- **Evaluation**: rouge-score, bert-score, scikit-learn (Cohen's κ)

## License

For academic and research use only.

"""
Streamlit Web Interface for the Faithfulness-Constrained
Legal Judgment Summarization Pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
import logging
from io import StringIO

import streamlit as st

# ── Page config (must be first st call) ──
st.set_page_config(
    page_title="Legal Judgment Summarizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# ── Project imports (after path setup) ──
from config import PipelineConfig
from data_loader import Document, load_sample_judgment
from retriever import retrieve
from generator import generate
from verifier import verify
from evaluate import evaluate_summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    /* ── Global Theming ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .hero-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: 0.05em;
    }

    /* ── Phase Cards ── */
    .phase-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .phase-card h3 {
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.5rem 0;
    }
    .phase-card .phase-stat {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .phase-card .phase-label {
        color: #64748b;
        font-size: 0.8rem;
    }

    /* ── Verdict Badges ── */
    .verdict-pass {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .verdict-warn {
        display: inline-block;
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .verdict-fail {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ── Summary Sentence Block ── */
    .sentence-block {
        background: #1e293b;
        border-left: 3px solid #6366f1;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-radius: 0 8px 8px 0;
    }
    .sentence-block p {
        color: #e2e8f0;
        margin: 0 0 0.5rem 0;
        line-height: 1.6;
    }
    .sentence-meta {
        color: #64748b;
        font-size: 0.8rem;
    }
    .chunk-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 4px;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-name {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* ── Input Text Area ── */
    .stTextArea textarea {
        font-family: 'Inter', monospace;
        font-size: 0.88rem;
    }

    /* ── Sidebar Polish ── */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hero Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div class="hero-header">
    <span class="hero-badge">⚖️ AI-POWERED LEGAL NLP</span>
    <h1>Faithfulness-Constrained Legal Judgment Summarizer</h1>
    <p>
        Every claim in the summary is <strong>traceable to a span in the source</strong>.
        Powered by extractive retrieval → grounded generation → NLI verification.
    </p>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    st.markdown("---")

    # API Key
    groq_key = st.text_input(
        "🔑 Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Your Groq API key. Also reads from GROQ_API_KEY env var.",
    )

    st.markdown("---")
    st.markdown("##### 🔧 Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Lower = more faithful, higher = more creative")
    max_tokens = st.slider("Max Output Tokens", 200, 2000, 800, 50,
                           help="Max tokens for the generated summary")
    target_sentences = st.slider("Target Sentences", 5, 25, 15,
                                 help="Desired number of summary sentences")

    st.markdown("---")
    st.markdown("##### 🔍 Verification")
    entailment_threshold = st.slider("Entailment Threshold", 0.3, 0.95, 0.7, 0.05,
                                     help="Minimum NLI score to pass verification")
    max_regen = st.slider("Max Regen Attempts", 0, 5, 3,
                          help="Retry count for failed sentences")

    st.markdown("---")
    st.markdown("##### 📊 Evaluation")
    run_eval = st.checkbox("Run evaluation metrics", value=True,
                           help="Compute ROUGE, BERTScore & faithfulness")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "Built with Retriever → Generator → Verifier<br>"
        "NLI backbone: DeBERTa-v3-base"
        "</div>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Content — Input Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

input_tab, upload_tab, sample_tab = st.tabs([
    "📝 Paste Text",
    "📁 Upload File",
    "🧪 Sample Judgment",
])

document = None

with input_tab:
    user_text = st.text_area(
        "Paste your legal judgment text here:",
        height=300,
        placeholder="Paste the full text of an Indian legal judgment…",
    )
    ref_summary = st.text_area(
        "Reference summary (optional, for evaluation):",
        height=100,
        placeholder="Paste a reference summary if available…",
    )
    if user_text.strip():
        document = Document(
            doc_id="user_input",
            text=user_text.strip(),
            reference_summary=ref_summary.strip() if ref_summary.strip() else None,
        )

with upload_tab:
    uploaded = st.file_uploader(
        "Upload a .txt file containing a legal judgment",
        type=["txt"],
    )
    if uploaded is not None:
        content = StringIO(uploaded.getvalue().decode("utf-8")).read()
        document = Document(
            doc_id=uploaded.name.replace(".txt", ""),
            text=content,
        )
        st.success(f"Loaded **{uploaded.name}** — {len(content):,} characters")

with sample_tab:
    st.info("Click below to load a built-in Indian Supreme Court judgment for testing.")
    if st.button("🚀 Load Sample Judgment", use_container_width=True):
        docs = load_sample_judgment()
        document = docs[0]
        st.session_state["sample_loaded"] = True

    if st.session_state.get("sample_loaded"):
        docs = load_sample_judgment()
        document = docs[0]
        st.success(f"Loaded sample: **{document.doc_id}** — {document.char_count:,} chars, ~{document.approx_token_count:,} tokens")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("---")

run_button = st.button(
    "⚡ Run Summarization Pipeline",
    use_container_width=True,
    type="primary",
    disabled=(document is None),
)

if document is None:
    st.markdown(
        "<p style='text-align:center; color:#64748b;'>"
        "👆 Provide a document above to get started</p>",
        unsafe_allow_html=True,
    )

if run_button and document is not None:
    # Validate API key
    api_key = groq_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("❌ Please provide a Groq API key in the sidebar.")
        st.stop()

    # Build config
    config = PipelineConfig()
    config.generator.api_key = api_key
    config.generator.temperature = temperature
    config.generator.max_output_tokens = max_tokens
    config.generator.target_summary_sentences = target_sentences
    config.verifier.entailment_threshold = entailment_threshold
    config.verifier.max_regen_attempts = max_regen

    start_time = time.time()

    # ── Phase 1 ──
    with st.status("🔍 Phase 1: Extractive Selection & Chunking…", expanded=True) as status:
        st.write("Segmenting document into sentences and creating overlapping chunks…")
        chunks = retrieve(document.text, config.retriever)
        status.update(
            label=f"✅ Phase 1 Complete — {len(chunks)} chunks extracted",
            state="complete",
        )

    # ── Phase 2 ──
    with st.status("✍️ Phase 2: Constrained Rewriting & Span Grounding…", expanded=True) as status:
        st.write(f"Generating grounded summary via Groq ({config.generator.model_name})…")
        generation = generate(chunks, config.generator)
        status.update(
            label=f"✅ Phase 2 Complete — {len(generation.grounded_sentences)} grounded sentences",
            state="complete",
        )

    # ── Phase 3 ──
    with st.status("🔬 Phase 3: Post-Hoc NLI Verification…", expanded=True) as status:
        st.write("Verifying each sentence with DeBERTa NLI entailment checks…")
        verification = verify(
            generation.grounded_sentences,
            chunks,
            config.verifier,
            config.generator,
        )
        pass_pct = verification.pass_rate * 100
        status.update(
            label=f"✅ Phase 3 Complete — {pass_pct:.0f}% pass rate",
            state="complete",
        )

    # ── Phase 4 (optional) ──
    evaluation = None
    if run_eval and document.reference_summary:
        with st.status("📊 Phase 4: Evaluation Metrics…", expanded=True) as status:
            st.write("Computing ROUGE, BERTScore, and faithfulness…")
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
            status.update(label="✅ Phase 4 Complete — Evaluation done", state="complete")

    elapsed = time.time() - start_time

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Results Display
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    st.markdown("---")
    st.markdown("## 📋 Results")

    # ── Overview Metrics Row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Pipeline Time", f"{elapsed:.1f}s")
    with col2:
        st.metric("📄 Chunks Extracted", len(chunks))
    with col3:
        st.metric("📝 Verified Sentences", len(verification.verified_sentences))
    with col4:
        st.metric("✅ Pass Rate", f"{verification.pass_rate * 100:.0f}%")

    # ── Verified Summary ──
    st.markdown("### 📖 Verified Summary")
    st.markdown(
        f"<div class='phase-card'><p style='color:#e2e8f0; line-height:1.8;'>"
        f"{verification.verified_summary}"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # ── Per-Sentence Breakdown ──
    st.markdown("### 🔬 Sentence-Level Verification")

    for i, vr in enumerate(verification.results):
        verdict_class = {
            "PASS": "verdict-pass",
            "WARN": "verdict-warn",
            "FAIL": "verdict-fail",
            "DROPPED": "verdict-fail",
        }.get(vr.verdict.value, "verdict-fail")

        verdict_emoji = {
            "PASS": "✅",
            "WARN": "⚠️",
            "FAIL": "❌",
            "DROPPED": "🗑️",
        }.get(vr.verdict.value, "❓")

        chunk_tags = " ".join(
            f'<span class="chunk-tag">Chunk {cid}</span>'
            for cid in vr.sentence.source_chunk_ids
        )

        st.markdown(f"""
        <div class="sentence-block">
            <p><strong>S{i+1}.</strong> {vr.sentence.sentence}</p>
            <div class="sentence-meta">
                <span class="{verdict_class}">{verdict_emoji} {vr.verdict.value}</span>
                &nbsp;·&nbsp; NLI: <strong>{vr.entailment_score:.2f}</strong>
                ({vr.nli_label})
                &nbsp;·&nbsp; {chunk_tags}
                {f'&nbsp;·&nbsp; Regen: {vr.regen_attempts}x' if vr.regen_attempts > 0 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Evaluation Metrics (if available) ──
    if evaluation:
        st.markdown("### 📊 Evaluation Metrics")

        rouge = evaluation.get("rouge", {})
        bert = evaluation.get("bertscore", {})
        faith = evaluation.get("faithfulness", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### ROUGE Scores")
            r1 = rouge.get("rouge1", {})
            r2 = rouge.get("rouge2", {})
            rL = rouge.get("rougeL", {})
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{r1.get('fmeasure', 0):.3f}</div>
                <div class="metric-name">ROUGE-1 F1</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{r2.get('fmeasure', 0):.3f}</div>
                <div class="metric-name">ROUGE-2 F1</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rL.get('fmeasure', 0):.3f}</div>
                <div class="metric-name">ROUGE-L F1</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### BERTScore")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bert.get('precision', 0):.3f}</div>
                <div class="metric-name">Precision</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bert.get('recall', 0):.3f}</div>
                <div class="metric-name">Recall</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bert.get('f1', 0):.3f}</div>
                <div class="metric-name">F1</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("#### Faithfulness (NLI)")
            pct_entailed = faith.get("pct_entailed", 0)
            mean_ent = faith.get("mean_entailment_prob", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pct_entailed:.0f}%</div>
                <div class="metric-name">% Sentences Entailed</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mean_ent:.3f}</div>
                <div class="metric-name">Mean Entailment Prob</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Extracted Chunks (Expandable) ──
    with st.expander("📦 View Extracted Chunks"):
        for chunk in chunks:
            st.markdown(f"**Chunk {chunk.chunk_id}** (salience: {chunk.salience_score:.4f})")
            st.text(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))
            st.markdown("---")

    # ── Raw JSON Output ──
    with st.expander("🗃️ Raw JSON Output"):
        import json
        result_dict = {
            "doc_id": document.doc_id,
            "input_chars": document.char_count,
            "num_chunks": len(chunks),
            "generation": generation.to_dict(),
            "verification": verification.to_dict(),
            "evaluation": evaluation,
            "elapsed_seconds": round(elapsed, 2),
        }
        st.json(result_dict)

    st.balloons()

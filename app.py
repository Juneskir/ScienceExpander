"""
Science Expander — Interdisciplinary Discovery Engine
Production Streamlit Web Interface (Sprint 5)

Run with:
    streamlit run app.py
"""

import sys
import os
import json
import re
from typing import List, Dict, Optional
import streamlit as st
import streamlit.components.v1 as components

# Import backend discovery engine
from science_expander import (
    MultilingualBridge,
    TopicCrossBreeder,
    SemanticCorridorEngine,
    ScholarLiteratureClient,
    OpenAlexClient,
    GeneratedTopic,
    RetrievedPaper,
    LiteratureTriad,
    generate_knowledge_graph,
    generate_research_brief,
    generate_bibtex,
    classify_topic,
    clean_topic_name,
    ScienceCategory,
)

# ---------------------------------------------------------------------------
# Custom Styling (Dark/Modern Glassmorphism & Sleek Badges)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main typography and headers */
h1, h2, h3 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 700;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}

/* Topic & Paper Cards */
.card-container {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.card-container:hover {
    border-color: rgba(99, 102, 241, 0.45);
    transform: translateY(-2px);
}

/* Affinity Badges */
.badge-high {
    background: linear-gradient(135deg, #059669, #10b981);
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
}
.badge-med {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
}
.badge-triad {
    background: #1e1b4b;
    border: 1px solid #6366f1;
    color: #a5b4fc;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.badge-source {
    background: #0f172a;
    border: 1px solid #475569;
    color: #cbd5e1;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    display: inline-block;
    margin-right: 4px;
}

/* Metadata pill */
.meta-pill {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(100, 116, 139, 0.3);
    color: #e2e8f0;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.82rem;
    display: inline-block;
    margin-right: 6px;
    margin-bottom: 4px;
}
</style>
"""


def compute_display_seeds(raw_seeds: List[str]) -> List[str]:
    """Return bilingual seed labels if Cyrillic detected."""
    display_seeds = []
    for s in raw_seeds:
        if MultilingualBridge.is_cyrillic(s):
            en = MultilingualBridge.translate_to_en(s)
            display_seeds.append(MultilingualBridge.format_bilingual(s, en))
        else:
            display_seeds.append(s)
    return display_seeds


def run_cross_breeding_core(raw_input: str, count: int = 10):
    """Executes topic synthesis, bilingual translation, and graph generation."""
    raw_seeds = [t.strip() for t in raw_input.split(",") if t.strip()]
    if not raw_seeds:
        raw_seeds = ["Cell Biology"]

    display_seeds = compute_display_seeds(raw_seeds)

    # Cross-breeding
    breeder = TopicCrossBreeder()
    topics = breeder.cross_breed(raw_seeds, count=count)

    # Attach bilingual labels to generated topics
    bilingual_map = {clean_topic_name(r): d for r, d in zip(raw_seeds, display_seeds)}
    for g in topics:
        if not g.display_primary_topic or g.display_primary_topic == g.primary_topic:
            g.display_primary_topic = bilingual_map.get(g.primary_topic, g.primary_topic)
        if g.secondary_topic and (not g.display_secondary_topic or g.display_secondary_topic == g.secondary_topic):
            g.display_secondary_topic = bilingual_map.get(g.secondary_topic, g.secondary_topic)

    # Generate initial knowledge graph with empty triad
    empty_triad = LiteratureTriad()
    graph_html = generate_knowledge_graph(topics[0], empty_triad, topics, display_seeds)

    return raw_seeds, display_seeds, topics, graph_html


def render_app():
    """Main rendering loop for the Streamlit web application."""
    st.set_page_config(
        page_title="Science Expander — Interdisciplinary Discovery Engine",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Session State Initialization
    # -----------------------------------------------------------------------
    if "seed_topics_input" not in st.session_state:
        st.session_state.seed_topics_input = "Cell Biology"
    if "generated_topics" not in st.session_state:
        st.session_state.generated_topics = []
    if "seed_topics_list" not in st.session_state:
        st.session_state.seed_topics_list = ["Cell Biology"]
    if "display_seeds_list" not in st.session_state:
        st.session_state.display_seeds_list = ["Cell Biology"]
    if "selected_topic_idx" not in st.session_state:
        st.session_state.selected_topic_idx = 0
    if "triads_cache" not in st.session_state:
        st.session_state.triads_cache = {}  # {topic_idx: LiteratureTriad}
    if "knowledge_graph_html" not in st.session_state:
        st.session_state.knowledge_graph_html = ""

    # -----------------------------------------------------------------------
    # Sidebar Configuration & Architecture Overview
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🛰️ Science Expander")
        st.caption("Zero-Token Interdisciplinary Research Engine")

        st.markdown("---")
        st.markdown("#### 🧠 Local Vector Intelligence")
        vec_engine = SemanticCorridorEngine.get_instance()
        if vec_engine.is_available:
            st.success("🟢 ONNX FastEmbed Active\n`bge-small-en-v1.5`")
        else:
            st.warning("🟡 Rule-Based Engine\nFallback Active")

        st.markdown("---")
        st.markdown("#### ⚙️ Engine Parameters")
        topic_count = st.slider("Discovery Yield (Topics)", min_value=5, max_value=14, value=10, step=1)

        st.markdown("---")
        st.markdown("#### 🧩 Core Architecture")
        st.markdown(
            """
            - **Zero LLM Token Cost**: 100% deterministic conceptual blending ontology.
            - **FastEmbed ONNX**: Local semantic cosine similarity & Goldilocks corridor scoring.
            - **Domain Affinity Matrix**: 14 sanitized scientific faculties; zero deep learning buzzwords.
            - **Tri-Axial Retrieval**: Landmark Foundation, Frontier 2024–2026, and Review Synthesis.
            - **Resilient Fallback**: Google Scholar ➔ OpenAlex (`relevance_score:desc`) ➔ Crossref API.
            - **Multilingual Bridge**: Zero-cost Russian (RU ↔ EN) academic translation layer.
            - **Interactive Topology**: Vis.js force-directed physics graph with vector-weighted edges.
            """
        )

        st.markdown("---")
        st.caption("Version 5.0 • Released under MIT License")

    # -----------------------------------------------------------------------
    # Main Header & Hero Section
    # -----------------------------------------------------------------------
    st.markdown('<div class="hero-title">🛰️ Science Expander</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Algorithmic Interdisciplinary Discovery • Tri-Axial Literature Retrieval • Multilingual (RU ↔ EN) Bridge</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Input Section & Preset Demo Buttons
    # -----------------------------------------------------------------------
    st.markdown("##### 🧪 Select a Preset Demo or Enter Custom Scientific Topics:")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🧬 Cell Biology", use_container_width=True):
            st.session_state.seed_topics_input = "Cell Biology"
            st.rerun()
    with col_p2:
        if st.button("🧠 Neurobiology & Quantum Physics", use_container_width=True):
            st.session_state.seed_topics_input = "Neurobiology, Quantum Physics"
            st.rerun()
    with col_p3:
        if st.button("🔬 Клеточная биология (RU)", use_container_width=True):
            st.session_state.seed_topics_input = "Клеточная биология"
            st.rerun()
    with col_p4:
        if st.button("📊 Topology & Genomics", use_container_width=True):
            st.session_state.seed_topics_input = "Topological Data Analysis, Genomics"
            st.rerun()

    col_input, col_btn = st.columns([4, 1.2])
    with col_input:
        user_input = st.text_input(
            "Enter topics (comma-separated, Russian or English):",
            value=st.session_state.seed_topics_input,
            placeholder="e.g. Клеточная биология OR Neurobiology, Quantum Physics",
            label_visibility="collapsed",
        )
    with col_btn:
        generate_btn = st.button("🚀 Explore Topics", type="primary", use_container_width=True)

    # Check if we need to run cross-breeding
    if generate_btn or not st.session_state.generated_topics:
        with st.spinner("Analyzing semantic domain affinity & generating interdisciplinary matrix..."):
            raw_s, disp_s, tops, g_html = run_cross_breeding_core(user_input, topic_count)
            st.session_state.seed_topics_list = raw_s
            st.session_state.display_seeds_list = disp_s
            st.session_state.generated_topics = tops
            st.session_state.seed_topics_input = user_input
            st.session_state.knowledge_graph_html = g_html
            st.session_state.triads_cache = {}
            st.session_state.selected_topic_idx = 0

    generated_topics: List[GeneratedTopic] = st.session_state.generated_topics
    display_seeds_list: List[str] = st.session_state.display_seeds_list

    # -----------------------------------------------------------------------
    # Navigation Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        f"💡 10 Cross-Disciplinary Ideas ({len(generated_topics)})",
        "📚 Literature Triad (3-Axis)",
        "🕸️ Interactive Knowledge Graph",
        "💾 Export & BibTeX",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: 10 Cross-Disciplinary Ideas
    # -----------------------------------------------------------------------
    with tab1:
        st.markdown("### 💡 Algorithmic Discovery Matrix")
        st.markdown(
            r"Filtered using the **Scientific Domain Affinity Matrix** ($\ge 80\%$ semantic compatibility) "
            "to guarantee legitimate, peer-reviewed scientific intersections without pseudo-scientific buzzwords."
        )

        # Metric Bar
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Seed Topics", ", ".join(display_seeds_list[:2]))
        with m_col2:
            top_cat = classify_topic(st.session_state.seed_topics_list[0]).value.replace("_", " ").title()
            st.metric("Primary Faculty", top_cat)
        with m_col3:
            st.metric("Discovery Yield", f"{len(generated_topics)} Topics")
        with m_col4:
            min_aff = min((t.affinity for t in generated_topics), default=0.85)
            st.metric("Min Domain Affinity", f"{int(min_aff * 100)}%")

        st.markdown("---")

        # Render topic cards
        for idx, top in enumerate(generated_topics):
            col_content, col_action = st.columns([5, 1.2])

            aff_pct = int(top.affinity * 100)
            if aff_pct >= 95:
                badge_html = f'<span class="badge-high">{aff_pct}% HIGH AFFINITY</span>'
            else:
                badge_html = f'<span class="badge-med">{aff_pct}% MED AFFINITY</span>'

            with col_content:
                st.markdown(
                    f"""
                    <div class="card-container">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; color: #f8fafc;">#{top.index:02d}. {top.title}</h4>
                            {badge_html}
                        </div>
                        <p style="color: #cbd5e1; margin-bottom: 0.6rem; font-size: 0.95rem;">
                            <strong>Synthesized Concept:</strong> {top.rationale}
                        </p>
                        <div style="margin-bottom: 0.4rem;">
                            <span class="meta-pill">🏛️ <strong>Bridge Domain:</strong> {top.domain.name}</span>
                            <span class="meta-pill">🔍 <strong>Methodological Lens:</strong> {top.operator_name}</span>
                            <span class="meta-pill">🧭 <strong>Vector Sim:</strong> {f'{top.cosine_similarity:.2f} ({top.vector_zone})' if top.cosine_similarity is not None else 'Rule-Based'}</span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.85rem; font-family: monospace;">
                            <strong>Target Literature Query:</strong> <code>{top.primary_query}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_action:
                st.write("")
                st.write("")
                if st.button(f"Select #{top.index:02d} for Triad", key=f"btn_explore_{idx}", use_container_width=True):
                    st.session_state.selected_topic_idx = idx
                    st.toast(f"Selected Topic #{top.index:02d}: {top.title[:30]}... Switch to Tab 2 to retrieve literature!", icon="📚")

    # -----------------------------------------------------------------------
    # TAB 2: Literature Triad (Foundation, Frontier, Review)
    # -----------------------------------------------------------------------
    with tab2:
        st.markdown("### 📚 Tri-Axial Literature Retrieval (3-Dimensional Slice)")
        st.markdown(
            "Fetches a 3-dimensional academic perspective: **The Foundation** (most-cited landmark paper), "
            "**The Frontier** (cutting-edge publication / preprint published between 2024 and 2026), and "
            "**The Review** (comprehensive state-of-the-art survey)."
        )

        topic_options = [
            f"#{t.index:02d}: {t.title} ({t.domain.name})" for t in generated_topics
        ]
        selected_str = st.selectbox(
            "Choose a synthesized idea to retrieve literature for:",
            options=topic_options,
            index=min(st.session_state.selected_topic_idx, len(topic_options) - 1),
        )
        current_idx = topic_options.index(selected_str)
        current_topic = generated_topics[current_idx]

        col_t1, col_t2 = st.columns([1.5, 3])
        with col_t1:
            fetch_triad_btn = st.button("📥 Retrieve 3-Axis Literature Triad", type="primary", use_container_width=True)

        triad: Optional[LiteratureTriad] = st.session_state.triads_cache.get(current_idx)

        if fetch_triad_btn:
            status_box = st.status("Querying Academic Knowledge Bases (Scholar ➔ OpenAlex ➔ Crossref)...", expanded=True)
            try:
                status_box.write("🛰️ Searching Google Scholar & OpenAlex with relevance sorting...")
                fetched_triad = ScholarLiteratureClient.fetch_triad(current_topic, silent=True)
                st.session_state.triads_cache[current_idx] = fetched_triad
                triad = fetched_triad

                # Re-generate interactive graph to include newly retrieved literature nodes
                updated_graph = generate_knowledge_graph(
                    current_topic,
                    fetched_triad,
                    generated_topics,
                    display_seeds_list,
                )
                st.session_state.knowledge_graph_html = updated_graph

                status_box.update(label="✅ Literature Triad Retrieved Successfully!", state="complete", expanded=False)
            except Exception as e:
                status_box.update(label=f"⚠️ Retrieval error: {str(e)}", state="error")

        if triad:
            st.markdown("---")
            st.markdown(f"#### 📖 Retrieved Literature Triad for: *{current_topic.title}*")

            cards_data = [
                ("🏛️ The Foundation (Фундамент)", "Seminal Landmark Paper Establishing the Field", triad.foundation),
                ("⚡ The Frontier (Фронтир, 2024–2026)", "Cutting-Edge Publication / Preprint", triad.frontier),
                ("📚 The Review (Обзор)", "Comprehensive Synthesis & State-of-the-Art Survey", triad.review),
            ]

            col_c1, col_c2, col_c3 = st.columns(3)
            cols = [col_c1, col_c2, col_c3]

            for i, (role_title, role_desc, paper) in enumerate(cards_data):
                with cols[i]:
                    if paper:
                        authors_str = ", ".join(paper.authors[:3])
                        if len(paper.authors) > 3:
                            authors_str += " et al."
                        year_str = str(paper.pub_year) if paper.pub_year else "N/A"
                        cites_str = f"{paper.citations:,}" if paper.citations is not None else "N/A"
                        source_str = paper.source or "Academic Index"

                        st.markdown(
                            f"""
                            <div class="card-container" style="min-height: 380px; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <span class="badge-triad">{role_title}</span>
                                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.6rem;">{role_desc}</div>
                                    <h4 style="margin-top: 0; font-size: 1.05rem; color: #f1f5f9; line-height: 1.4;">
                                        <a href="{paper.url}" target="_blank" style="color: #60a5fa; text-decoration: none;">{paper.title}</a>
                                    </h4>
                                    <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.4rem;">
                                        <strong>Authors:</strong> {authors_str}
                                    </p>
                                    <div style="margin-bottom: 0.6rem;">
                                        <span class="badge-source">📅 {year_str}</span>
                                        <span class="badge-source">⭐ {cites_str} citations</span>
                                        <span class="badge-source">🔗 {source_str}</span>
                                    </div>
                                    <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4; max-height: 120px; overflow: hidden; text-overflow: ellipsis;">
                                        {paper.abstract or "No abstract preview available."}
                                    </p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if paper.oa_url:
                            st.link_button(
                                "📥 Download Open Access PDF",
                                paper.oa_url,
                                use_container_width=True,
                                type="primary",
                            )
                        elif paper.url:
                            st.link_button(
                                "🌐 View Publication Page",
                                paper.url,
                                use_container_width=True,
                            )
                    else:
                        st.markdown(
                            f"""
                            <div class="card-container" style="min-height: 380px;">
                                <span class="badge-triad">{role_title}</span>
                                <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 1rem;">{role_desc}</div>
                                <p style="color: #64748b; font-style: italic;">No publication retrieved for this category.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("👆 Click **'Retrieve 3-Axis Literature Triad'** to fetch Foundation, Frontier (2024–2026), and Review papers.")

    # -----------------------------------------------------------------------
    # TAB 3: Interactive Knowledge Graph
    # -----------------------------------------------------------------------
    with tab3:
        st.markdown("### 🕸️ Interactive Knowledge Topology Graph")
        st.markdown(
            "A force-directed network rendered with **Vis.js Physics**. Explore topological clusters connecting your "
            "seed disciplines (neon cyan hubs) through bridge domains (purple diamonds) to synthesized research vectors (green/emerald dots) "
            "and retrieved literature triad stars (gold/amber)."
        )

        if st.session_state.knowledge_graph_html:
            components.html(
                st.session_state.knowledge_graph_html,
                height=750,
                scrolling=False,
            )

            st.caption(
                "💡 **Interaction Tips**: Drag nodes to inspect local connections • Scroll wheel to zoom in/out • "
                "Double-click to release or pin nodes • The physics simulation automatically stabilizes into harmonic equilibrium."
            )
        else:
            st.warning("Generate topics first to render the interactive knowledge topology.")

    # -----------------------------------------------------------------------
    # TAB 4: Export & BibTeX
    # -----------------------------------------------------------------------
    with tab4:
        st.markdown("### 💾 Export Research Assets & Citations")
        st.markdown(
            "Download comprehensive research dossiers, formatted BibTeX citations for your reference manager, "
            "or standalone interactive HTML knowledge graphs."
        )

        current_triad = st.session_state.triads_cache.get(current_idx) or LiteratureTriad()
        brief_markdown = generate_research_brief(
            current_topic,
            current_triad,
            generated_topics,
            display_seeds_list,
        )

        # Consolidated BibTeX across all cached triads
        all_bibtex_entries = []
        for t_idx, t_obj in st.session_state.triads_cache.items():
            for p in t_obj.papers:
                all_bibtex_entries.append(generate_bibtex(p))

        if not all_bibtex_entries and current_triad.papers:
            for p in current_triad.papers:
                all_bibtex_entries.append(generate_bibtex(p))

        bibtex_str = (
            "\n\n".join(all_bibtex_entries)
            if all_bibtex_entries
            else "% No papers retrieved yet. Fetch a triad in Tab 2 to generate BibTeX."
        )

        json_export = json.dumps(
            {
                "seed_topics": display_seeds_list,
                "generated_topics": [
                    {
                        "index": t.index,
                        "title": t.title,
                        "rationale": t.rationale,
                        "bridge_domain": t.domain.name,
                        "affinity": t.affinity,
                        "cosine_similarity": t.cosine_similarity,
                        "goldilocks_score": t.goldilocks_score,
                        "vector_zone": t.vector_zone,
                        "primary_query": t.primary_query,
                        "operator_name": t.operator_name,
                    }
                    for t in generated_topics
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

        st.markdown("#### 📦 One-Click Downloads")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        with d_col1:
            st.download_button(
                label="📄 Research_Brief.md",
                data=brief_markdown,
                file_name="Research_Brief.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with d_col2:
            st.download_button(
                label="📚 references.bib",
                data=bibtex_str,
                file_name="references.bib",
                mime="text/x-bibtex",
                use_container_width=True,
            )

        with d_col3:
            st.download_button(
                label="🕸️ knowledge_graph.html",
                data=st.session_state.knowledge_graph_html,
                file_name="knowledge_graph.html",
                mime="text/html",
                use_container_width=True,
            )

        with d_col4:
            st.download_button(
                label="📊 discovery_matrix.json",
                data=json_export,
                file_name="discovery_matrix.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### 📜 Consolidated BibTeX Preview")
        st.code(bibtex_str, language="bibtex")

        st.markdown("#### 📝 Research Brief Markdown Preview")
        with st.expander("Expand to preview Research_Brief.md"):
            st.markdown(brief_markdown)


# ---------------------------------------------------------------------------
# Entry Point Check
# ---------------------------------------------------------------------------
if __name__ == "__main__" or (hasattr(st, "runtime") and st.runtime.exists()):
    render_app()

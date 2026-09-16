# 🛰️ Science Expander — Interdisciplinary Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero-Token Architecture](https://img.shields.io/badge/LLM%20Tokens-0%20(Free)-success.svg)](https://github.com/)
[![FastEmbed ONNX](https://img.shields.io/badge/Vectors-FastEmbed%20ONNX-38bdf8.svg)](https://qdrant.github.io/fastembed/)
[![OpenAlex API](https://img.shields.io/badge/OpenAlex-Indexed-orange.svg)](https://openalex.org/)
[![Crossref Failover](https://img.shields.io/badge/Crossref-API%20Failover-yellow.svg)](https://www.crossref.org/)
[![Vis.js Physics](https://img.shields.io/badge/Topology-Vis.js%20Force--Directed-blueviolet.svg)](https://visjs.org/)
[![Streamlit UI](https://img.shields.io/badge/Web%20App-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Algorithmic Cross-Disciplinary Research Engine**: Synthesizes high-affinity crossover scientific directions without paid LLM tokens, computes local vector similarities via FastEmbed ONNX with Goldilocks sweet-spot scoring, fetches 3-dimensional academic literature triads (*Foundation, Frontier 2024–2026, Review*), and maps topological knowledge networks in real time.

---

## 📑 Table of Contents
- [Why Science Expander?](#-why-science-expander)
- [System Architecture](#-system-architecture)
  - [1. Conceptual Blending & Domain Affinity Matrix](#1-conceptual-blending--domain-affinity-matrix)
  - [2. Local Vector Intelligence via FastEmbed ONNX (Semantic Corridor Engine)](#2-local-vector-intelligence-via-fastembed-onnx-semantic-corridor-engine)
  - [3. Tri-Axial Literature Retrieval (3D Academic Slice)](#3-tri-axial-literature-retrieval-3d-academic-slice)
  - [4. Zero-Token Multilingual Bridge (RU ↔ EN)](#4-zero-token-multilingual-bridge-ru--en)
  - [5. Interactive Knowledge Topology (Vis.js Physics)](#5-interactive-knowledge-topology-visjs-physics)
- [Quickstart Installation](#-quickstart-installation)
- [Streamlit Web Application](#-streamlit-web-application)
- [Command-Line Interface (CLI)](#-command-line-interface-cli)
- [Generated Research Artifacts](#-generated-research-artifacts)
- [Automated Test Suite](#-automated-test-suite)
- [License](#-license)

---

## 🔭 Why Science Expander?

Most AI research brainstorming tools rely on proprietary LLMs that hallucinate citations, incur recurring API token costs, and produce shallow buzzword combinations (*e.g., crossing cell biology with "activation patching" or "grokking"*).

**Science Expander** takes a rigorous, epistemologically sound approach:
1. **Zero LLM Token Ingestion**: 100% deterministic, reproducible conceptual blending derived from cognitive metaphor theory and scientific epistemology.
2. **Sanitized Ontology**: Hyper-niche AI buzzwords are replaced with 14 foundational interdisciplinary disciplines.
3. **Domain Affinity Matrix**: Pairs disciplines with $\ge 80\%$ semantic compatibility, guaranteeing legitimate physical and mathematical interfaces.
4. **Triple-Layer Academic Resilience**: Queries Google Scholar, OpenAlex, and Crossref with automatic failover so research pipelines never stall on CAPTCHAs or rate limits.

---

## 🧩 System Architecture

```
                    ┌──────────────────────────────┐
                    │    Input Scientific Topic    │
                    │   (e.g., Клеточная биология) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Multilingual Bridge (RU↔EN) │
                    │   (Zero-Token deep-transl)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Domain Affinity Matrix     │
                    │   (14 Sanitized Faculties)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 10 High-Affinity Directions  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
  ┌─────────────────────────┐             ┌─────────────────────────┐
  │   Tri-Axial Retrieval   │             │   Knowledge Topology    │
  │   Foundation (Seminal)  │             │   Force-Directed Graph  │
  │   Frontier (2024–2026)  │             │   (Vis.js Physics HTML) │
  │   Review (State-of-Art) │             └─────────────────────────┘
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Export: Brief, BibTeX  │
  └─────────────────────────┘
```

### 1. Conceptual Blending & Domain Affinity Matrix
Topics are algorithmically categorized into core scientific faculties (`BIOLOGY`, `PHYSICS`, `MATERIALS`, `MATHEMATICS`, `COMPUTER_SCIENCE`, `SOCIAL_COGNITIVE`) and mapped to 14 legitimate interdisciplinary fields:
- *Biophysics & Mechanobiology*
- *Systems Biology & Gene Regulatory Networks*
- *Stochastic Thermodynamics of Living Systems*
- *Active Matter & Microfluidics*
- *Information Theory in Biological Signalling*
- *Complex Network Science & Nonlinear Dynamics*
- *Bio-Imaging, Optical Diffraction & Inverse Problems*
- *Topological Data Analysis & Differential Geometry*
- *Synthetic Biology & Morphogenetic Engineering*
- *Non-Equilibrium Thermodynamics & Statistical Physics*
- *Quantum Information & Metrology*
- *Metamaterials & Nanophotonics*
- *Evolutionary Game Dynamics & Population Ecology*
- *Information Geometry & Statistical Manifolds*

Each pair is evaluated against the **Domain Affinity Matrix**. Unrelated combinations (*e.g., cell biology with metamaterials transformation optics*) are automatically penalized and suppressed.

### 2. Local Vector Intelligence via FastEmbed ONNX (Semantic Corridor Engine)
Science Expander integrates a local ONNX vector scoring layer powered by `fastembed` (`BAAI/bge-small-en-v1.5`):
- **100% Local Inference**: Runs directly on your CPU/GPU using ONNX Runtime with zero external API calls, token metering, or data leakage.
- **In-Memory Cache**: Seed and domain representations are cached for sub-millisecond inference ($< 1$ ms) after warm-up.
- **Pairwise Cosine Similarity**: Evaluates semantic alignment $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
- **Interdisciplinary Goldilocks Sweet Spot ($\mathcal{S}_{\text{Goldilocks}}$)**:
  - **Sweet Spot ($0.35 \le s \le 0.65$)**: Peak score ($\sim 1.0$) for fertile interdisciplinary ground where conceptual distance enables genuine scientific breakthroughs.
  - **Trivial Overlap ($s > 0.80$)**: Penalized score ($\le 0.35$) for semantic redundancy or disciplinary tautology.
  - **Conceptual Disconnect ($s < 0.20$)**: Penalized score ($\le 0.20$) for pseudo-scientific or incoherent pairings.
- **Resilient Fallback**: If ONNX weights are unavailable or memory is constrained, the system degrades seamlessly to pure rule-based ontology heuristics without crashing.

### 3. Tri-Axial Literature Retrieval (3D Academic Slice)
Instead of retrieving a single random paper, Science Expander pulls a 3-dimensional academic triad for any chosen topic:
- 🏛️ **The Foundation (Фундамент)**: Landmark paper with the highest historical citation impact.
- ⚡ **The Frontier (Фронтир, 2024–2026)**: Cutting-edge recent paper or preprint strictly sorted by `relevance_score:desc` to ensure tight semantic alignment.
- 📚 **The Review (Обзор)**: Comprehensive survey paper synthesizing the state of the art.

#### 3-Tier Failover Cascade
```
  [Google Scholar (scholarly)]
              │
              ▼ (on CAPTCHA / rate-limit / timeout)
     [OpenAlex REST API]
              │
              ▼ (on HTTP 429 budget cap)
     [Crossref Works API]
```
If an Open Access PDF is indexed, Science Expander extracts direct download URLs (`open_access.oa_url`) automatically.

### 4. Zero-Token Multilingual Bridge (RU ↔ EN)
- **Automatic Language Detection**: Detects Cyrillic characters via Unicode regex (`[\u0400-\u04FF]`).
- **Free Translation**: Powered by `deep-translator` with direct Google GTX endpoint failover (zero API keys, zero fees).
- **Bilingual Surface**: Input seeds are presented bilingually across the Rich TUI, Streamlit Web App, Research Briefs, and Knowledge Graph nodes (*e.g., `Клеточная биология (Cell Biology)`*).

### 5. Interactive Knowledge Topology (Vis.js Physics)
Generates standalone, self-contained HTML graph visualizations rendered via Vis.js:
- 🔷 **Neon Cyan Hexagons**: Core seed topics (Hubs).
- 🟣 **Purple Diamonds**: Interdisciplinary bridge domains.
- 🟢 **Emerald Dots**: Generated research directions with affinity badges.
- ⭐ **Gold/Amber Stars**: Literature triad publications dynamically linked to selected topics.
- 〰️ **Vector-Weighted Edges**: Edge thickness and hover tooltips mathematically reflect cosine similarities between nodes.

---

## 🚀 Quickstart Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/ScienceExpender.git
cd ScienceExpender

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🌐 Streamlit Web Application

Launch the modern, responsive web dashboard:

```bash
streamlit run app.py
```

The Web UI features:
- **Preset Demo Buttons**: One-click exploration for *Cell Biology*, *Neurobiology & Quantum Physics*, *Клеточная биология*, and *Topology & Genomics*.
- **Tab 1 — 10 Cross-Disciplinary Ideas**: Responsive cards with affinity badges ($\ge 95\%$ High, $80-94\%$ Med), methodological lenses, and target literature queries.
- **Tab 2 — Literature Triad**: Foundation, Frontier (2024–2026), and Review paper cards with direct **Download Open Access PDF** buttons.
- **Tab 3 — Interactive Knowledge Graph**: Embedded Vis.js force-directed physics graph with zoom, drag, and node-inspection controls.
- **Tab 4 — Export & BibTeX**: One-click download buttons for `Research_Brief.md`, `references.bib`, `knowledge_graph.html`, and `discovery_matrix.json`.

---

## 💻 Command-Line Interface (CLI)

Science Expander features a complete CLI interface with Rich formatting and pipeline integrations:

### 1. Single Topic Exploration
```bash
python science_expander.py -t "Cell Biology" --select 1
```

### 2. Russian Multilingual Seed with Interactive Graph & Report
```bash
python science_expander.py -t "Клеточная биология" --select 1 \
  --graph cell_biology_ru_graph.html \
  --report cell_biology_ru_brief.md \
  --bibtex references.bib
```

### 3. Multi-Topic Synthesis
```bash
python science_expander.py -t "Neurobiology, Quantum Physics" --auto
```

### 4. Automated Headless JSON Mode (for CI / Agentic Pipelines)
```bash
python science_expander.py -t "Genomics" --auto --json -o discovery.json
```

### CLI Options Reference
| Flag | Description | Default |
|---|---|---|
| `-t, --topics` | Comma-separated scientific topics (Russian or English) | *Interactive prompt* |
| `-c, --count` | Number of cross-disciplinary topics to synthesize | `10` |
| `--select N` | Automatically pick topic index `N` (1-10) and fetch literature | *Interactive prompt* |
| `--auto` | Non-interactive mode: selects top-ranked topic #1 | `False` |
| `--demo` | Run with pre-configured cutting-edge scientific topics | `False` |
| `--report [PATH]` | Export Markdown Research Brief | `Research_Brief.md` |
| `--graph [PATH]` | Export interactive Vis.js HTML Knowledge Graph | `knowledge_graph.html` |
| `--bibtex [PATH]` | Append citations to BibTeX file | `references.bib` |
| `--json` | Emit raw JSON payload | `False` |
| `-o, --output` | Save output to file | `stdout` |

---

## 📦 Generated Research Artifacts

Every run can emit 4 publication-ready deliverables:
1. **[`Research_Brief.md`](file:///Users/kirillshevchenko/ScienceExpender/cell_biology_ru_brief.md)**: Executive research dossier containing problem statements, methodological lenses, landscape tables, and the 3-axis publication triad.
2. **[`references.bib`](file:///Users/kirillshevchenko/ScienceExpender/references.bib)**: Standard BibTeX entries with DOIs, citations, and abstracts ready for Zotero/Mendeley/Overleaf.
3. **[`knowledge_graph.html`](file:///Users/kirillshevchenko/ScienceExpender/cell_biology_ru_graph.html)**: Standalone HTML file embedding the physics-based Vis.js network.
4. **`discovery_matrix.json`**: Structured JSON payload for automated downstream processing.

---

## 🧪 Automated Test Suite

Science Expander includes a comprehensive test suite covering ontology heuristics, domain affinity matrices, multilingual translation, academic API clients, CLI subprocessing, and web app components:

```bash
# Run the entire test suite
python -m unittest discover -s . -p "test_*.py"
```

All 32 tests execute in headless and CI environments with zero external credentials required.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

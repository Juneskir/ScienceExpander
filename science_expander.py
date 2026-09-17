#!/usr/bin/env python3
"""
Science Expander: Interdisciplinary Scientific Topic Cross-Breeder & Paper Discovery Tool.

Cross-breeds input scientific topics with interdisciplinary domains using a domain
affinity matrix and computational conceptual blending (without paid LLM tokens),
outputs 10 high-value research directions, and retrieves real literature via Google Scholar (scholarly).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import sys
import threading
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scholarly import scholarly
except ImportError as e:
    sys.stderr.write(
        "\n[ERROR] Failed to import 'scholarly'.\n"
        "Ensure requirements are installed: pip install -r requirements.txt\n"
        f"Details: {e}\n\n"
    )
    sys.exit(1)


# =====================================================================
# Rich Terminal UI & Terminal Styling Helpers
# =====================================================================
try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import box
    HAS_RICH = True
    rich_console = Console()
except ImportError:
    HAS_RICH = False
    rich_console = None

class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    # Colors
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    WHITE = "\033[37m"


def color(text: str, *styles: str) -> str:
    """Apply terminal styles to text if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    prefix = "".join(styles)
    return f"{prefix}{text}{Style.RESET}"


# =====================================================================
# Zero-Token Multilingual Bridge (RU ↔ EN Translation Layer)
# =====================================================================
class MultilingualBridge:
    """Zero-token multilingual translation layer for scientific topics."""
    _cache: Dict[str, str] = {}
    _DISK_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "science_expander_translations.json")

    _LEXICON: Dict[str, str] = {
        "клеточная биология": "Cell Biology",
        "биология клетки": "Cell Biology",
        "квантовая физика": "Quantum Physics",
        "квантовая оптика": "Quantum Optics",
        "нейробиология": "Neurobiology",
        "молекулярная биология": "Molecular Biology",
        "генетика": "Genetics",
        "редактирование генома": "Genome Editing",
        "биоинформатика": "Bioinformatics",
        "биофизика": "Biophysics",
        "термодинамика": "Thermodynamics",
        "статистическая физика": "Statistical Physics",
        "астрофизика": "Astrophysics",
        "экология": "Ecology",
        "микробиология": "Microbiology",
        "машинное обучение": "Machine Learning",
        "глубокое обучение": "Deep Learning",
        "теория графов": "Graph Theory",
        "топологический анализ данных": "Topological Data Analysis",
        "дифференциальная геометрия": "Differential Geometry",
        "стохастическая термодинамика": "Stochastic Thermodynamics",
        "нанотехнологии": "Nanotechnology",
        "синтетическая биология": "Synthetic Biology",
        "теория информации": "Information Theory",
        "сложные сети": "Complex Networks",
        "иммунология": "Immunology",
        "онкология": "Oncology",
        "эпигенетика": "Epigenetics",
        "протеомика": "Proteomics",
        "геномика": "Genomics",
        "повреждение днк": "DNA Damage",
        "ответ на повреждение днк": "DNA Damage Response",
        "репарация днк": "DNA Repair",
        "сверхпроводимость": "Superconductivity",
    }

    @classmethod
    def _load_disk_cache(cls) -> None:
        if not cls._cache and os.path.exists(cls._DISK_CACHE_PATH):
            try:
                with open(cls._DISK_CACHE_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cls._cache.update(loaded)
            except Exception:
                pass

    @classmethod
    def _save_disk_cache(cls) -> None:
        try:
            os.makedirs(os.path.dirname(cls._DISK_CACHE_PATH), exist_ok=True)
            with open(cls._DISK_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cls._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def is_cyrillic(cls, text: str) -> bool:
        """Detect if text contains Cyrillic characters."""
        return bool(re.search(r'[а-яА-ЯёЁ]', text))

    @classmethod
    def translate_to_en(cls, text: str) -> str:
        """Translate scientific topic to academic English with caching and resilient fallback."""
        cleaned = text.strip()
        if not cleaned or not cls.is_cyrillic(cleaned):
            return cleaned

        # 1. Check curated scientific lexicon (0ms latency, 100% offline resilient)
        cleaned_lower = cleaned.lower()
        if cleaned_lower in cls._LEXICON:
            return cls._LEXICON[cleaned_lower]

        # 2. Check in-memory & disk cache
        cls._load_disk_cache()
        if cleaned in cls._cache:
            return cls._cache[cleaned]

        # 3. Fast Google GTX endpoint
        try:
            q = urllib.parse.quote(cleaned)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={q}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ScienceExpander/2.0)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and data[0] and data[0][0] and data[0][0][0]:
                    translated = data[0][0][0].strip().title()
                    cls._cache[cleaned] = translated
                    cls._save_disk_cache()
                    return translated
        except Exception:
            pass

        # 4. Fallback to deep-translator GoogleTranslator
        try:
            from deep_translator import GoogleTranslator
            res = GoogleTranslator(source='auto', target='en').translate(cleaned)
            if res and res.strip():
                translated = res.strip().title()
                cls._cache[cleaned] = translated
                cls._save_disk_cache()
                return translated
        except Exception:
            pass

        # 5. Graceful fallback: return original text
        return cleaned

    @classmethod
    def format_bilingual(cls, raw: str, en: str) -> str:
        """Format bilingual string: e.g. 'Клеточная биология (Cell Biology)'."""
        if cls.is_cyrillic(raw) and raw.strip().lower() != en.strip().lower():
            return f"{raw.strip()} ({en.strip()})"
        return en.strip()


# =====================================================================
# Scientific Taxonomy & Domain Affinity System
# =====================================================================
class ScienceCategory(str, Enum):
    BIOLOGY = "biology"
    PHYSICS = "physics"
    MATERIALS = "materials"
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    SOCIAL_COGNITIVE = "social_cognitive"
    GENERAL_SCIENCE = "general_science"


CATEGORY_LEXICON: Dict[ScienceCategory, Set[str]] = {
    ScienceCategory.BIOLOGY: {
        "cell", "cellular", "gene", "genes", "genetic", "genetics", "dna", "rna", "protein",
        "proteins", "crispr", "biology", "biological", "biophysics", "tissue", "immune",
        "immunology", "organ", "organoid", "neuron", "neural", "neuroscience", "cancer",
        "oncology", "tumor", "genome", "genomics", "metabolic", "metabolism", "enzyme",
        "enzymes", "molecular", "plant", "botany", "microbiology", "microbiome", "bacteria",
        "bacterial", "viral", "virology", "virus", "organism", "ecology", "ecological",
        "epigenetics", "epigenome", "developmental", "evolution", "evolutionary", "morphogenesis",
        "biochemistry", "pathology", "synapse", "synaptic", "biomedical"
    },
    ScienceCategory.PHYSICS: {
        "quantum", "qubit", "optic", "optics", "optical", "photon", "photonic", "photonics",
        "thermal", "thermodynamic", "thermodynamics", "superconduct", "superconductivity",
        "superconductor", "particle", "particles", "gravitational", "gravity", "magnetic",
        "magnetism", "fluid", "fluids", "hydrodynamics", "acoustic", "acoustics", "plasma",
        "cosmology", "astrophysics", "astronomy", "electrodynamics", "condensed matter",
        "statistical mechanics", "relativity", "laser", "spectroscopy", "nuclear", "field theory",
        "mechanics", "semiconductor", "soliton", "entropy", "scattering"
    },
    ScienceCategory.MATERIALS: {
        "material", "materials", "metamaterial", "metamaterials", "graphene", "polymer",
        "polymers", "nanomaterial", "nanomaterials", "nanotechnology", "alloy", "alloys",
        "crystal", "crystals", "crystallography", "ceramic", "ceramics", "perovskite",
        "perovskites", "composite", "composites", "hydrogel", "hydrogels", "membrane",
        "membranes", "nanotube", "nanotubes", "colloid", "colloids", "thin film", "photovoltaic"
    },
    ScienceCategory.MATHEMATICS: {
        "topology", "topological", "geometry", "geometric", "manifold", "manifolds", "algebraic",
        "differential", "dynamical", "dynamics", "stochastic", "graph", "graphs", "combinatorics",
        "probability", "ergodic", "chaos", "number theory", "pde", "optimization", "spectral",
        "homology", "homotopy", "sheaf", "invariance", "curvature"
    },
    ScienceCategory.COMPUTER_SCIENCE: {
        "algorithm", "algorithms", "algorithmic", "computational", "computing", "cryptography",
        "distributed", "software", "complexity", "information theory", "robotics", "database",
        "data science", "compiler", "hardware", "automata", "consensus", "cryptographic",
        "machine learning", "artificial intelligence"
    },
    ScienceCategory.SOCIAL_COGNITIVE: {
        "cognition", "cognitive", "behavior", "behavioral", "economic", "economics",
        "game theory", "psychology", "social", "decision", "linguistics", "governance",
        "cooperation", "population", "sociology", "incentive", "welfare"
    }
}


def classify_topic(topic: str) -> ScienceCategory:
    """Classify a scientific topic string into its primary science category."""
    if MultilingualBridge.is_cyrillic(topic) and not re.search(r"[a-zA-Z]", topic):
        en_topic = MultilingualBridge.translate_to_en(topic)
        topic = f"{topic} ({en_topic})"

    tokens = re.findall(r"[a-z0-9\-]+", topic.lower())
    if not tokens:
        return ScienceCategory.GENERAL_SCIENCE

    scores: Dict[ScienceCategory, int] = {cat: 0 for cat in ScienceCategory if cat != ScienceCategory.GENERAL_SCIENCE}

    for token in tokens:
        for cat, lexicon in CATEGORY_LEXICON.items():
            if token in lexicon:
                scores[cat] += 3
            else:
                for word in lexicon:
                    if len(token) > 4 and (token.startswith(word) or word.startswith(token)):
                        scores[cat] += 1

    best_cat, best_score = max(scores.items(), key=lambda item: item[1])
    return best_cat if best_score > 0 else ScienceCategory.GENERAL_SCIENCE


def format_topic_for_title(topic: str) -> str:
    """Format scientific topic for an academic paper title, preserving uppercase acronyms."""
    words = topic.split()
    formatted = []
    for w in words:
        if (w.isupper() and len(w) > 1) or any(c.isupper() for c in w[1:]):
            formatted.append(w)
        else:
            formatted.append(w.capitalize())
    return " ".join(formatted)


# =====================================================================
# Interdisciplinary Knowledge Base & Ontology
# =====================================================================
@dataclass(frozen=True)
class InterdisciplinaryDomain:
    name: str
    category_label: str
    primary_category: ScienceCategory
    affinities: Dict[ScienceCategory, float]
    methods: Tuple[str, ...]
    phenomena: Tuple[str, ...]
    concepts: Tuple[str, ...]
    properties: Tuple[str, ...]
    keywords: Tuple[str, ...]


DOMAINS: Tuple[InterdisciplinaryDomain, ...] = (
    InterdisciplinaryDomain(
        name="Biophysics & Mechanobiology",
        category_label="Physics & Biological Physics",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.MATERIALS: 0.9,
            ScienceCategory.PHYSICS: 0.85,
            ScienceCategory.MATHEMATICS: 0.7,
            ScienceCategory.COMPUTER_SCIENCE: 0.4,
            ScienceCategory.SOCIAL_COGNITIVE: 0.1,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Traction Force Microscopy",
            "Atomic Force Spectroscopy",
            "Viscoelastic Rheology Profiling",
            "Optical Tweezers Manipulation",
        ),
        phenomena=(
            "Mechanotransductive Gating",
            "Cytoskeletal Tension Remodeling",
            "Strain-Induced Fluidization",
            "Membrane Curvature Sorting",
        ),
        concepts=(
            "Mechanosensitive Ion Channels",
            "Cellular Viscoelasticity",
            "Extracellular Matrix Stiffness Coupling",
            "Force-Dependent Molecular Conformations",
        ),
        properties=(
            "Mechanosensory Sensitivity",
            "Structural Elasticity",
            "Tensional Homeostasis",
            "Deformation Resilience",
        ),
        keywords=("mechanobiology", "cellular biophysics", "mechanotransduction"),
    ),
    InterdisciplinaryDomain(
        name="Systems Biology & Gene Regulatory Networks",
        category_label="Computational & Quantitative Biology",
        primary_category=ScienceCategory.BIOLOGY,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.MATHEMATICS: 0.9,
            ScienceCategory.COMPUTER_SCIENCE: 0.85,
            ScienceCategory.PHYSICS: 0.75,
            ScienceCategory.MATERIALS: 0.3,
            ScienceCategory.SOCIAL_COGNITIVE: 0.3,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Bifurcation and Phase Plane Analysis",
            "Sensitivity and Metabolic Flux Analysis",
            "Dynamic Network Motif Profiling",
            "Chemical Master Equation Formulations",
        ),
        phenomena=(
            "Bistable State Switching",
            "Limit Cycle Oscillations",
            "Noise-Induced State Transitions",
            "Transcriptional Hysteresis",
        ),
        concepts=(
            "Feedback Control Motifs",
            "Quorum-Driven Attractors",
            "Epigenetic Waddington Landscapes",
            "Oscillatory Gene Circuits",
        ),
        properties=(
            "Attractor Stability",
            "Bistable Switchability",
            "Noise Buffering",
            "Dynamical Homeostasis",
        ),
        keywords=("systems biology", "gene regulatory networks", "dynamical systems biology"),
    ),
    InterdisciplinaryDomain(
        name="Stochastic Thermodynamics of Living Systems",
        category_label="Non-Equilibrium Physics & Biophysics",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.PHYSICS: 1.0,
            ScienceCategory.MATHEMATICS: 0.85,
            ScienceCategory.MATERIALS: 0.6,
            ScienceCategory.COMPUTER_SCIENCE: 0.5,
            ScienceCategory.SOCIAL_COGNITIVE: 0.2,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Thermodynamic Uncertainty Relations (TUR)",
            "Fluctuation Theorem Decompositions",
            "Entropy Production Rate Quantification",
            "Stochastic Energetics Formulations",
        ),
        phenomena=(
            "Non-Equilibrium Steady State Dissipation",
            "Energy-Accuracy Trade-Offs",
            "Free-Energy Driven Kinetic Proofreading",
            "Microscopic Reversibility Breaking",
        ),
        concepts=(
            "Thermodynamic Cost of Signalling",
            "Dissipative Proofreading Cascades",
            "Nonequilibrium Fluctuation Landscapes",
            "Thermodynamic Length and Free Energy Bounds",
        ),
        properties=(
            "Thermodynamic Efficiency",
            "Kinetic Proofreading Fidelity",
            "Entropy Dissipation Minimization",
            "Fluctuation Robustness",
        ),
        keywords=("stochastic thermodynamics", "nonequilibrium energetics", "thermodynamic uncertainty relation"),
    ),
    InterdisciplinaryDomain(
        name="Active Matter & Microfluidics",
        category_label="Soft Matter Physics & Fluid Mechanics",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.BIOLOGY: 0.95,
            ScienceCategory.PHYSICS: 1.0,
            ScienceCategory.MATERIALS: 0.95,
            ScienceCategory.MATHEMATICS: 0.75,
            ScienceCategory.COMPUTER_SCIENCE: 0.4,
            ScienceCategory.SOCIAL_COGNITIVE: 0.2,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Microfluidic Droplet Confinement",
            "Hydrodynamic Flow Field Velocimetry",
            "Continuum Active Gel Hydrodynamics",
            "Optical Vorticity Trapping",
        ),
        phenomena=(
            "Motility-Induced Phase Separation (MIPS)",
            "Collective Bacterial Swarming",
            "Active Cytoplasmic Streaming",
            "Topological Defect Motion",
        ),
        concepts=(
            "Self-Propelled Colloidal Engines",
            "Active Stress Tensors",
            "Microfluidic Compartmentalization",
            "Non-Newtonian Rheology Matrices",
        ),
        properties=(
            "Non-Equilibrium Self-Assembly",
            "Coordinated Swarming",
            "Shear-Induced Transport",
            "Directed Flow Invariance",
        ),
        keywords=("active matter", "microfluidics", "cytoplasmic streaming"),
    ),
    InterdisciplinaryDomain(
        name="Information Theory in Biological Signalling",
        category_label="Information Theory & Quantitative Biology",
        primary_category=ScienceCategory.COMPUTER_SCIENCE,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.COMPUTER_SCIENCE: 0.9,
            ScienceCategory.MATHEMATICS: 0.9,
            ScienceCategory.PHYSICS: 0.8,
            ScienceCategory.SOCIAL_COGNITIVE: 0.4,
            ScienceCategory.MATERIALS: 0.2,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Channel Capacity Quantification",
            "Mutual Information Estimation in Single Cells",
            "Rate-Distortion Analysis of Receptors",
            "Transfer Entropy Mapping",
        ),
        phenomena=(
            "Information Bottlenecks in Transduction",
            "Noise Filtering via Multi-Step Relays",
            "Stochastic Receptor Gating Limits",
            "Signal Distortion Thresholds",
        ),
        concepts=(
            "Input-Output Channel Capacity",
            "Biochemical Signal-to-Noise Bounds",
            "Redundant Sensory Encoding",
            "Optimal Information Transmission",
        ),
        properties=(
            "Signalling Fidelity",
            "Information Conservation",
            "Noise Suppression",
            "Transduction Bandwidth",
        ),
        keywords=("information theory in biology", "biological signalling channel capacity", "information transduction"),
    ),
    InterdisciplinaryDomain(
        name="Complex Network Science & Nonlinear Dynamics",
        category_label="Mathematics & Complex Systems",
        primary_category=ScienceCategory.MATHEMATICS,
        affinities={
            ScienceCategory.BIOLOGY: 0.95,
            ScienceCategory.PHYSICS: 0.95,
            ScienceCategory.COMPUTER_SCIENCE: 0.95,
            ScienceCategory.MATHEMATICS: 1.0,
            ScienceCategory.SOCIAL_COGNITIVE: 0.95,
            ScienceCategory.MATERIALS: 0.6,
            ScienceCategory.GENERAL_SCIENCE: 0.9,
        },
        methods=(
            "Spectral Graph Decomposition",
            "Percolation and Robustness Analysis",
            "Nonlinear Time Series Embedding",
            "Community Detection Algorithms",
        ),
        phenomena=(
            "Critical Slowing Down at Tipping Points",
            "Cascade Breakdown Dynamics",
            "Phase Synchronization",
            "Self-Organized Criticality",
        ),
        concepts=(
            "Scale-Free Network Architecture",
            "Dynamical Centrality Metrics",
            "Resilience Landscapes",
            "Coupled Nonlinear Oscillators",
        ),
        properties=(
            "Percolation Resilience",
            "Network Controllability",
            "Synchronizability",
            "Fault Tolerance",
        ),
        keywords=("complex networks", "critical transitions", "nonlinear dynamics"),
    ),
    InterdisciplinaryDomain(
        name="Bio-Imaging, Optical Diffraction & Inverse Problems",
        category_label="Optics & Applied Physics",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.PHYSICS: 0.95,
            ScienceCategory.MATERIALS: 0.85,
            ScienceCategory.COMPUTER_SCIENCE: 0.75,
            ScienceCategory.MATHEMATICS: 0.75,
            ScienceCategory.SOCIAL_COGNITIVE: 0.05,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Super-Resolution Structured Illumination (SIM)",
            "Phase Retrieval and Deconvolution",
            "Cryo-Electron Tomography Reconstruction",
            "Adaptive Optics Wavefront Correction",
        ),
        phenomena=(
            "Sub-Diffraction Waveguide Localization",
            "Optical Phase Singularity",
            "Scattering-Induced Speckle Correlation",
            "Evanescent Wave Coupling",
        ),
        concepts=(
            "Inverse Scattering Algorithms",
            "Single-Molecule Localization Precision",
            "Wavefront Engineering",
            "High-Numerical-Aperture Focal Fields",
        ),
        properties=(
            "Spatial Resolution Fidelity",
            "High-Contrast Resolvability",
            "Non-Invasive Penetration",
            "Dynamic Phase Sensitivity",
        ),
        keywords=("super-resolution imaging", "cryo-electron tomography", "inverse problems in optics"),
    ),
    InterdisciplinaryDomain(
        name="Topological Data Analysis & Differential Geometry",
        category_label="Mathematics & Theoretical Physics",
        primary_category=ScienceCategory.MATHEMATICS,
        affinities={
            ScienceCategory.MATHEMATICS: 1.0,
            ScienceCategory.PHYSICS: 0.95,
            ScienceCategory.COMPUTER_SCIENCE: 0.95,
            ScienceCategory.MATERIALS: 0.85,
            ScienceCategory.BIOLOGY: 0.85,
            ScienceCategory.SOCIAL_COGNITIVE: 0.5,
            ScienceCategory.GENERAL_SCIENCE: 0.85,
        },
        methods=(
            "Persistent Homology and Filtration",
            "Manifold Learning and Geodesic Mapping",
            "Euler Characteristic Transforms",
            "Sheaf-Theoretic Data Integration",
        ),
        phenomena=(
            "Geometric Phase Invariance",
            "Topological Boundary Preservation",
            "Betti Number Transitions",
            "Curvature-Induced Trapping",
        ),
        concepts=(
            "Invariant Topological Signatures",
            "High-Dimensional Phenotypic Manifolds",
            "Simplicial Complex Ensembles",
            "Persistence Landscapes",
        ),
        properties=(
            "Topological Invariance",
            "Coordinate-Free Robustness",
            "Manifold Fidelity",
            "Geometric Continuity",
        ),
        keywords=("topological data analysis", "persistent homology", "manifold learning"),
    ),
    InterdisciplinaryDomain(
        name="Synthetic Biology & Morphogenetic Engineering",
        category_label="Bioengineering & Developmental Biology",
        primary_category=ScienceCategory.BIOLOGY,
        affinities={
            ScienceCategory.BIOLOGY: 1.0,
            ScienceCategory.MATERIALS: 0.85,
            ScienceCategory.COMPUTER_SCIENCE: 0.75,
            ScienceCategory.PHYSICS: 0.75,
            ScienceCategory.MATHEMATICS: 0.6,
            ScienceCategory.SOCIAL_COGNITIVE: 0.2,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "CRISPR-Guided Epigenetic Circuit Design",
            "Bio-Orthogonal Metabolic Flux Redirection",
            "Turing Pattern Reaction-Diffusion Modeling",
            "Cell-Free Biochemical System Assembly",
        ),
        phenomena=(
            "Membraneless Condensation and Phase Separation",
            "Quorum Sensing Coherence",
            "Morphogen Gradient Formation",
            "Self-Assembling Morphogenesis",
        ),
        concepts=(
            "Synthetic Gene Logic Gates",
            "Epigenetic Memory Switches",
            "Living Biomaterial Scaffolds",
            "Cellular Allostasis Relays",
        ),
        properties=(
            "Biocompatibility",
            "Adaptive Homeostasis",
            "Programmable Morphogenesis",
            "Self-Healing Autonomy",
        ),
        keywords=("synthetic biology", "morphogenesis", "metabolic engineering"),
    ),
    InterdisciplinaryDomain(
        name="Non-Equilibrium Thermodynamics & Statistical Physics",
        category_label="Theoretical & Statistical Physics",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.PHYSICS: 1.0,
            ScienceCategory.MATERIALS: 0.95,
            ScienceCategory.BIOLOGY: 0.85,
            ScienceCategory.MATHEMATICS: 0.8,
            ScienceCategory.COMPUTER_SCIENCE: 0.5,
            ScienceCategory.SOCIAL_COGNITIVE: 0.3,
            ScienceCategory.GENERAL_SCIENCE: 0.85,
        },
        methods=(
            "Fluctuation-Dissipation Decomposition",
            "Entropy Production Rate Minimization",
            "Master Equation Formulations",
            "Langevin Dynamics Modeling",
        ),
        phenomena=(
            "Spontaneous Symmetry Breaking",
            "Dissipative Self-Organization",
            "Phase Separation Dynamics",
            "Bifurcation Under Flow",
        ),
        concepts=(
            "Non-Equilibrium Steady States",
            "Free-Energy Landscapes",
            "Thermodynamic Forces and Fluxes",
            "Kinetic Arrest Transitions",
        ),
        properties=(
            "Dissipative Efficiency",
            "Kinetic Stability",
            "Fluctuation Tolerance",
            "Non-Equilibrium Homeostasis",
        ),
        keywords=("non-equilibrium thermodynamics", "dissipative structures", "fluctuation theorems"),
    ),
    InterdisciplinaryDomain(
        name="Quantum Information & Metrology",
        category_label="Quantum Physics & Information Science",
        primary_category=ScienceCategory.PHYSICS,
        affinities={
            ScienceCategory.PHYSICS: 1.0,
            ScienceCategory.COMPUTER_SCIENCE: 0.95,
            ScienceCategory.MATERIALS: 0.9,
            ScienceCategory.MATHEMATICS: 0.85,
            ScienceCategory.BIOLOGY: 0.25,
            ScienceCategory.SOCIAL_COGNITIVE: 0.05,
            ScienceCategory.GENERAL_SCIENCE: 0.7,
        },
        methods=(
            "Tensor Network Decompositions",
            "Quantum-Enhanced Interferometric Sensing",
            "Entanglement Witnessing and Tomography",
            "Decoherence-Free Subspace Encoding",
        ),
        phenomena=(
            "Quantum Coherence and Tunneling",
            "Entanglement Swapping",
            "Superradiance and Collective Emission",
            "Non-Locality in Condensed Phases",
        ),
        concepts=(
            "Quantum Sensing Protocols",
            "Squeezed-State Precision",
            "Entanglement Entropy Scaling",
            "Coherent State Control",
        ),
        properties=(
            "Sub-Shot-Noise Precision",
            "Quantum Coherence Longevity",
            "Sensing Sensitivity",
            "Information-Theoretic Security",
        ),
        keywords=("quantum sensing", "quantum metrology", "quantum coherence"),
    ),
    InterdisciplinaryDomain(
        name="Metamaterials & Nanophotonics",
        category_label="Materials Science & Applied Physics",
        primary_category=ScienceCategory.MATERIALS,
        affinities={
            ScienceCategory.MATERIALS: 1.0,
            ScienceCategory.PHYSICS: 1.0,
            ScienceCategory.COMPUTER_SCIENCE: 0.5,
            ScienceCategory.MATHEMATICS: 0.6,
            ScienceCategory.BIOLOGY: 0.25,
            ScienceCategory.SOCIAL_COGNITIVE: 0.05,
            ScienceCategory.GENERAL_SCIENCE: 0.7,
        },
        methods=(
            "Transformation Optics Inversion",
            "Subwavelength Plasmonic Near-Field Imaging",
            "Inverse Photonic Bandgap Design",
            "Chiral Optical Metasurface Engineering",
        ),
        phenomena=(
            "Negative Refractive Index Propagation",
            "Fano Resonance Coupling",
            "Topological Edge State Transport",
            "Localized Surface Plasmon Resonance",
        ),
        concepts=(
            "Hyperbolic Metamaterial Cavities",
            "Phase-Change Nanophotonic Scaffolds",
            "Sub-Diffraction Resonators",
            "Zero-Index Optical Matrices",
        ),
        properties=(
            "Wavefront Manipulation",
            "Sub-Diffraction Localization",
            "Broadband Absorption",
            "Polarization Control",
        ),
        keywords=("metamaterials", "nanophotonics", "plasmonic resonance"),
    ),
    InterdisciplinaryDomain(
        name="Evolutionary Game Dynamics & Population Ecology",
        category_label="Applied Mathematics & Evolutionary Biology",
        primary_category=ScienceCategory.SOCIAL_COGNITIVE,
        affinities={
            ScienceCategory.BIOLOGY: 0.9,
            ScienceCategory.SOCIAL_COGNITIVE: 1.0,
            ScienceCategory.MATHEMATICS: 0.9,
            ScienceCategory.COMPUTER_SCIENCE: 0.8,
            ScienceCategory.PHYSICS: 0.6,
            ScienceCategory.MATERIALS: 0.1,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Replicator Dynamics Formulations",
            "Evolutionary Stable Strategy (ESS) Analysis",
            "Spatial Population Lattice Modeling",
            "Adaptive Dynamics Frameworks",
        ),
        phenomena=(
            "Cooperative Trait Stabilization",
            "Frequency-Dependent Selection Transitions",
            "Tragedy of the Commons Avoidance",
            "Eco-Evolutionary Feedback Loops",
        ),
        concepts=(
            "Public Goods Dynamics in Microorganisms",
            "Stochastic Moran Processes",
            "Spatial Clustered Cooperation",
            "Resource Partitioning Games",
        ),
        properties=(
            "Evolutionary Stability",
            "Collective Fitness Optimization",
            "Cheater Resistance",
            "Population Persistence",
        ),
        keywords=("evolutionary game theory", "replicator dynamics", "population dynamics"),
    ),
    InterdisciplinaryDomain(
        name="Information Geometry & Statistical Manifolds",
        category_label="Differential Geometry & Statistics",
        primary_category=ScienceCategory.MATHEMATICS,
        affinities={
            ScienceCategory.MATHEMATICS: 1.0,
            ScienceCategory.COMPUTER_SCIENCE: 0.95,
            ScienceCategory.PHYSICS: 0.85,
            ScienceCategory.SOCIAL_COGNITIVE: 0.7,
            ScienceCategory.BIOLOGY: 0.55,
            ScienceCategory.MATERIALS: 0.4,
            ScienceCategory.GENERAL_SCIENCE: 0.8,
        },
        methods=(
            "Fisher Information Metric Decompositions",
            "Kullback-Leibler Divergence Geometry",
            "Natural Gradient Formulations",
            "Differential Geometric Statistics",
        ),
        phenomena=(
            "Curvature-Driven Estimation Drift",
            "Information Geometric Phase Transitions",
            "Riemannian Metric Contraction",
            "Geodesic Asymptotics",
        ),
        concepts=(
            "Statistical Manifolds and Dual Connections",
            "Fisher Information Riemannian Metrics",
            "Information Monotonicity",
            "Alpha-Connections",
        ),
        properties=(
            "Estimation Invariance",
            "Sample Complexity Bounds",
            "Geometric Efficiency",
            "Cramer-Rao Optimality",
        ),
        keywords=("information geometry", "Fisher information metric", "statistical manifold"),
    ),
)


# =====================================================================
# Conceptual Blending Models & Generator
# =====================================================================
@dataclass
class GeneratedTopic:
    index: int
    title: str
    primary_topic: str
    secondary_topic: Optional[str]
    domain: InterdisciplinaryDomain
    operator_name: str
    rationale: str
    affinity: float
    primary_query: str
    fallback_query: str
    display_primary_topic: Optional[str] = None
    display_secondary_topic: Optional[str] = None
    cosine_similarity: Optional[float] = None
    goldilocks_score: Optional[float] = None
    vector_zone: Optional[str] = None
    anchor_phrase: Optional[str] = None
    anchor_terms: Optional[List[str]] = None


def clean_topic_name(topic: str) -> str:
    """Normalize input topic strings for grammatical and natural blend insertion."""
    t = topic.strip()
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = t.strip("\"' ")
    return t


def extract_core_keywords(text: str) -> List[str]:
    """Extract 1-3 salient search terms from a topic phrase."""
    stop_words = {
        "a", "an", "the", "and", "or", "of", "in", "for", "on", "with",
        "to", "at", "by", "from", "via", "using", "into", "through",
        "novel", "study", "analysis", "framework", "approach", "principles",
        "investigating", "resolving", "unlocking", "advances", "perspective"
    }
    words = re.findall(r"[A-Za-z0-9\-]+", text)
    filtered = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return filtered[:3] if filtered else words[:2]


def extract_anchor_phrase_and_terms(topic: str) -> Tuple[str, List[str]]:
    """
    Extract immutable anchor phrase and core keywords from a scientific topic.
    For example:
      'DNA damage response' -> ('DNA damage', ['dna', 'damage', 'response'])
      'Cell Biology' -> ('Cell Biology', ['cell', 'biology'])
      'CRISPR-Cas9 gene editing' -> ('CRISPR-Cas9', ['crispr', 'cas9', 'gene', 'editing'])
    """
    cleaned = clean_topic_name(topic)
    m = re.match(r"^(.+?)\s*\((.+?)\)$", cleaned)
    primary_part = m.group(2).strip() if m else cleaned

    words = re.findall(r"[A-Za-z0-9\-]+", primary_part)
    if not words:
        words = [primary_part]

    generic_suffixes = {
        "response", "responses", "signaling", "signalling", "pathway", "pathways",
        "mechanism", "mechanisms", "analysis", "system", "systems", "dynamics",
        "study", "studies", "perspective", "perspectives", "principles", "approach"
    }

    if len(words) > 1 and words[-1].lower() in generic_suffixes:
        anchor_words = words[:-1]
    else:
        anchor_words = words

    anchor_phrase = " ".join(anchor_words)

    stop_words = {"a", "an", "the", "in", "of", "on", "for", "with", "to", "at", "by", "via", "and", "or"}
    anchor_terms: List[str] = []
    for w in words:
        w_low = w.lower().strip("-")
        if w_low not in stop_words and (len(w_low) >= 3 or w_low in {"dna", "rna", "sim", "tur", "tda", "qkd"}):
            if w_low not in anchor_terms:
                anchor_terms.append(w_low)
        if "-" in w_low:
            for subpart in w_low.split("-"):
                sub_clean = subpart.strip()
                if sub_clean not in stop_words and (len(sub_clean) >= 3 or sub_clean in {"dna", "rna", "sim", "tur", "tda", "qkd"}):
                    if sub_clean not in anchor_terms:
                        anchor_terms.append(sub_clean)

    if m:
        cyr_words = re.findall(r"[а-яА-ЯёЁ0-9\-]+", m.group(1).strip())
        for cw in cyr_words:
            cw_low = cw.lower().strip("-")
            if len(cw_low) >= 3 and cw_low not in anchor_terms:
                anchor_terms.append(cw_low)

    return anchor_phrase, anchor_terms


def check_anchor_gate(paper: Any, anchor_terms: List[str]) -> bool:
    """
    Verify if paper contains the core immutable anchor keywords in its title or abstract.
    For topics with prominent markers like 'dna', 'rna', 'crispr', 'quantum', 'cell', 'neural',
    the paper MUST contain relevant domain anchor terminology.
    """
    if not anchor_terms:
        return True

    title_lower = getattr(paper, "title", "").lower()
    abstract_lower = (getattr(paper, "abstract", "") or "").lower()
    full_text = f"{title_lower} {abstract_lower}"

    # Strict biological / chemical checks for key acronyms
    if "dna" in anchor_terms:
        return bool(re.search(r'\bdna\b', full_text))
    if "rna" in anchor_terms:
        return bool(re.search(r'\brna\b', full_text))
    if "crispr" in anchor_terms:
        return bool(re.search(r'\bcrispr\b', full_text))

    # General check: at least one substantive anchor term or morphological variant must appear
    for term in anchor_terms:
        if len(term) >= 3:
            term_l = term.lower()
            if term_l == "cell":
                if re.search(r'\b(cells?|cellular)\b', full_text):
                    return True
            elif term_l == "gene":
                if re.search(r'\b(genes?|genetics?)\b', full_text):
                    return True
            elif term_l in {"neuron", "neural"}:
                if re.search(r'\b(neurons?|neuronal|neural)\b', full_text):
                    return True
            elif term_l in {"molecule", "molecular"}:
                if re.search(r'\b(molecules?|molecular)\b', full_text):
                    return True
            elif term_l.endswith("y"):
                stem = term_l[:-1]
                if re.search(rf'\b{re.escape(stem)}(y|ies|ic|ical|ist|ists)?\b', full_text):
                    return True
            else:
                pattern = rf'\b{re.escape(term_l)}(s|es|ed|ing|ic|ical|ar|ular)?\b'
                if re.search(pattern, full_text):
                    return True

    return False


# =====================================================================
# Local Vector Intelligence via FastEmbed ONNX (Semantic Corridor Engine)
# =====================================================================
class SemanticCorridorEngine:
    """
    Autonomous, local vector scoring engine utilizing FastEmbed ONNX.
    Computes pairwise semantic cosine similarities and evaluates candidate topic blends
    using the Interdisciplinary Goldilocks Sweet Spot scoring curve.
    """
    _instance: Optional["SemanticCorridorEngine"] = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.is_available = False
        self._cache: Dict[str, Any] = {}
        self._init_engine()

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-small-en-v1.5") -> "SemanticCorridorEngine":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _init_engine(self) -> None:
        """Initialize ONNX TextEmbedding model with graceful fallback."""
        if np is None:
            self.is_available = False
            return
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=self.model_name)
            self.is_available = True
        except Exception:
            self.model = None
            self.is_available = False

    def embed(self, text: str) -> Optional[Any]:
        """Compute normalized vector embedding for input text with caching (< 10ms)."""
        if not self.is_available or self.model is None or np is None:
            return None
        cleaned = text.strip().lower()
        if not cleaned:
            return None
        if cleaned in self._cache:
            return self._cache[cleaned]
        try:
            generator = self.model.embed([cleaned])
            vec = next(generator)
            arr = np.array(vec, dtype=np.float32)
            self._cache[cleaned] = arr
            return arr
        except Exception:
            return None

    @staticmethod
    def cosine_similarity(u: Optional[Any], v: Optional[Any]) -> float:
        """Calculate pairwise cosine similarity: cos(u, v) = (u . v) / (||u|| ||v||)."""
        if u is None or v is None or np is None:
            return 0.0
        try:
            norm_u = float(np.linalg.norm(u))
            norm_v = float(np.linalg.norm(v))
            if norm_u == 0.0 or norm_v == 0.0:
                return 0.0
            return float(np.dot(u, v) / (norm_u * norm_v))
        except Exception:
            return 0.0

    @staticmethod
    def calculate_goldilocks_score(similarity: float) -> float:
        """
        Interdisciplinary Goldilocks Sweet Spot Function:
        - Sweet spot [0.35, 0.65]: Peak score (~1.0)
        - Penalize trivial overlap (s > 0.80, lack of novelty)
        - Penalize extreme disconnect (s < 0.20, conceptual nonsense)
        """
        s = max(-1.0, min(1.0, similarity))
        if 0.40 <= s <= 0.60:
            return 1.0
        elif 0.35 <= s < 0.40:
            return round(0.90 + (s - 0.35) * (0.10 / 0.05), 4)
        elif 0.60 < s <= 0.65:
            return round(1.0 - (s - 0.60) * (0.10 / 0.05), 4)
        elif 0.20 <= s < 0.35:
            return round(0.30 + (s - 0.20) * (0.60 / 0.15), 4)
        elif 0.65 < s <= 0.80:
            return round(0.90 - (s - 0.65) * (0.50 / 0.15), 4)
        elif s > 0.80:
            return round(max(0.05, 0.40 - (s - 0.80) * (0.35 / 0.20)), 4)
        else:
            if s <= 0.0:
                return 0.0
            return round(s * (0.30 / 0.20), 4)

    @staticmethod
    def get_vector_zone(similarity: float) -> str:
        """Categorize semantic corridor into descriptive interdisciplinary zones."""
        if 0.35 <= similarity <= 0.65:
            return "Sweet Spot"
        elif 0.65 < similarity <= 0.80:
            return "Moderate Overlap"
        elif similarity > 0.80:
            return "Trivial Overlap"
        elif 0.20 <= similarity < 0.35:
            return "Distant Analogy"
        else:
            return "Conceptual Disconnect"


class TopicCrossBreeder:
    """
    Algorithmic cross-breeding engine with Domain Affinity filtering and Local Vector Intelligence.
    Blends input scientific topics with compatible interdisciplinary domains
    via formal epistemic transfer operators (without paid LLMs).
    """

    def __init__(self, seed: Optional[int] = None, vector_engine: Optional[SemanticCorridorEngine] = None):
        self.rng = random.Random(seed)
        self.vector_engine = vector_engine or SemanticCorridorEngine.get_instance()

    def cross_breed(self, topics: List[str], count: int = 10) -> List[GeneratedTopic]:
        if not topics:
            raise ValueError("At least one scientific topic must be provided.")

        cleaned_topics = [clean_topic_name(t) for t in topics if clean_topic_name(t)]
        if not cleaned_topics:
            raise ValueError("All provided topics were empty after cleaning.")

        topic_map: Dict[str, str] = {}
        resolved_topics: List[str] = []
        for raw_t in cleaned_topics:
            if MultilingualBridge.is_cyrillic(raw_t):
                en_t = clean_topic_name(MultilingualBridge.translate_to_en(raw_t))
                disp_t = MultilingualBridge.format_bilingual(raw_t, en_t)
                topic_map[en_t] = disp_t
                resolved_topics.append(en_t)
            else:
                topic_map[raw_t] = raw_t
                resolved_topics.append(raw_t)

        # Classify each input topic into its primary science category
        topic_categories = {t: classify_topic(t) for t in resolved_topics}

        # Collect and filter domains based on affinity
        candidates: List[GeneratedTopic] = []

        def article_for(word: str) -> str:
            return "An" if word and word[0].lower() in "aeiou" else "A"

        # Generate single-topic cross-breeds
        for topic in resolved_topics:
            cat = topic_categories[topic]
            fmt_topic = format_topic_for_title(topic)
            disp_topic = topic_map.get(topic, fmt_topic)

            # Sort available domains by affinity to ensure high-affinity domains are prioritized
            scored_domains = [
                (d, d.affinities.get(cat, 0.5 if cat != ScienceCategory.GENERAL_SCIENCE else 0.8))
                for d in DOMAINS
            ]
            # Filter out incompatible domains (affinity < 0.6), but keep top if all are low
            compatible_domains = [d for d, aff in scored_domains if aff >= 0.6]
            if len(compatible_domains) < 8:
                compatible_domains = [d for d, aff in sorted(scored_domains, key=lambda x: x[1], reverse=True)[:10]]

            self.rng.shuffle(compatible_domains)

            for domain in compatible_domains:
                aff = domain.affinities.get(cat, 0.8)
                method = self.rng.choice(domain.methods)
                phenom = self.rng.choice(domain.phenomena)
                concept = self.rng.choice(domain.concepts)
                prop = self.rng.choice(domain.properties)
                domain_short = domain.name.split('&')[0].strip()

                # Operator 1: Methodological Transfer
                m_title = f"{method} in {fmt_topic}: Resolving {prop}"
                m_rat = f"Transfers {domain.name} methods ({method}) to investigate {prop.lower()} in {fmt_topic}."
                candidates.append(self._build_candidate(
                    title=m_title,
                    primary_topic=fmt_topic,
                    secondary_topic=None,
                    domain=domain,
                    operator_name="Methodological Transfer",
                    rationale=m_rat,
                    affinity=aff,
                    display_primary_topic=disp_topic
                ))

                # Operator 2: Mechanistic Analogy & Dynamics
                p_title = f"{phenom} Dynamics in {fmt_topic}: Mechanisms of {prop}"
                p_rat = f"Analyzes {fmt_topic} through the lens of {phenom.lower()}, mapping dynamic transitions to {prop.lower()}."
                candidates.append(self._build_candidate(
                    title=p_title,
                    primary_topic=fmt_topic,
                    secondary_topic=None,
                    domain=domain,
                    operator_name="Mechanistic Analogy",
                    rationale=p_rat,
                    affinity=aff,
                    display_primary_topic=disp_topic
                ))

                # Operator 3: Structural & Bio-Physical Coupling
                c_title = f"Coupling {concept} with {fmt_topic}: Implications for {prop}"
                c_rat = f"Investigates the interfacial interplay between {concept.lower()} and {fmt_topic}."
                candidates.append(self._build_candidate(
                    title=c_title,
                    primary_topic=fmt_topic,
                    secondary_topic=None,
                    domain=domain,
                    operator_name="Structural Coupling",
                    rationale=c_rat,
                    affinity=aff,
                    display_primary_topic=disp_topic
                ))

        # Operator 4: Dual-Topic Convergence (if >= 2 topics provided)
        if len(resolved_topics) >= 2:
            pairs = []
            for i in range(len(resolved_topics)):
                for j in range(i + 1, len(resolved_topics)):
                    pairs.append((resolved_topics[i], resolved_topics[j]))

            for (t_a, t_b) in pairs:
                fmt_a = format_topic_for_title(t_a)
                fmt_b = format_topic_for_title(t_b)
                cat_a = topic_categories[t_a]
                cat_b = topic_categories[t_b]
                disp_a = topic_map.get(t_a, fmt_a)
                disp_b = topic_map.get(t_b, fmt_b)

                for domain in DOMAINS:
                    aff_a = domain.affinities.get(cat_a, 0.6)
                    aff_b = domain.affinities.get(cat_b, 0.6)
                    dual_aff = (aff_a + aff_b) / 2.0

                    # Only bridge topics using domains with solid joint affinity
                    if dual_aff < 0.6:
                        continue

                    concept = self.rng.choice(domain.concepts)
                    domain_short = domain.name.split('&')[0].strip()
                    title = f"Bridging {fmt_a} and {fmt_b} via {concept}: A {domain_short} Framework"
                    rationale = f"Synthesizes {fmt_a} and {fmt_b} utilizing {concept.lower()} from {domain.name} as a formal conceptual bridge."
                    candidates.append(self._build_candidate(
                        title=title,
                        primary_topic=fmt_a,
                        secondary_topic=fmt_b,
                        domain=domain,
                        operator_name="Dual-Topic Synthesis",
                        rationale=rationale,
                        affinity=dual_aff,
                        display_primary_topic=disp_a,
                        display_secondary_topic=disp_b
                    ))

        # Diversity & Affinity Selection: Pick exactly `count` unique domains
        selected = self._select_diverse(candidates, count=count, has_multiple_inputs=len(resolved_topics) >= 2)

        # Re-index 1..count
        for idx, item in enumerate(selected, start=1):
            item.index = idx

        return selected

    def _build_candidate(
        self,
        title: str,
        primary_topic: str,
        secondary_topic: Optional[str],
        domain: InterdisciplinaryDomain,
        operator_name: str,
        rationale: str,
        affinity: float,
        display_primary_topic: Optional[str] = None,
        display_secondary_topic: Optional[str] = None
    ) -> GeneratedTopic:
        """Construct a candidate with targeted primary and fallback search queries."""
        anchor_p, terms_p = extract_anchor_phrase_and_terms(primary_topic)
        domain_kw = domain.keywords[0]
        domain_alt = domain.keywords[1] if len(domain.keywords) > 1 else domain_kw
        alt_concepts = [c for c in (domain.methods[:2] + domain.phenomena[:2] + domain.keywords[:2]) if c.lower() != domain_kw.lower()]
        concept_alt = alt_concepts[0] if alt_concepts else domain_alt

        # Search query engineering with Anchor Invariance:
        # The user's seed topic is an immutable anchor that is NEVER stripped.
        if secondary_topic:
            anchor_s, terms_s = extract_anchor_phrase_and_terms(secondary_topic)
            primary_query = f'"{anchor_p}" "{anchor_s}" {domain_kw}'
            fallback_query = f'"{anchor_p}" "{anchor_s}" ({domain_alt} OR "{concept_alt}")'
            all_anchor_terms = list(dict.fromkeys(terms_p + terms_s))
            all_anchor_phrase = f"{anchor_p} {anchor_s}"
        else:
            primary_query = f'"{anchor_p}" ({domain_kw} OR "{concept_alt}")'
            fallback_query = f'"{anchor_p}" ({domain_alt} OR "{concept_alt}")'
            all_anchor_terms = terms_p
            all_anchor_phrase = anchor_p

        # Autonomous Vector Scoring via FastEmbed ONNX
        cos_sim = None
        goldilocks = None
        zone = None
        if self.vector_engine and self.vector_engine.is_available:
            try:
                u = self.vector_engine.embed(primary_topic)
                v = self.vector_engine.embed(f"{domain.name}: {', '.join(domain.methods[:2])}")
                if u is not None and v is not None:
                    cos_sim = round(self.vector_engine.cosine_similarity(u, v), 3)
                    goldilocks = round(self.vector_engine.calculate_goldilocks_score(cos_sim), 3)
                    zone = self.vector_engine.get_vector_zone(cos_sim)
            except Exception:
                cos_sim = None
                goldilocks = None
                zone = None

        return GeneratedTopic(
            index=0,
            title=title,
            primary_topic=primary_topic,
            secondary_topic=secondary_topic,
            domain=domain,
            operator_name=operator_name,
            rationale=rationale,
            affinity=affinity,
            primary_query=primary_query,
            fallback_query=fallback_query,
            display_primary_topic=display_primary_topic or primary_topic,
            display_secondary_topic=display_secondary_topic or secondary_topic,
            cosine_similarity=cos_sim,
            goldilocks_score=goldilocks,
            vector_zone=zone,
            anchor_phrase=all_anchor_phrase,
            anchor_terms=all_anchor_terms
        )

    def _select_diverse(
        self,
        candidates: List[GeneratedTopic],
        count: int = 10,
        has_multiple_inputs: bool = False
    ) -> List[GeneratedTopic]:
        """Greedily maximize domain diversity, affinity score, and operator balance."""
        selected: List[GeneratedTopic] = []
        used_domains: Set[str] = set()
        used_operators: Dict[str, int] = {}

        # Sort candidates by composite of domain affinity and Goldilocks vector sweet spot
        shuffled = list(candidates)
        self.rng.shuffle(shuffled)
        def rank_score(c: GeneratedTopic) -> float:
            base = c.affinity
            if c.goldilocks_score is not None:
                return 0.7 * base + 0.3 * c.goldilocks_score
            return base

        shuffled.sort(key=rank_score, reverse=True)

        # Priority 1: If multiple inputs, ensure at least 1-2 Dual-Topic Syntheses
        if has_multiple_inputs:
            dual_candidates = [c for c in shuffled if c.operator_name == "Dual-Topic Synthesis"]
            for c in dual_candidates:
                if len([s for s in selected if s.operator_name == "Dual-Topic Synthesis"]) >= 2:
                    break
                if c.domain.name not in used_domains:
                    selected.append(c)
                    used_domains.add(c.domain.name)
                    used_operators[c.operator_name] = used_operators.get(c.operator_name, 0) + 1

        # Priority 2: Distinct domains with highest affinity
        for c in shuffled:
            if len(selected) >= count:
                break
            if c.domain.name not in used_domains:
                selected.append(c)
                used_domains.add(c.domain.name)
                used_operators[c.operator_name] = used_operators.get(c.operator_name, 0) + 1

        # Priority 3: Fill remainder with minimal operator clash
        if len(selected) < count:
            remaining = [c for c in shuffled if c not in selected]
            remaining.sort(key=lambda x: (used_operators.get(x.operator_name, 0), -x.affinity))
            for c in remaining:
                if len(selected) >= count:
                    break
                selected.append(c)
                used_operators[c.operator_name] = used_operators.get(c.operator_name, 0) + 1

        return selected[:count]


# =====================================================================
# =====================================================================
# Tri-Axial Academic Literature Models & Clients
# =====================================================================
@dataclass
class RetrievedPaper:
    title: str
    authors: List[str]
    pub_year: Optional[str]
    venue: Optional[str]
    citations: Optional[int]
    abstract: Optional[str]
    url: Optional[str]
    query_used: str
    is_fallback: bool
    source: str = "Google Scholar"
    doi: Optional[str] = None
    oa_url: Optional[str] = None
    bibtex: Optional[str] = None
    axis: str = "Landmark Publication"


@dataclass
class LiteratureTriad:
    foundation: Optional[RetrievedPaper] = None
    frontier: Optional[RetrievedPaper] = None
    review: Optional[RetrievedPaper] = None

    @property
    def papers(self) -> List[RetrievedPaper]:
        return [p for p in (self.foundation, self.frontier, self.review) if p is not None]

    @property
    def consolidated_bibtex(self) -> str:
        entries = [p.bibtex.strip() for p in self.papers if p.bibtex]
        return "\n\n".join(entries)


def is_same_paper(p1: Optional[RetrievedPaper], p2: Optional[RetrievedPaper]) -> bool:
    """Check if two papers refer to the same publication via DOI or title match."""
    if not p1 or not p2:
        return False
    if p1.doi and p2.doi and p1.doi.strip().lower() == p2.doi.strip().lower():
        return True
    t1 = re.sub(r'[^a-zA-Z0-9]', '', p1.title.lower())
    t2 = re.sub(r'[^a-zA-Z0-9]', '', p2.title.lower())
    return t1 == t2


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct plain text abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    indexed_words: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            indexed_words.append((pos, word))
    indexed_words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in indexed_words)


def generate_bibtex(paper: RetrievedPaper) -> str:
    """Generate a clean, standard BibTeX entry from paper metadata."""
    first_author = "scholar"
    if paper.authors:
        parts = paper.authors[0].strip().split()
        if parts:
            first_author = re.sub(r'[^a-zA-Z]', '', parts[-1].lower()) or "author"

    year_str = str(paper.pub_year) if paper.pub_year and str(paper.pub_year).isdigit() else "2026"
    
    title_words = re.findall(r'[a-zA-Z]+', paper.title)
    title_key = "ref"
    for w in title_words:
        if w.lower() not in {"a", "an", "the", "on", "in", "for", "of", "and", "to", "with", "by", "is"}:
            title_key = w.lower()
            break

    citekey = f"{first_author}{year_str}{title_key}"
    authors_bib = " and ".join(paper.authors) if paper.authors else "Anonymous"

    lines = [
        f"@article{{{citekey},",
        f"  title     = {{{paper.title}}},",
        f"  author    = {{{authors_bib}}},",
        f"  year      = {{{year_str}}},",
    ]
    if paper.venue and paper.venue not in {"Academic Venue", "NA", "N/A"}:
        lines.append(f"  journal   = {{{paper.venue}}},")
    if paper.doi:
        doi_val = paper.doi.replace("https://doi.org/", "")
        lines.append(f"  doi       = {{{doi_val}}},")
    if paper.oa_url:
        lines.append(f"  note      = {{Open Access: {paper.oa_url}}},")
    if paper.url and paper.url != "https://scholar.google.com":
        lines.append(f"  url       = {{{paper.url}}},")

    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def save_bibtex(bibtex_entry: str, filepath: str = "references.bib") -> bool:
    """Append a BibTeX entry to the specified .bib file."""
    try:
        with open(filepath, "a", encoding="utf-8") as fh:
            fh.write("\n" + bibtex_entry.strip() + "\n")
        return True
    except Exception as err:
        sys.stderr.write(f"Error saving BibTeX to {filepath}: {err}\n")
        return False


# =====================================================================
# Academic Search Clients: OpenAlex (Free) & Google Scholar
# =====================================================================
class OpenAlexClient:
    """Free academic metadata client querying the OpenAlex API without requiring an API key."""

    BASE_URL = "https://api.openalex.org/works"
    USER_AGENT = "ScienceExpander/2.0 (mailto:science-expander@academic-tools.org)"

    @classmethod
    def search_works(
        cls,
        query: str,
        min_year: Optional[int] = None,
        is_review: bool = False,
        sort: str = "relevance_score:desc",
        per_page: int = 5,
        timeout: int = 10
    ) -> List[RetrievedPaper]:
        # Punctuation cleaning & whitespace normalization for exact multi-term grouping
        clean_q = re.sub(r'[:;,."\'\-()\[\]/?!]', ' ', query)
        clean_q = re.sub(r'\b(OR|AND)\b', ' ', clean_q)
        clean_q = re.sub(r'\s+', ' ', clean_q).strip()

        params = [
            f"search={urllib.parse.quote(clean_q)}",
            f"per-page={per_page}",
            f"sort={sort}"
        ]
        filters = []
        if min_year:
            filters.append(f"from_publication_date:{min_year}-01-01")
        if is_review:
            filters.append("type:review")
        if filters:
            params.append(f"filter={','.join(filters)}")

        url = f"{cls.BASE_URL}?{'&'.join(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})

        papers: List[RetrievedPaper] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return cls._search_crossref(clean_q, min_year=min_year, is_review=is_review, per_page=per_page, timeout=timeout)
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                for work in results:
                    title = work.get("title") or work.get("display_name") or "Untitled Publication"
                    authors = [
                        a.get("author", {}).get("display_name")
                        for a in work.get("authorships", [])
                        if a.get("author", {}).get("display_name")
                    ]
                    pub_year = str(work.get("publication_year", "N/A"))

                    primary_loc = work.get("primary_location") or {}
                    source_obj = primary_loc.get("source") or {}
                    venue = source_obj.get("display_name") or "Academic Publication"

                    citations = work.get("cited_by_count")
                    doi = work.get("doi")
                    open_access = work.get("open_access") or {}
                    oa_url = open_access.get("oa_url")
                    url_link = oa_url or doi or work.get("id") or "https://openalex.org"

                    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

                    paper = RetrievedPaper(
                        title=title,
                        authors=authors,
                        pub_year=pub_year,
                        venue=venue,
                        citations=citations,
                        abstract=abstract,
                        url=url_link,
                        query_used=query,
                        is_fallback=True,
                        source="OpenAlex (Open Access)" if oa_url else "OpenAlex",
                        doi=doi,
                        oa_url=oa_url
                    )
                    paper.bibtex = generate_bibtex(paper)
                    papers.append(paper)
            if not papers:
                return cls._search_crossref(clean_q, min_year=min_year, is_review=is_review, per_page=per_page, timeout=timeout)
            return papers
        except Exception:
            return cls._search_crossref(clean_q, min_year=min_year, is_review=is_review, per_page=per_page, timeout=timeout)

    @classmethod
    def _search_crossref(
        cls,
        query: str,
        min_year: Optional[int] = None,
        is_review: bool = False,
        per_page: int = 5,
        timeout: int = 10
    ) -> List[RetrievedPaper]:
        """Free, unlimited academic fallback querying the Crossref API."""
        try:
            q = urllib.parse.quote(query)
            params = [f"query={q}", f"rows={per_page}", "sort=relevance"]
            filters = ["type:journal-article"]
            if min_year:
                filters.append(f"from-pub-date:{min_year}-01-01")
            params.append(f"filter={','.join(filters)}")
            url = f"https://api.crossref.org/works?{'&'.join(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "ScienceExpander/2.0 (mailto:academic-tools@science.org)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("message", {}).get("items", [])
                papers: List[RetrievedPaper] = []
                for it in items:
                    titles = it.get("title") or []
                    if not titles or not titles[0] or not str(titles[0]).strip():
                        continue
                    title = str(titles[0]).strip()
                    authors = []
                    for a in it.get("author", []):
                        fn = a.get("given", "")
                        ln = a.get("family", "")
                        name = f"{fn} {ln}".strip() or a.get("name")
                        if name:
                            authors.append(name)
                    issued = it.get("issued", {}).get("date-parts", [[None]])[0]
                    pub_year = str(issued[0]) if issued and issued[0] else "N/A"
                    venues = it.get("container-title") or []
                    venue = venues[0] if venues else "Academic Venue"
                    citations = it.get("is-referenced-by-count", 0)
                    doi = it.get("DOI")
                    url_link = it.get("URL") or (f"https://doi.org/{doi}" if doi else "https://crossref.org")
                    oa_url = None
                    for lk in it.get("link", []):
                        if "pdf" in lk.get("content-type", "") or "application/pdf" in lk.get("content-type", ""):
                            oa_url = lk.get("URL")
                            break
                    if not oa_url and doi:
                        oa_url = f"https://doi.org/{doi}"

                    raw_abstract = it.get("abstract", "")
                    if raw_abstract:
                        clean_abstract = re.sub(r'<[^>]+>', ' ', raw_abstract)
                        abstract = re.sub(r'\s+', ' ', clean_abstract).strip()
                    else:
                        abstract = f"Indexed publication in Crossref under DOI {doi}. Focuses on {title}."

                    paper = RetrievedPaper(
                        title=title,
                        authors=authors,
                        pub_year=pub_year,
                        venue=venue,
                        citations=citations,
                        abstract=abstract,
                        url=url_link,
                        query_used=query,
                        is_fallback=True,
                        source="Crossref (Open Academic Index)",
                        doi=doi,
                        oa_url=oa_url
                    )
                    paper.bibtex = generate_bibtex(paper)
                    papers.append(paper)
                return papers
        except Exception:
            return []

    @classmethod
    def search_work(cls, query: str, timeout: int = 10) -> Optional[RetrievedPaper]:
        res = cls.search_works(query, timeout=timeout, per_page=1)
        return res[0] if res else None


class ScholarLiteratureClient:
    """
    Tri-Axial Academic Literature Retrieval Engine.
    Coordinates Google Scholar queries with automated OpenAlex fallbacks across three axes:
    1. The Foundation (Seminal landmark publication)
    2. The Frontier (2024-2026 cutting-edge publication/preprint)
    3. The Review (Comprehensive state-of-the-art survey)
    """

    @classmethod
    def rerank_and_filter_candidates(
        cls,
        candidates: List[RetrievedPaper],
        topic: GeneratedTopic,
        anchor_terms: List[str],
        axis: str,
        min_similarity: float = 0.35
    ) -> List[Tuple[RetrievedPaper, float, float]]:
        """
        Rerank candidate papers using local FastEmbed vector embeddings and the Anchor Gate.
        Returns sorted list of (paper, composite_score, cosine_sim) for qualifying papers.
        """
        if not candidates:
            return []

        engine = SemanticCorridorEngine.get_instance()
        topic_target_text = f"{topic.title}. {topic.rationale}. {topic.domain.name}"
        vec_topic = engine.embed(topic_target_text) if (engine and engine.is_available) else None

        qualifying: List[Tuple[RetrievedPaper, float, float]] = []

        for p in candidates:
            # 1. Anchor Gate Check
            if not check_anchor_gate(p, anchor_terms):
                continue

            # 2. Semantic Vector Gate Check
            p_text = f"{p.title}. {p.abstract or ''}"
            if vec_topic is not None and engine and engine.is_available:
                vec_p = engine.embed(p_text)
                cos_sim = engine.cosine_similarity(vec_p, vec_topic) if vec_p is not None else 0.5
            else:
                cos_sim = 0.5

            if cos_sim < min_similarity:
                continue

            # 3. Composite Relevance Score tailored to axis
            title_lower = p.title.lower()
            anchor_in_title = any(re.search(rf'\b{re.escape(t)}\b', title_lower) for t in anchor_terms)
            title_boost = 0.15 if anchor_in_title else 0.0

            if axis == "foundation":
                cit_score = min(1.0, (p.citations or 0) / 100.0)
                composite = 0.55 * cos_sim + 0.30 * cit_score + title_boost
            elif axis == "frontier":
                year_val = int(p.pub_year) if p.pub_year and p.pub_year.isdigit() else 2020
                year_score = 1.0 if year_val >= 2024 else (0.5 if year_val >= 2022 else 0.1)
                composite = 0.60 * cos_sim + 0.25 * year_score + title_boost
            elif axis == "review":
                is_rev = ("review" in title_lower or "survey" in title_lower or 
                          "advances" in title_lower or "progress" in title_lower or 
                          "perspective" in title_lower or (p.venue and "review" in p.venue.lower()))
                rev_score = 1.0 if is_rev else 0.3
                composite = 0.60 * cos_sim + 0.25 * rev_score + title_boost
            else:
                composite = cos_sim + title_boost

            qualifying.append((p, composite, cos_sim))

        qualifying.sort(key=lambda item: item[1], reverse=True)
        return qualifying

    @staticmethod
    def fetch_triad(topic: GeneratedTopic, silent: bool = False) -> LiteratureTriad:
        def log(msg: str) -> None:
            if not silent:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()

        triad = LiteratureTriad()

        status_ctx = None
        if HAS_RICH and not silent and sys.stderr.isatty() and rich_console:
            status_ctx = rich_console.status("[bold cyan]Querying Google Scholar & OpenAlex across 3 academic axes...", spinner="dots")
            status_ctx.start()

        def set_status(msg: str, fallback_color=Style.CYAN):
            if status_ctx:
                status_ctx.update(f"[bold]{msg}[/]")
            else:
                log(color(msg, Style.DIM, fallback_color))

        try:
            # Anchor Invariance: derive core anchor terms and relaxed anchor query
            anchor_phrase = topic.anchor_phrase or extract_anchor_phrase_and_terms(topic.primary_topic)[0]
            anchor_terms = topic.anchor_terms or extract_anchor_phrase_and_terms(topic.primary_topic)[1]

            clean_domain_kws = []
            for k in topic.domain.keywords[:2]:
                k_clean = re.sub(r'\(.*?\)', '', k).strip()
                if k_clean and k_clean not in clean_domain_kws:
                    clean_domain_kws.append(k_clean)
            if not clean_domain_kws:
                clean_domain_kws = [topic.domain.name]
            domain_str = " OR ".join([f'"{k}"' if " " in k else k for k in clean_domain_kws])
            relaxed_anchor_query = f'"{anchor_phrase}" ({domain_str})'

            # =================================================================
            # Axis 1: THE FOUNDATION (Seminal Landmark Publication)
            # =================================================================
            set_status(f"[*] [1/3 Foundation] Fetching seminal landmark papers for: {topic.primary_query}...", Style.CYAN)
            found_cands = ScholarLiteratureClient._execute_scholar_search(topic.primary_query, log_func=log, max_results=5)
            qualifying = ScholarLiteratureClient.rerank_and_filter_candidates(
                found_cands, topic, anchor_terms, axis="foundation", min_similarity=0.35
            )
            oa_found_cands = []
            if not qualifying:
                oa_found_cands = OpenAlexClient.search_works(topic.primary_query, sort="relevance_score:desc", per_page=10)
                qualifying = ScholarLiteratureClient.rerank_and_filter_candidates(
                    oa_found_cands, topic, anchor_terms, axis="foundation", min_similarity=0.35
                )
            relaxed_cands = []
            if not qualifying:
                relaxed_cands = OpenAlexClient.search_works(relaxed_anchor_query, sort="relevance_score:desc", per_page=10)
                qualifying = ScholarLiteratureClient.rerank_and_filter_candidates(
                    relaxed_cands, topic, anchor_terms, axis="foundation", min_similarity=0.35
                )
            if not qualifying and relaxed_cands:
                qualifying = ScholarLiteratureClient.rerank_and_filter_candidates(
                    relaxed_cands, topic, anchor_terms, axis="foundation", min_similarity=0.20
                )
            anchor_only_cands = []
            if not qualifying:
                anchor_only_cands = OpenAlexClient.search_works(f'"{anchor_phrase}"', sort="relevance_score:desc", per_page=10)
                qualifying = ScholarLiteratureClient.rerank_and_filter_candidates(
                    anchor_only_cands, topic, anchor_terms, axis="foundation", min_similarity=0.15
                )

            if qualifying:
                triad.foundation = qualifying[0][0]
                triad.foundation.axis = "The Foundation (Seminal Landmark)"
            else:
                all_cands = found_cands + oa_found_cands + relaxed_cands + anchor_only_cands
                anchor_matches = [c for c in all_cands if check_anchor_gate(c, anchor_terms)]
                if anchor_matches:
                    triad.foundation = anchor_matches[0]
                    triad.foundation.axis = "The Foundation (Seminal Landmark)"

            # =================================================================
            # Axis 2: THE FRONTIER (2024–2026 Cutting-Edge Research & Preprints)
            # =================================================================
            set_status("[*] [2/3 Frontier] Fetching 2024–2026 frontier publications...", Style.YELLOW)
            frontier_cands = ScholarLiteratureClient._execute_scholar_search(topic.primary_query, year_low=2024, log_func=log, max_results=5)
            qualifying_frontier = ScholarLiteratureClient.rerank_and_filter_candidates(
                frontier_cands, topic, anchor_terms, axis="frontier", min_similarity=0.35
            )
            for cand, comp_score, sim in qualifying_frontier:
                if not is_same_paper(cand, triad.foundation):
                    triad.frontier = cand
                    triad.frontier.axis = "The Frontier (2024-2026 Emerging Edge)"
                    break

            oa_frontier_cands = []
            if not triad.frontier:
                oa_frontier_cands = OpenAlexClient.search_works(topic.primary_query, min_year=2024, sort="relevance_score:desc", per_page=10)
                qualifying_frontier = ScholarLiteratureClient.rerank_and_filter_candidates(
                    oa_frontier_cands, topic, anchor_terms, axis="frontier", min_similarity=0.35
                )
                for cand, comp_score, sim in qualifying_frontier:
                    if not is_same_paper(cand, triad.foundation):
                        triad.frontier = cand
                        triad.frontier.axis = "The Frontier (2024-2026 Emerging Edge)"
                        break

            relaxed_frontier_cands = []
            if not triad.frontier:
                relaxed_frontier_cands = OpenAlexClient.search_works(relaxed_anchor_query, min_year=2024, sort="relevance_score:desc", per_page=10)
                qualifying_frontier = ScholarLiteratureClient.rerank_and_filter_candidates(
                    relaxed_frontier_cands, topic, anchor_terms, axis="frontier", min_similarity=0.25
                )
                for cand, comp_score, sim in qualifying_frontier:
                    if not is_same_paper(cand, triad.foundation):
                        triad.frontier = cand
                        triad.frontier.axis = "The Frontier (2024-2026 Emerging Edge)"
                        break

            anchor_frontier_cands = []
            if not triad.frontier:
                anchor_frontier_cands = OpenAlexClient.search_works(f'"{anchor_phrase}"', min_year=2024, sort="relevance_score:desc", per_page=10)
                qualifying_frontier = ScholarLiteratureClient.rerank_and_filter_candidates(
                    anchor_frontier_cands, topic, anchor_terms, axis="frontier", min_similarity=0.15
                )
                for cand, comp_score, sim in qualifying_frontier:
                    if not is_same_paper(cand, triad.foundation):
                        triad.frontier = cand
                        triad.frontier.axis = "The Frontier (2024-2026 Emerging Edge)"
                        break

            if not triad.frontier:
                all_frontier_cands = frontier_cands + oa_frontier_cands + relaxed_frontier_cands + anchor_frontier_cands
                anchor_matches = [c for c in all_frontier_cands if check_anchor_gate(c, anchor_terms) and not is_same_paper(c, triad.foundation)]
                if anchor_matches:
                    triad.frontier = anchor_matches[0]
                    triad.frontier.axis = "The Frontier (2024-2026 Emerging Edge)"

            # =================================================================
            # Axis 3: THE REVIEW (State-of-the-Art Comprehensive Survey)
            # =================================================================
            set_status("[*] [3/3 Review] Fetching comprehensive review/survey literature...", Style.MAGENTA)
            review_q = f"{topic.primary_query} review"
            review_cands = ScholarLiteratureClient._execute_scholar_search(review_q, log_func=log, max_results=5)
            qualifying_review = ScholarLiteratureClient.rerank_and_filter_candidates(
                review_cands, topic, anchor_terms, axis="review", min_similarity=0.35
            )
            for cand, comp_score, sim in qualifying_review:
                if not is_same_paper(cand, triad.foundation) and not is_same_paper(cand, triad.frontier):
                    triad.review = cand
                    triad.review.axis = "The Review (State-of-the-Art Survey)"
                    break

            oa_review_cands = []
            if not triad.review:
                oa_review_cands = OpenAlexClient.search_works(topic.primary_query, is_review=True, sort="relevance_score:desc", per_page=10)
                if not oa_review_cands:
                    oa_review_cands = OpenAlexClient.search_works(review_q, sort="relevance_score:desc", per_page=10)
                qualifying_review = ScholarLiteratureClient.rerank_and_filter_candidates(
                    oa_review_cands, topic, anchor_terms, axis="review", min_similarity=0.35
                )
                for cand, comp_score, sim in qualifying_review:
                    if not is_same_paper(cand, triad.foundation) and not is_same_paper(cand, triad.frontier):
                        triad.review = cand
                        triad.review.axis = "The Review (State-of-the-Art Survey)"
                        break

            relaxed_review_cands = []
            if not triad.review:
                relaxed_rev_q = f"{relaxed_anchor_query} review"
                relaxed_review_cands = OpenAlexClient.search_works(relaxed_rev_q, sort="relevance_score:desc", per_page=10)
                qualifying_review = ScholarLiteratureClient.rerank_and_filter_candidates(
                    relaxed_review_cands, topic, anchor_terms, axis="review", min_similarity=0.25
                )
                for cand, comp_score, sim in qualifying_review:
                    if not is_same_paper(cand, triad.foundation) and not is_same_paper(cand, triad.frontier):
                        triad.review = cand
                        triad.review.axis = "The Review (State-of-the-Art Survey)"
                        break

            anchor_review_cands = []
            if not triad.review:
                anchor_review_cands = OpenAlexClient.search_works(anchor_phrase, is_review=True, sort="relevance_score:desc", per_page=10)
                if not anchor_review_cands:
                    anchor_review_cands = OpenAlexClient.search_works(f'"{anchor_phrase}" review', sort="relevance_score:desc", per_page=10)
                qualifying_review = ScholarLiteratureClient.rerank_and_filter_candidates(
                    anchor_review_cands, topic, anchor_terms, axis="review", min_similarity=0.15
                )
                for cand, comp_score, sim in qualifying_review:
                    if not is_same_paper(cand, triad.foundation) and not is_same_paper(cand, triad.frontier):
                        triad.review = cand
                        triad.review.axis = "The Review (State-of-the-Art Survey)"
                        break

            if not triad.review:
                all_rev_cands = review_cands + oa_review_cands + relaxed_review_cands + anchor_review_cands
                anchor_matches = [c for c in all_rev_cands if check_anchor_gate(c, anchor_terms) and not is_same_paper(c, triad.foundation) and not is_same_paper(c, triad.frontier)]
                if anchor_matches:
                    triad.review = anchor_matches[0]
                    triad.review.axis = "The Review (State-of-the-Art Survey)"

            # Ensure BibTeX entries exist for all papers in triad
            for p in triad.papers:
                if not p.bibtex:
                    p.bibtex = generate_bibtex(p)

        finally:
            if status_ctx:
                status_ctx.stop()

        return triad

    @staticmethod
    def fetch_paper(topic: GeneratedTopic, silent: bool = False) -> Optional[RetrievedPaper]:
        """Backwards-compatible single paper fetcher delegating to the Foundation axis."""
        triad = ScholarLiteratureClient.fetch_triad(topic, silent=silent)
        return triad.foundation or (triad.papers[0] if triad.papers else None)

    @staticmethod
    def _execute_scholar_search(
        query: str,
        year_low: Optional[int] = None,
        is_fallback: bool = False,
        log_func=None,
        max_results: int = 3,
        timeout: float = 2.5
    ) -> List[RetrievedPaper]:
        papers: List[RetrievedPaper] = []
        err_box: List[Exception] = []

        def _worker():
            try:
                kwargs = {}
                if year_low:
                    kwargs["year_low"] = year_low
                search_gen = scholarly.search_pubs(query, **kwargs)
                for _ in range(max_results):
                    pub = next(search_gen, None)
                    if not pub:
                        break
                    bib = pub.get("bib", {})
                    authors = bib.get("author", [])
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(" and ")]

                    paper = RetrievedPaper(
                        title=bib.get("title", "Untitled Publication"),
                        authors=authors,
                        pub_year=str(bib.get("pub_year", "N/A")),
                        venue=bib.get("venue", bib.get("journal", "Academic Venue")),
                        citations=pub.get("num_citations"),
                        abstract=bib.get("abstract", "No abstract snippet available."),
                        url=pub.get("pub_url") or pub.get("eprint_url") or "https://scholar.google.com",
                        query_used=query,
                        is_fallback=is_fallback,
                        source="Google Scholar"
                    )
                    paper.bibtex = generate_bibtex(paper)
                    papers.append(paper)
            except StopIteration:
                pass
            except Exception as err:
                err_box.append(err)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout=timeout)

        if worker_thread.is_alive():
            if log_func:
                log_func(color(f"[!] Google Scholar search timed out ({timeout}s / anti-bot challenge). Falling back smoothly to OpenAlex.", Style.YELLOW))
            return []

        if err_box and log_func:
            log_func(color(f"[!] Warning during Scholar search: {err_box[0]}", Style.YELLOW))

        return papers


# =====================================================================
# Automated Research Brief Generator (Research_Brief.md)
# =====================================================================
def generate_research_brief(
    topic: GeneratedTopic,
    triad: LiteratureTriad,
    all_topics: List[GeneratedTopic],
    raw_topics: List[str]
) -> str:
    """Generate a publication-ready, comprehensive Markdown Research Brief."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds_str = ", ".join(raw_topics)
    cat_name = classify_topic(topic.primary_topic).value.replace("_", " ").title()

    table_rows = []
    for t in all_topics:
        sel_mark = " **(Selected)**" if t.index == topic.index else ""
        row = f"| {t.index:02d} | {t.title}{sel_mark} | {t.domain.name} | {int(t.affinity * 100)}% | {t.operator_name} |"
        table_rows.append(row)
    landscape_table = "\n".join(table_rows)

    def format_paper_section(p: Optional[RetrievedPaper]) -> str:
        if not p:
            return "*No indexed publication retrieved for this axis.*"
        md = [
            f"- **Title**: {p.title}",
            f"- **Authors**: {', '.join(p.authors) if p.authors else 'Unknown'}",
            f"- **Year**: {p.pub_year} | **Venue**: {p.venue} | **Citations**: {p.citations if p.citations is not None else 'N/A'}",
            f"- **Source**: {p.source}",
            f"- **DOI / Landing Page**: [{p.url}]({p.url})"
        ]
        if p.oa_url:
            md.append(f"- **Direct Open Access PDF**: [{p.oa_url}]({p.oa_url})")
        if p.abstract and p.abstract != "No abstract snippet available.":
            md.append(f"\n> **Abstract Snippet**: \"{p.abstract.strip()}\"")
        return "\n".join(md)

    brief_md = f"""# Research Brief: {topic.title}
*Generated automatically on {now_str} by Science Expander v2.0*

## Executive Summary & Conceptual Formulation
- **Seed Scientific Topics**: {seeds_str}
- **Primary Scientific Faculty**: {cat_name}
- **Interdisciplinary Bridge**: {topic.domain.name} (`{topic.domain.category_label}`)
- **Epistemic Transfer Lens**: {topic.operator_name}
- **Domain Compatibility Affinity**: {int(topic.affinity * 100)}%

### Core Scientific Formulation & Research Problem
{topic.rationale}

This interdisciplinary inquiry establishes a formal bridge between **{topic.display_primary_topic or topic.primary_topic}** and the established methods of **{topic.domain.name}**. By translating analytical mechanisms such as *{topic.domain.methods[0]}* and physical phenomena like *{topic.domain.phenomena[0]}*, this research direction addresses foundational questions in *{topic.domain.properties[0].lower()}* and macroscopic system dynamics.

---

## Interdisciplinary Landscape Map
Overview of the 10 algorithmically generated research directions across compatible scientific faculties:

| # | Proposed Research Direction | Interdisciplinary Bridge | Affinity | Epistemic Lens |
|---|---|---|:---:|---|
{landscape_table}

---

## Deep Dive: The Literature Triad
Detailed 3-dimensional academic slice for Topic #{topic.index:02d}: **{topic.title}**

### 🏛️ The Foundation: Seminal Landmark Publication
{format_paper_section(triad.foundation)}

### ⚡ The Frontier (2024–2026): Emerging Edge & Preprints
{format_paper_section(triad.frontier)}

### 📚 The Review: Comprehensive Survey of the State-of-the-Art
{format_paper_section(triad.review)}

---

## Integrated BibTeX Appendix
Consolidated bibliography entries ready for direct import into Overleaf, LaTeX, Zotero, or Mendeley:

```bibtex
{triad.consolidated_bibtex if triad.consolidated_bibtex else "% No BibTeX entries generated."}
```
"""
    return brief_md


def save_research_brief(content: str, filepath: str = "Research_Brief.md") -> bool:
    """Save formatted Markdown Research Brief to a file."""
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True
    except Exception as err:
        sys.stderr.write(f"Error saving Research Brief to {filepath}: {err}\n")
        return False


# =====================================================================
# Standalone Interactive Knowledge Graph Generator (knowledge_graph.html)
# =====================================================================
HTML_GRAPH_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Science Expander | Knowledge Graph: __PAGE_TITLE__</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #07090e;
      --bg-surface: rgba(15, 22, 36, 0.78);
      --border-color: rgba(255, 255, 255, 0.1);
      --accent-cyan: #00e5ff;
      --accent-purple: #9d4edd;
      --accent-green: #10b981;
      --accent-gold: #f59e0b;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body, html { width: 100%; height: 100%; overflow: hidden; background-color: var(--bg-primary); font-family: 'Inter', system-ui, -apple-system, sans-serif; color: var(--text-primary); }

    .ambient-glow {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: radial-gradient(circle at 20% 30%, rgba(157, 78, 221, 0.08) 0%, transparent 50%),
                  radial-gradient(circle at 80% 70%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
                  radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 60%);
      pointer-events: none;
      z-index: 0;
    }

    #network-container { width: 100%; height: 100%; position: absolute; top: 0; left: 0; z-index: 1; }

    .header-bar {
      position: absolute; top: 16px; left: 20px; z-index: 10;
      background: var(--bg-surface);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 12px 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      display: flex; align-items: center; gap: 18px;
    }
    .header-title { font-size: 15px; font-weight: 700; letter-spacing: 0.5px; background: linear-gradient(135deg, #00e5ff 0%, #9d4edd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .header-sub { font-size: 13px; color: var(--text-secondary); max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .stat-badge { font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px; background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.08); color: var(--text-secondary); }

    .toolbar {
      position: absolute; top: 16px; right: 20px; z-index: 10;
      background: var(--bg-surface);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 8px 14px;
      display: flex; align-items: center; gap: 10px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .search-box {
      background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border-color); border-radius: 8px;
      padding: 6px 12px; color: var(--text-primary); font-size: 12px; outline: none; width: 190px;
      transition: all 0.2s ease;
    }
    .search-box:focus { border-color: var(--accent-cyan); width: 230px; box-shadow: 0 0 12px rgba(0, 229, 255, 0.3); }
    .btn {
      background: rgba(255, 255, 255, 0.06); border: 1px solid var(--border-color); border-radius: 8px;
      color: var(--text-primary); font-size: 12px; font-weight: 500; padding: 6px 12px; cursor: pointer;
      display: flex; align-items: center; gap: 6px; transition: all 0.2s ease;
    }
    .btn:hover { background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.2); transform: translateY(-1px); }

    .legend-card {
      position: absolute; bottom: 20px; left: 20px; z-index: 10;
      background: var(--bg-surface);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 14px 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      font-size: 12px;
    }
    .legend-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 8px; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; color: var(--text-secondary); }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot-hub { background: #00e5ff; box-shadow: 0 0 8px #00e5ff; }
    .dot-bridge { background: #9d4edd; box-shadow: 0 0 8px #9d4edd; }
    .dot-idea { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-lit { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }

    .inspector-card {
      position: absolute; bottom: 20px; right: 20px; z-index: 10;
      width: 410px; max-height: calc(100vh - 100px); overflow-y: auto;
      background: var(--bg-surface);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-color);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
      transition: transform 0.3s ease, opacity 0.3s ease;
    }
    .inspector-card::-webkit-scrollbar { width: 4px; }
    .inspector-card::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }

    .node-category-badge {
      display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
      padding: 4px 10px; border-radius: 6px; margin-bottom: 12px; color: #ffffff;
    }
    .node-title { font-size: 16px; font-weight: 700; line-height: 1.35; margin-bottom: 12px; color: #ffffff; }
    .node-meta-grid { display: grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 14px; font-size: 12px; }
    .meta-row { display: flex; align-items: baseline; gap: 8px; color: var(--text-secondary); }
    .meta-label { font-weight: 600; color: var(--text-muted); min-width: 75px; }
    .meta-value { color: var(--text-primary); }

    .btn-action-row { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
    .action-link {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      padding: 9px 14px; border-radius: 10px; font-size: 12px; font-weight: 600; text-decoration: none;
      transition: all 0.2s ease;
    }
    .action-doi { background: rgba(0, 229, 255, 0.12); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.3); }
    .action-doi:hover { background: rgba(0, 229, 255, 0.22); border-color: #00e5ff; }
    .action-oa { background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }
    .action-oa:hover { background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.4) 100%); border-color: #34d399; }

    .content-box {
      margin-top: 14px; padding: 12px; border-radius: 10px; background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.05); font-size: 12px; line-height: 1.5; color: var(--text-secondary);
    }
    .content-box-title { font-weight: 600; color: var(--text-primary); margin-bottom: 6px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }

    .placeholder-guide { text-align: center; padding: 24px 10px; color: var(--text-muted); font-size: 13px; line-height: 1.6; }
  </style>
</head>
<body>
  <div class="ambient-glow"></div>
  <div id="network-container"></div>

  <div class="header-bar">
    <div>
      <div class="header-title">🔬 SCIENCE EXPANDER</div>
      <div class="header-sub">Topic Focus: <strong>__PAGE_TITLE__</strong></div>
    </div>
    <div class="stat-badge">__TOTAL_NODES__ Nodes</div>
    <div class="stat-badge">__TOTAL_EDGES__ Edges</div>
  </div>

  <div class="toolbar">
    <input type="text" id="search-input" class="search-box" placeholder="Search node, author, bridge..." />
    <button id="btn-physics" class="btn">⏸ Freeze</button>
    <button id="btn-fit" class="btn">🔄 Fit View</button>
  </div>

  <div class="legend-card">
    <div class="legend-title">Topology Layers</div>
    <div class="legend-item"><span class="dot dot-hub"></span> Core Input Seed Topics</div>
    <div class="legend-item"><span class="dot dot-bridge"></span> Interdisciplinary Bridges</div>
    <div class="legend-item"><span class="dot dot-idea"></span> Research Directions (10 Ideas)</div>
    <div class="legend-item"><span class="dot dot-lit"></span> Literature Triad (Papers)</div>
  </div>

  <div id="inspector" class="inspector-card">
    <div id="inspector-content">
      <div class="placeholder-guide">
        <div style="font-size: 28px; margin-bottom: 10px;">🌌</div>
        <strong>Interactive Knowledge Topology</strong><br>
        Click or hover over any node to inspect its epistemic framework, bridge domain, or publication metrics.
      </div>
    </div>
  </div>

  <script>
    const nodesData = __NODES_DATA__;
    const edgesData = __EDGES_DATA__;

    const container = document.getElementById('network-container');
    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);

    const data = { nodes: nodes, edges: edges };
    const options = {
      nodes: {
        borderWidth: 2,
        shadow: { enabled: true, color: 'rgba(0,0,0,0.6)', size: 10, x: 0, y: 4 }
      },
      edges: {
        smooth: { type: 'continuous', roundness: 0.35 },
        hoverWidth: 2.5
      },
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -75,
          centralGravity: 0.015,
          springLength: 135,
          springConstant: 0.08,
          damping: 0.45,
          avoidOverlap: 0.7
        },
        stabilization: { iterations: 180, fit: true }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        navigationButtons: true,
        keyboard: true
      }
    };

    const network = new vis.Network(container, data, options);

    function renderInspector(c) {
      if (!c) return;
      const card = document.getElementById('inspector-content');
      let html = `<div class="node-category-badge" style="background:${c.badge_color || '#3b82f6'}">${c.category}</div>`;
      html += `<div class="node-title">${c.title}</div>`;
      html += `<div class="node-meta-grid">`;

      if (c.authors) html += `<div class="meta-row"><span class="meta-label">Authors:</span><span class="meta-value">${c.authors}</span></div>`;
      if (c.year && c.year !== 'N/A') html += `<div class="meta-row"><span class="meta-label">Year:</span><span class="meta-value">${c.year}</span></div>`;
      if (c.venue && c.venue !== 'Academic Publication') html += `<div class="meta-row"><span class="meta-label">Venue:</span><span class="meta-value">${c.venue}</span></div>`;
      if (c.citations) html += `<div class="meta-row"><span class="meta-label">Impact:</span><span class="meta-value" style="color:#fbbf24; font-weight:600;">⚡ ${c.citations}</span></div>`;
      if (c.source) html += `<div class="meta-row"><span class="meta-label">Source:</span><span class="meta-value">${c.source}</span></div>`;
      if (c.faculty) html += `<div class="meta-row"><span class="meta-label">Faculty:</span><span class="meta-value">${c.faculty}</span></div>`;
      if (c.bridge) html += `<div class="meta-row"><span class="meta-label">Bridge:</span><span class="meta-value">${c.bridge}</span></div>`;
      if (c.affinity) html += `<div class="meta-row"><span class="meta-label">Affinity:</span><span class="meta-value" style="color:#34d399; font-weight:700;">${c.affinity}</span></div>`;
      if (c.lens) html += `<div class="meta-row"><span class="meta-label">Epistemic Lens:</span><span class="meta-value">${c.lens}</span></div>`;
      html += `</div>`;

      if (c.url || c.oa_url) {
        html += `<div class="btn-action-row">`;
        if (c.oa_url) {
          html += `<a href="${c.oa_url}" target="_blank" class="action-link action-oa">📄 Direct Open Access PDF</a>`;
        }
        if (c.url && c.url !== 'https://scholar.google.com') {
          html += `<a href="${c.url}" target="_blank" class="action-link action-doi">🔗 Publication Landing Page</a>`;
        }
        html += `</div>`;
      }

      if (c.abstract && c.abstract !== 'No abstract snippet available.') {
        html += `<div class="content-box"><div class="content-box-title">Abstract Snippet</div>"${c.abstract}"</div>`;
      }
      if (c.rationale) {
        html += `<div class="content-box"><div class="content-box-title">Conceptual Blending Formulation</div>${c.rationale}</div>`;
      }
      if (c.description) {
        html += `<div class="content-box"><div class="content-box-title">Overview</div>${c.description}</div>`;
      }
      if (c.query) {
        html += `<div class="content-box" style="font-family:'JetBrains Mono',monospace; font-size:11px;"><div class="content-box-title">Search Query</div>${c.query}</div>`;
      }

      card.innerHTML = html;
    }

    network.on('click', function(params) {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);
        if (node && node.custom_data) renderInspector(node.custom_data);
      }
    });

    network.on('hoverNode', function(params) {
      const node = nodes.get(params.node);
      if (node && node.custom_data) renderInspector(node.custom_data);
    });

    let physicsRunning = true;
    const btnPhys = document.getElementById('btn-physics');
    btnPhys.addEventListener('click', () => {
      physicsRunning = !physicsRunning;
      network.setOptions({ physics: { enabled: physicsRunning } });
      btnPhys.innerText = physicsRunning ? '⏸ Freeze' : '▶ Run Physics';
    });

    document.getElementById('btn-fit').addEventListener('click', () => {
      network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    });

    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        nodes.forEach(n => nodes.update({ id: n.id, opacity: 1 }));
        return;
      }
      let firstMatch = null;
      nodes.forEach(n => {
        const title = (n.custom_data?.title || n.label || '').toLowerCase();
        const cat = (n.custom_data?.category || '').toLowerCase();
        const auth = (n.custom_data?.authors || '').toLowerCase();
        const bridge = (n.custom_data?.bridge || '').toLowerCase();
        const match = title.includes(q) || cat.includes(q) || auth.includes(q) || bridge.includes(q);
        if (match && !firstMatch) firstMatch = n.id;
        nodes.update({ id: n.id, opacity: match ? 1 : 0.25 });
      });
      if (firstMatch) {
        network.focus(firstMatch, { scale: 1.2, animation: { duration: 400 } });
        const matchedNode = nodes.get(firstMatch);
        if (matchedNode?.custom_data) renderInspector(matchedNode.custom_data);
      }
    });
  </script>
</body>
</html>
"""


def generate_knowledge_graph(
    topic: GeneratedTopic,
    triad: LiteratureTriad,
    all_topics: List[GeneratedTopic],
    raw_topics: List[str]
) -> str:
    """Generate a self-contained interactive Vis.js HTML knowledge graph visualization."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nodes = []
    edges = []
    hub_ids = {}

    # 1. Hub Nodes (Core Seed Topics) - Neon Cyan (#00e5ff)
    for i, seed in enumerate(raw_topics):
        hid = f"hub_{i+1}"
        hub_ids[seed] = hid
        m = re.match(r"^(.+?)\s*\((.+?)\)$", seed)
        if m:
            hub_ids[m.group(1).strip()] = hid
            hub_ids[m.group(2).strip()] = hid
        cat = classify_topic(seed)
        nodes.append({
            "id": hid,
            "label": seed,
            "group": "hub",
            "shape": "hexagon",
            "size": 32,
            "color": {
                "background": "#00e5ff",
                "border": "#ffffff",
                "highlight": {"background": "#38bdf8", "border": "#ffffff"},
                "hover": {"background": "#38bdf8", "border": "#ffffff"}
            },
            "font": {"color": "#ffffff", "size": 15, "face": "Inter, sans-serif", "bold": True},
            "borderWidth": 2,
            "custom_data": {
                "category": "Core Seed Topic",
                "title": seed,
                "badge_color": "#00e5ff",
                "faculty": cat.value.replace("_", " ").title(),
                "description": "Input scientific seed topic driving conceptual cross-breeding."
            }
        })

    # 2. Bridge Nodes (Interdisciplinary Domains) - Royal Purple (#9d4edd)
    unique_domains = {}
    for t in all_topics:
        if t.domain.name not in unique_domains:
            unique_domains[t.domain.name] = t.domain

    bridge_ids = {}
    for d_name, d_obj in unique_domains.items():
        bid = f"bridge_{re.sub(r'[^a-zA-Z0-9]', '_', d_name)}"
        bridge_ids[d_name] = bid
        nodes.append({
            "id": bid,
            "label": d_name,
            "group": "bridge",
            "shape": "diamond",
            "size": 24,
            "color": {
                "background": "#9d4edd",
                "border": "#c77dff",
                "highlight": {"background": "#b5179e", "border": "#ffffff"},
                "hover": {"background": "#b5179e", "border": "#ffffff"}
            },
            "font": {"color": "#e0aaff", "size": 13, "face": "Inter, sans-serif"},
            "borderWidth": 2,
            "custom_data": {
                "category": "Interdisciplinary Bridge",
                "title": d_name,
                "badge_color": "#9d4edd",
                "faculty": d_obj.category_label,
                "methods": ", ".join(d_obj.methods[:3]),
                "phenomena": ", ".join(d_obj.phenomena[:3]),
                "description": f"Legitimate interdisciplinary scientific field contributing analytical mechanisms ({', '.join(d_obj.methods[:2])}) and physical phenomena ({', '.join(d_obj.phenomena[:2])})."
            }
        })

    # 3. Idea Nodes (10 Generated Research Topics) - Emerald Green (#10b981 / #00f59b)
    for t in all_topics:
        iid = f"idea_{t.index}"
        is_sel = (t.index == topic.index)
        aff_pct = int(t.affinity * 100)
        aff_badge = f"{aff_pct}% HIGH" if aff_pct >= 95 else f"{aff_pct}% MED"

        nodes.append({
            "id": iid,
            "label": f"#{t.index:02d} {t.title[:24]}..." if len(t.title) > 24 else f"#{t.index:02d} {t.title}",
            "group": "selected_idea" if is_sel else "idea",
            "shape": "dot",
            "size": 28 if is_sel else 18,
            "color": {
                "background": "#00f59b" if is_sel else "#10b981",
                "border": "#ffffff" if is_sel else "#059669",
                "highlight": {"background": "#34d399", "border": "#ffffff"},
                "hover": {"background": "#34d399", "border": "#ffffff"}
            },
            "font": {
                "color": "#ffffff" if is_sel else "#a7f3d0",
                "size": 13 if is_sel else 11,
                "face": "Inter, sans-serif",
                "bold": is_sel
            },
            "borderWidth": 3 if is_sel else 1,
            "custom_data": {
                "category": "Selected Research Direction" if is_sel else "Generated Research Direction",
                "index": t.index,
                "title": t.title,
                "badge_color": "#00f59b" if is_sel else "#10b981",
                "bridge": t.domain.name,
                "faculty": t.domain.category_label,
                "affinity": aff_badge,
                "lens": t.operator_name,
                "rationale": t.rationale,
                "query": t.primary_query,
                "selected": is_sel
            }
        })

    # 4. Literature Triad Nodes (Foundation, Frontier, Review) - Gold/Amber (#f59e0b)
    lit_axes = [
        ("foundation", "🏛️ The Foundation", triad.foundation, "#f59e0b", "#fbbf24"),
        ("frontier", "⚡ The Frontier (2024–2026)", triad.frontier, "#ff9f1c", "#ffd166"),
        ("review", "📚 The Review", triad.review, "#e63946", "#ff758f"),
    ]
    for key, name, p, bg_col, border_col in lit_axes:
        if p:
            lid = f"lit_{key}"
            cit_str = f"{p.citations} citations" if p.citations is not None else "Citations unlisted"
            nodes.append({
                "id": lid,
                "label": f"{name}\n{p.title[:26]}...",
                "group": "literature",
                "shape": "star",
                "size": 25,
                "color": {
                    "background": bg_col,
                    "border": border_col,
                    "highlight": {"background": "#ffffff", "border": border_col},
                    "hover": {"background": "#ffffff", "border": border_col}
                },
                "font": {"color": "#ffffff", "size": 12, "face": "Inter, sans-serif", "bold": True},
                "borderWidth": 2,
                "custom_data": {
                    "category": name,
                    "title": p.title,
                    "badge_color": bg_col,
                    "authors": ", ".join(p.authors) if p.authors else "Unknown",
                    "year": p.pub_year,
                    "venue": p.venue,
                    "citations": cit_str,
                    "source": p.source,
                    "url": p.url,
                    "oa_url": p.oa_url,
                    "doi": p.doi,
                    "abstract": p.abstract
                }
            })

    # Edges Construction
    # Hub -> Bridge (Edge thickness & tooltip reflects cosine similarity)
    connected_bridges = set()
    for t in all_topics:
        h_id = hub_ids.get(t.display_primary_topic) or hub_ids.get(t.primary_topic) or list(hub_ids.values())[0]
        b_id = bridge_ids[t.domain.name]
        pair = (h_id, b_id)
        if pair not in connected_bridges:
            connected_bridges.add(pair)
            sim_val = t.cosine_similarity if t.cosine_similarity is not None else 0.5
            e_width = max(1.5, min(6.0, round(sim_val * 6, 1)))
            edges.append({
                "from": h_id,
                "to": b_id,
                "color": {"color": "rgba(0, 229, 255, 0.45)", "highlight": "#00e5ff", "hover": "#00e5ff"},
                "width": e_width,
                "title": f"Cosine Similarity: {sim_val:.2f} ({t.vector_zone or 'Bridge'})",
                "value": round(sim_val, 2),
                "smooth": {"type": "continuous"}
            })

    # Bridge -> Idea (Edge thickness reflects cosine similarity & selection)
    for t in all_topics:
        b_id = bridge_ids[t.domain.name]
        i_id = f"idea_{t.index}"
        is_sel = (t.index == topic.index)
        sim_val = t.cosine_similarity if t.cosine_similarity is not None else 0.5
        base_w = (sim_val * 8) if is_sel else (sim_val * 4.5)
        e_width = max(1.5, min(8.0, round(base_w, 1)))
        edges.append({
            "from": b_id,
            "to": i_id,
            "color": {
                "color": "rgba(0, 245, 155, 0.75)" if is_sel else "rgba(157, 78, 221, 0.45)",
                "highlight": "#00f59b" if is_sel else "#c77dff"
            },
            "width": e_width,
            "title": f"Cosine Sim: {sim_val:.2f} ({t.vector_zone or 'Synthesized Idea'})",
            "value": round(sim_val, 2),
            "smooth": {"type": "continuous"}
        })

    # Selected Idea -> Literature Nodes
    sel_idea_id = f"idea_{topic.index}"
    for key, name, p, bg_col, border_col in lit_axes:
        if p:
            lid = f"lit_{key}"
            edges.append({
                "from": sel_idea_id,
                "to": lid,
                "color": {"color": bg_col, "highlight": "#ffffff"},
                "width": 3,
                "dashes": [6, 4],
                "label": f"{key.title()} Match",
                "font": {"color": border_col, "size": 10, "strokeWidth": 0}
            })

    nodes_json = json.dumps(nodes, indent=2, ensure_ascii=False)
    edges_json = json.dumps(edges, indent=2, ensure_ascii=False)
    seeds_str = ", ".join(raw_topics)

    html = (
        HTML_GRAPH_TEMPLATE
        .replace("__PAGE_TITLE__", topic.title)
        .replace("__SEEDS__", seeds_str)
        .replace("__TOTAL_NODES__", str(len(nodes)))
        .replace("__TOTAL_EDGES__", str(len(edges)))
        .replace("__NODES_DATA__", nodes_json)
        .replace("__EDGES_DATA__", edges_json)
    )
    return html


def save_knowledge_graph(content: str, filepath: str = "knowledge_graph.html") -> bool:
    """Save interactive Vis.js HTML knowledge graph to a file."""
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True
    except Exception as err:
        sys.stderr.write(f"Error saving Knowledge Graph to {filepath}: {err}\n")
        return False


# =====================================================================
# CLI Presentation and User Interaction
# =====================================================================
def print_banner() -> None:
    banner = r"""
   _____ ______ _____ ______ _   _  _____ ______ 
  / ____/ ____|_   _|  ____| \ | |/ ____|  ____|
 | (___| |      | | | |__  |  \| | |     | |__   
  \___ \ |      | | |  __| | . ` | |     |  __|  
  ____) | |____ _| |_| |____| |\  | |____ | |____ 
 |_____/ \_____|_____|______|_| \_|\_____|______|
         E X P A N D E R   v 2 . 0
  Tri-Axial Scientific Literature Discovery Engine
    """
    print(color(banner, Style.BOLD, Style.CYAN))


def display_topics(topics: List[GeneratedTopic]) -> None:
    if HAS_RICH and rich_console:
        table = Table(
            title="GENOMICS OF IDEAS: 10 HIGH-AFFINITY INTERDISCIPLINARY TOPICS",
            box=box.ROUNDED,
            header_style="bold cyan",
            title_style="bold white",
            show_header=True,
            show_lines=True
        )
        table.add_column("#", style="bold green", justify="center", width=4)
        table.add_column("Topic & Conceptual Blend", style="bold white", min_width=30)
        table.add_column("Bridge Domain", style="magenta", min_width=18)
        table.add_column("Affinity", justify="center", min_width=10)
        table.add_column("Vector Affinity", justify="center", min_width=16)
        table.add_column("Methodological Lens", style="cyan", min_width=18)

        for t in topics:
            aff_pct = int(t.affinity * 100)
            if aff_pct >= 95:
                badge = f"[bold green]{aff_pct}% HIGH[/]"
            elif aff_pct >= 80:
                badge = f"[bold yellow]{aff_pct}% MED[/]"
            else:
                badge = f"[bold red]{aff_pct}% LOW[/]"

            if t.cosine_similarity is not None:
                sim_str = f"Sim: {t.cosine_similarity:.2f}"
                zone = t.vector_zone or "Vector"
                if zone == "Sweet Spot":
                    vec_badge = f"{sim_str}\n[bold green][{zone}][/]"
                elif zone in ("Moderate Overlap", "Distant Analogy"):
                    vec_badge = f"{sim_str}\n[bold yellow][{zone}][/]"
                else:
                    vec_badge = f"{sim_str}\n[bold red][{zone}][/]"
            else:
                vec_badge = "[dim]Rule-Based[/]"

            disp_p = t.display_primary_topic or t.primary_topic
            disp_s = t.display_secondary_topic or t.secondary_topic
            source_repr = f"{disp_p}"
            if disp_s:
                source_repr += f" × {disp_s}"

            topic_cell = f"[bold]{t.title}[/]\n[dim]{source_repr} ↔ {t.domain.name}[/]"
            bridge_cell = f"{t.domain.name}\n[dim]({t.domain.category_label})[/]"
            lens_cell = f"{t.operator_name}\n[dim]{t.rationale[:55]}...[/]"

            table.add_row(
                f"{t.index:02d}",
                topic_cell,
                bridge_cell,
                badge,
                vec_badge,
                lens_cell
            )

        print()
        rich_console.print(table)
        print()
    else:
        print(color("\n" + "=" * 80, Style.DIM))
        print(color("  GENOMICS OF IDEAS: 10 HIGH-AFFINITY INTERDISCIPLINARY TOPICS", Style.BOLD, Style.WHITE))
        print(color("=" * 80 + "\n", Style.DIM))

        for t in topics:
            idx_str = f"[{t.index:02d}]"
            print(f"{color(idx_str, Style.BOLD, Style.GREEN)} {color(t.title, Style.BOLD, Style.WHITE)}")
            
            disp_p = t.display_primary_topic or t.primary_topic
            disp_s = t.display_secondary_topic or t.secondary_topic
            source_repr = f"'{disp_p}'"
            if disp_s:
                source_repr += f" × '{disp_s}'"
            
            aff_pct = int(t.affinity * 100)
            aff_color = Style.GREEN if aff_pct >= 85 else Style.YELLOW
            vec_info = f"Vector: Sim {t.cosine_similarity:.2f} ({t.vector_zone})" if t.cosine_similarity is not None else "Rule-Based"
            print(f"     {color('Bridge:', Style.CYAN)} {source_repr} ↔ {color(t.domain.name, Style.YELLOW)}  [{color(f'Affinity: {aff_pct}%', aff_color)}] [{color(vec_info, Style.GREEN)}]")
            print(f"     {color('Lens  :', Style.MAGENTA)} {t.operator_name} | {color(t.rationale, Style.DIM)}")
            print(f"     {color('Query :', Style.DIM)} {t.primary_query}")
            print()


def display_triad(topic: GeneratedTopic, triad: LiteratureTriad) -> None:
    """Display the Tri-Axial literature triad (Foundation, Frontier, Review) with Rich panels & syntax highlighting."""
    if HAS_RICH and rich_console:
        header_text = (
            f"[bold white]Topic #{topic.index}: \"{topic.title}\"[/]\n"
            f"[dim cyan]Bridge:[/] [magenta]{topic.domain.name}[/] ({topic.domain.category_label})  |  "
            f"[dim cyan]Epistemic Lens:[/] [cyan]{topic.operator_name}[/]"
        )
        rich_console.print(Panel(
            header_text,
            title=f"[bold cyan]THE LITERATURE TRIAD FOR TOPIC #{topic.index}[/]",
            border_style="cyan",
            padding=(1, 2)
        ))

        sections = [
            ("🏛️  [1/3] THE FOUNDATION (Seminal Landmark Publication)", triad.foundation, "green"),
            ("⚡  [2/3] THE FRONTIER (2024–2026 Emerging Edge & Preprints)", triad.frontier, "yellow"),
            ("📚  [3/3] THE REVIEW (Comprehensive State-of-the-Art Survey)", triad.review, "magenta"),
        ]

        for title, paper, border_col in sections:
            if not paper:
                rich_console.print(Panel(
                    "[dim red]No indexed publication retrieved on this axis.[/]",
                    title=f"[bold {border_col}]{title}[/]",
                    border_style=border_col
                ))
                continue

            lines = []
            lines.append(f"[bold white]Title   :[/] [bold]{paper.title}[/]")
            lines.append(f"[dim cyan]Authors :[/] {', '.join(paper.authors) if paper.authors else 'Unknown'}")
            cit_str = f"[bold yellow]{paper.citations} citations[/]" if paper.citations is not None else "[dim]Citations unavailable[/]"
            lines.append(f"[bold white]Year    :[/] {paper.pub_year}   |   [bold white]Venue:[/] {paper.venue}   |   [bold white]Impact:[/] {cit_str}")
            lines.append(f"[dim cyan]Source  :[/] [cyan]{paper.source}[/]")
            if paper.doi:
                lines.append(f"[dim cyan]DOI     :[/] {paper.doi}")
            if paper.url:
                lines.append(f"[dim cyan]URL     :[/] [blue underline]{paper.url}[/]")
            if paper.oa_url:
                lines.append(f"[bold green]OA PDF  :[/] [bold green underline]{paper.oa_url}[/] [bold green]\\[Direct Open Access PDF][/]")
            if paper.abstract and paper.abstract != "No abstract snippet available.":
                snip = paper.abstract.strip()
                if len(snip) > 280:
                    snip = snip[:280] + "..."
                lines.append(f"[dim italic]\"{snip}\"[/]")

            rich_console.print(Panel(
                "\n".join(lines),
                title=f"[bold {border_col}]{title}[/]",
                border_style=border_col,
                padding=(0, 2)
            ))

        if triad.consolidated_bibtex:
            bib_syntax = Syntax(triad.consolidated_bibtex, "bibtex", theme="monokai", line_numbers=False)
            rich_console.print(Panel(
                bib_syntax,
                title="[bold white]CONSOLIDATED BIBTEX (Overleaf / Zotero Ready) - BibTeX Entry:[/]",
                border_style="dim white",
                padding=(1, 2)
            ))
    else:
        print(color("\n" + "=" * 80, Style.DIM))
        print(color(f"  THE LITERATURE TRIAD FOR TOPIC #{topic.index}", Style.BOLD, Style.CYAN))
        print(color(f"  \"{topic.title}\"", Style.BOLD, Style.WHITE))
        print(color("=" * 80, Style.DIM))

        sections = [
            ("🏛️  [1/3] THE FOUNDATION (Seminal Landmark Publication)", triad.foundation, Style.GREEN),
            ("⚡  [2/3] THE FRONTIER (2024–2026 Emerging Edge & Preprints)", triad.frontier, Style.YELLOW),
            ("📚  [3/3] THE REVIEW (Comprehensive State-of-the-Art Survey)", triad.review, Style.MAGENTA),
        ]

        for header, paper, hdr_color in sections:
            print(f"\n{color(header, Style.BOLD, hdr_color)}")
            if not paper:
                print(color("    [-] No publication indexed on this axis.", Style.DIM, Style.RED))
                continue
            print(f"    {color('Title   :', Style.BOLD, Style.WHITE)} {paper.title}")
            print(f"    {color('Authors :', Style.BOLD, Style.CYAN)} {', '.join(paper.authors) if paper.authors else 'Unknown'}")
            print(f"    {color('Year    :', Style.BOLD, Style.WHITE)} {paper.pub_year}   |   {color('Venue:', Style.BOLD, Style.WHITE)} {paper.venue}")
            cit_str = f"{paper.citations} citations" if paper.citations is not None else "Citation count unavailable"
            print(f"    {color('Impact  :', Style.BOLD, Style.YELLOW)} {cit_str}   |   {color('Source:', Style.BOLD, Style.CYAN)} {paper.source}")
            if paper.doi:
                print(f"    {color('DOI     :', Style.BOLD, Style.CYAN)} {paper.doi}")
            print(f"    {color('URL     :', Style.BOLD, Style.BLUE)} {color(paper.url or 'N/A', Style.UNDERLINE)}")
            if paper.oa_url:
                print(f"    {color('OA PDF  :', Style.BOLD, Style.GREEN)} {color(paper.oa_url, Style.UNDERLINE)} {color('[Direct Open Access PDF]', Style.GREEN)}")
            if paper.abstract and paper.abstract != "No abstract snippet available.":
                snippet = paper.abstract.strip()
                if len(snippet) > 260:
                    snippet = snippet[:260] + "..."
                print(f"    {color('Abstract:', Style.BOLD, Style.DIM)} \"{snippet}\"")

        if triad.consolidated_bibtex:
            print(color("\n" + "-" * 80, Style.DIM))
            print(color("  CONSOLIDATED BIBTEX (Overleaf / Zotero Ready) - BibTeX Entry:", Style.BOLD, Style.WHITE))
            print(color("-" * 80, Style.DIM))
            print(color(triad.consolidated_bibtex, Style.DIM, Style.CYAN))

        print("\n" + "=" * 80 + "\n")


def display_paper(topic: GeneratedTopic, paper: Optional[RetrievedPaper]) -> None:
    """Backwards-compatible wrapper around display_triad."""
    triad = LiteratureTriad(foundation=paper)
    display_triad(topic, triad)


def prompt_user_selection(topics: List[GeneratedTopic]) -> GeneratedTopic:
    """Interactively prompt user to select one topic from 1 to 10."""
    while True:
        try:
            choice = input(color(f"Select a topic to fetch real literature (1-{len(topics)}) [Default: 1]: ", Style.BOLD, Style.CYAN)).strip()
            if not choice:
                return topics[0]
            val = int(choice)
            if 1 <= val <= len(topics):
                return topics[val - 1]
            print(color(f"Please enter an integer between 1 and {len(topics)}.", Style.RED))
        except (ValueError, EOFError):
            print(color("Invalid input or non-interactive stream. Defaulting to Topic #1.", Style.YELLOW))
            return topics[0]


# =====================================================================
# Main Application Entrypoint
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Science Expander: Tri-axial literature retrieval and automated research brief generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python science_expander.py "Cell Biology"
  python science_expander.py -t "Cell Biology" --select 1 --report Cell_Biology_Brief.md
  python science_expander.py --demo --auto --bibtex
        """
    )
    parser.add_argument(
        "positional_topics",
        nargs="*",
        help="List of scientific topics as space-separated arguments."
    )
    parser.add_argument(
        "-t", "--topics",
        type=str,
        help="Comma-separated list of scientific topics."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a text file containing topics (one per line)."
    )
    parser.add_argument(
        "-s", "--select",
        type=int,
        help="Automatically select topic index (1-10) to fetch without interactive prompt."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive mode: automatically selects the top-ranked topic (#1)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with pre-configured cutting-edge scientific topics."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save output (JSON or text summary)."
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="Research_Brief.md",
        default=None,
        help="Generate and export a publication-ready Markdown Research Brief (default: Research_Brief.md)."
    )
    parser.add_argument(
        "--graph",
        nargs="?",
        const="knowledge_graph.html",
        default=None,
        help="Generate and export an interactive Vis.js HTML knowledge graph (default: knowledge_graph.html)."
    )
    parser.add_argument(
        "--bibtex",
        nargs="?",
        const="references.bib",
        default=None,
        help="Save/append retrieved paper's BibTeX citation to a .bib file (default: references.bib)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible cross-breeding generations."
    )

    args = parser.parse_args()

    if not args.json:
        print_banner()

    # Collect topics
    raw_topics: List[str] = []

    if args.demo:
        raw_topics = ["Cell Biology", "Quantum Metamaterials", "Neuroplasticity"]
        if not args.json:
            print(color(f"[*] Demo mode activated with topics: {', '.join(raw_topics)}\n", Style.GREEN))
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                raw_topics = [line.strip() for line in fh if line.strip()]
        except Exception as e:
            sys.stderr.write(color(f"Error reading file '{args.file}': {e}\n", Style.RED))
            sys.exit(1)
    elif args.topics:
        raw_topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif args.positional_topics:
        raw_topics = [t.strip() for t in args.positional_topics if t.strip()]
    else:
        # Interactive prompt if no arguments passed
        if not args.json:
            print(color("No topics specified via command line arguments.", Style.YELLOW))
        user_input = input(color("Enter scientific topic(s) separated by commas (or press Enter for demo): ", Style.CYAN)).strip()
        if user_input:
            raw_topics = [t.strip() for t in user_input.split(",") if t.strip()]
        else:
            raw_topics = ["Cell Biology", "Quantum Dots"]
            if not args.json:
                print(color(f"[*] Using curated seed topics: {', '.join(raw_topics)}\n", Style.GREEN))

    # Multilingual Bridge Translation & Bilingual Representation
    input_en_topics: List[str] = []
    bilingual_display_topics: List[str] = []
    bilingual_map: Dict[str, str] = {}

    for rt in raw_topics:
        if MultilingualBridge.is_cyrillic(rt):
            en_t = clean_topic_name(MultilingualBridge.translate_to_en(rt))
            bilingual_t = MultilingualBridge.format_bilingual(rt, en_t)
            input_en_topics.append(en_t)
            bilingual_display_topics.append(bilingual_t)
            bilingual_map[en_t] = bilingual_t
            bilingual_map[clean_topic_name(en_t)] = bilingual_t
            bilingual_map[rt] = bilingual_t
            bilingual_map[clean_topic_name(rt)] = bilingual_t
        else:
            input_en_topics.append(rt)
            bilingual_display_topics.append(rt)
            bilingual_map[rt] = rt
            bilingual_map[clean_topic_name(rt)] = rt

    if not args.json:
        print(color(f"Input Scientific Topics ({len(raw_topics)}):", Style.BOLD))
        for rt, bt, et in zip(raw_topics, bilingual_display_topics, input_en_topics):
            cat = classify_topic(et)
            cat_name = cat.value.replace("_", " ").title()
            if MultilingualBridge.is_cyrillic(rt):
                print(f"  • {color(bt, Style.CYAN)} [{color(f'Faculty: {cat_name}', Style.MAGENTA)}] {color(f'🌐 Multilingual Bridge: {rt} → {et}', Style.DIM)}")
            else:
                print(f"  • {color(bt, Style.CYAN)} [{color(f'Faculty: {cat_name}', Style.MAGENTA)}]")

    # Algorithmic Cross-Breeding with Domain Affinity
    breeder = TopicCrossBreeder(seed=args.seed)
    generated = breeder.cross_breed(input_en_topics, count=10)

    for g in generated:
        if not g.display_primary_topic or g.display_primary_topic == g.primary_topic:
            g.display_primary_topic = bilingual_map.get(g.primary_topic, g.primary_topic)
        if g.secondary_topic and (not g.display_secondary_topic or g.display_secondary_topic == g.secondary_topic):
            g.display_secondary_topic = bilingual_map.get(g.secondary_topic, g.secondary_topic)

    # Display 10 Generated Topics (if not json mode)
    if not args.json:
        display_topics(generated)

    # Topic Selection
    if args.auto or args.json or (args.select is not None and 1 <= args.select <= len(generated)):
        chosen_idx = 1 if (args.auto or args.json or args.select is None) else args.select
        selected_topic = generated[chosen_idx - 1]
        if not args.json:
            print(color(f"[*] Selected Topic #{selected_topic.index}: \"{selected_topic.title}\"\n", Style.BOLD, Style.GREEN))
    else:
        selected_topic = prompt_user_selection(generated)

    # Tri-Axial Literature Retrieval (Foundation, Frontier, Review)
    triad = ScholarLiteratureClient.fetch_triad(selected_topic, silent=args.json)
    paper = triad.foundation or (triad.papers[0] if triad.papers else None)

    # Generate Markdown Research Brief & Knowledge Graph
    brief_content = generate_research_brief(selected_topic, triad, generated, bilingual_display_topics)
    graph_content = generate_knowledge_graph(selected_topic, triad, generated, bilingual_display_topics)

    # Handle JSON output
    if args.json:
        result_payload = {
            "input_topics": raw_topics,
            "bilingual_topics": bilingual_display_topics,
            "generated_topics": [
                {
                    "index": t.index,
                    "title": t.title,
                    "primary_topic": t.primary_topic,
                    "display_primary_topic": t.display_primary_topic or t.primary_topic,
                    "secondary_topic": t.secondary_topic,
                    "display_secondary_topic": t.display_secondary_topic or t.secondary_topic,
                    "domain": t.domain.name,
                    "category": t.domain.category_label,
                    "lens": t.operator_name,
                    "rationale": t.rationale,
                    "affinity": t.affinity,
                    "cosine_similarity": t.cosine_similarity,
                    "goldilocks_score": t.goldilocks_score,
                    "vector_zone": t.vector_zone,
                    "primary_query": t.primary_query,
                    "fallback_query": t.fallback_query
                }
                for t in generated
            ],
            "selected_topic_index": selected_topic.index,
            "retrieved_paper": asdict(paper) if paper else None,
            "literature_triad": {
                "foundation": asdict(triad.foundation) if triad.foundation else None,
                "frontier": asdict(triad.frontier) if triad.frontier else None,
                "review": asdict(triad.review) if triad.review else None
            }
        }
        json_str = json.dumps(result_payload, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as out_f:
                out_f.write(json_str)
        else:
            print(json_str)
        if args.bibtex and triad.consolidated_bibtex:
            save_bibtex(triad.consolidated_bibtex, args.bibtex)
        if args.report:
            save_research_brief(brief_content, args.report)
        if args.graph:
            save_knowledge_graph(graph_content, args.graph)
        return

    # Display Publication Triad
    display_triad(selected_topic, triad)

    # Save Research Brief if requested via CLI flag
    if args.report:
        if save_research_brief(brief_content, args.report):
            print(color(f"[✓] Research Brief successfully saved to: {args.report}\n", Style.GREEN))

    # Save Knowledge Graph if requested via CLI flag
    if args.graph:
        if save_knowledge_graph(graph_content, args.graph):
            print(color(f"[✓] Interactive Knowledge Graph successfully saved to: {args.graph}\n", Style.GREEN))

    # Save BibTeX if requested via CLI flag
    if args.bibtex and triad.consolidated_bibtex:
        if save_bibtex(triad.consolidated_bibtex, args.bibtex):
            print(color(f"[✓] Consolidated BibTeX entries saved to: {args.bibtex}\n", Style.GREEN))

    # Interactive action prompt only if running interactively without explicit action flags
    is_interactive_cli = (
        sys.stdin.isatty() and
        not args.auto and
        args.select is None and
        not args.report and
        not args.graph and
        not args.bibtex
    )
    if is_interactive_cli:
        try:
            action = input(color("Actions: [R] Export Research Brief (Markdown) | [G] Export Knowledge Graph (HTML) | [B] Save BibTeX to references.bib | [Enter] Exit: ", Style.BOLD, Style.CYAN)).strip().lower()
            if action in {"r", "report"}:
                rep_path = "Research_Brief.md"
                if save_research_brief(brief_content, rep_path):
                    print(color(f"[✓] Research Brief successfully generated and saved to: {rep_path}\n", Style.GREEN))
            elif action in {"g", "graph"}:
                graph_path = "knowledge_graph.html"
                if save_knowledge_graph(graph_content, graph_path):
                    print(color(f"[✓] Interactive Knowledge Graph successfully generated and saved to: {graph_path}\n", Style.GREEN))
            elif action in {"b", "bibtex"}:
                bib_path = "references.bib"
                if save_bibtex(triad.consolidated_bibtex, bib_path):
                    print(color(f"[✓] BibTeX entries appended to: {bib_path}\n", Style.GREEN))
        except (EOFError, KeyboardInterrupt):
            pass

    # Optional file output in text mode
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as out_f:
                out_f.write(f"SCIENCE EXPANDER RESULTS\n\nInput Topics:\n")
                for t in raw_topics:
                    out_f.write(f"- {t} ({classify_topic(t).value})\n")
                out_f.write("\nGenerated 10 Interdisciplinary Topics:\n")
                for t in generated:
                    out_f.write(f"[{t.index:02d}] {t.title}\n    Bridge: {t.domain.name} (Affinity: {int(t.affinity*100)}%)\n    Lens: {t.operator_name}\n\n")
                if paper:
                    out_f.write(f"\nRetrieved Literature for Topic #{selected_topic.index}:\n")
                    out_f.write(f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.pub_year}\nVenue: {paper.venue}\nURL: {paper.url}\nCitations: {paper.citations}\nSource: {paper.source}\n\nAbstract:\n{paper.abstract}\n\nBibTeX:\n{paper.bibtex}\n")
            print(color(f"[✓] Output saved to: {args.output}\n", Style.GREEN))
        except Exception as e:
            sys.stderr.write(color(f"Error saving to file '{args.output}': {e}\n", Style.RED))


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as se:
        exit_code = se.code if isinstance(se.code, int) else (0 if se.code is None else 1)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)

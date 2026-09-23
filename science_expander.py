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

import asyncio
import aiohttp
import math
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


class DiscoveryTier(str, Enum):
    """Classifies ontology domains by epistemic distance from the seed topic."""
    ADJACENT = "adjacent"            # High-yield intra-domain sub-field crossovers (60%)
    TRANSLATIONAL = "translational"  # Applied bioengineering / methodology bridges (25%)
    FRONTIER = "frontier"            # Deep fundamental physics/math leaps (15%)


class DiscoveryMode(str, Enum):
    """User-selectable discovery mode controlling tier distribution."""
    BALANCED = "balanced"           # 60% adjacent, 25% translational, 15% frontier
    ADJACENT = "adjacent"           # Prioritize sub-field crossovers
    TRANSLATIONAL = "translational" # Prioritize applied engineering
    FRONTIER = "frontier"           # Prioritize deep fundamental leaps (legacy behavior)


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

# =====================================================================
# Dynamic Concept Miner (OpenAlex Async Pipeline)
# =====================================================================
class DynamicConceptMiner:
    @classmethod
    async def fetch_seed_concepts(cls, seed_topic: str) -> Dict[str, dict]:
        """
        Query OpenAlex API for top highly cited & recent works matching seed_topic.
        Extract concepts (levels 2,3,4) and compute frequencies.
        """
        url = "https://api.openalex.org/works"
        
        # We query for works matching the seed topic in title/abstract
        params_cited = {
            "search": seed_topic,
            "sort": "cited_by_count:desc",
            "per-page": 100,
            "mailto": "test@example.com"
        }
        params_recent = {
            "search": seed_topic,
            "sort": "publication_date:desc",
            "per-page": 100,
            "mailto": "test@example.com"
        }
        
        async def fetch_json(u, s, p):
            async with s.get(u, params=p) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

        async with aiohttp.ClientSession() as session:
            # Concurrently fetch both highly cited and recent
            tasks = [
                fetch_json(url, session, params_cited),
                fetch_json(url, session, params_recent)
            ]
            responses = await asyncio.gather(*tasks)
            
            concept_counts = {}
            for data in responses:
                for work in data.get("results", []):
                        for concept in work.get("concepts", []):
                            lvl = concept.get("level", 0)
                            if lvl in (2, 3, 4):
                                c_id = concept["id"]
                                if c_id not in concept_counts:
                                    concept_counts[c_id] = {
                                        "id": c_id,
                                        "display_name": concept["display_name"],
                                        "level": lvl,
                                        "count": 0
                                    }
                                concept_counts[c_id]["count"] += 1
                                
        return concept_counts

# =====================================================================
# Conceptual Blending Models & Generator
# =====================================================================
@dataclass
class GeneratedTopic:
    index: int
    title: str
    primary_topic: str
    secondary_topic: Optional[str]
    concept_id: str
    concept_name: str
    concept_level: int
    bimodal_score: float
    tier: DiscoveryTier
    operator_name: str
    rationale: str
    display_primary_topic: Optional[str] = None
    display_secondary_topic: Optional[str] = None
    cosine_similarity: Optional[float] = None
    goldilocks_score: Optional[float] = None
    vector_zone: Optional[str] = None
    anchor_phrase: Optional[str] = None
    anchor_terms: Optional[List[str]] = None
    target_query: str = ""


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
        if not self.is_available or self.model is None or np is None:
            return None
        cleaned = text.strip().lower()
        if not cleaned: return None
        if cleaned in self._cache: return self._cache[cleaned]
        try:
            generator = self.model.embed([cleaned])
            vec = next(generator)
            arr = np.array(vec, dtype=np.float32)
            self._cache[cleaned] = arr
            return arr
        except Exception:
            return None

    async def embed_async(self, text: str) -> Optional[Any]:
        return await asyncio.to_thread(self.embed, text)

    @staticmethod
    def cosine_similarity(u: Optional[Any], v: Optional[Any]) -> float:
        if u is None or v is None or np is None: return 0.0
        try:
            norm_u = float(np.linalg.norm(u))
            norm_v = float(np.linalg.norm(v))
            if norm_u == 0.0 or norm_v == 0.0: return 0.0
            return float(np.dot(u, v) / (norm_u * norm_v))
        except Exception:
            return 0.0
            
    async def cosine_similarity_async(self, u: Optional[Any], v: Optional[Any]) -> float:
        return await asyncio.to_thread(self.cosine_similarity, u, v)

    @staticmethod
    def calculate_goldilocks_score(similarity: float) -> float:
        s = max(-1.0, min(1.0, similarity))
        if 0.40 <= s <= 0.60: return 1.0
        elif 0.35 <= s < 0.40: return round(0.90 + (s - 0.35) * (0.10 / 0.05), 4)
        elif 0.60 < s <= 0.65: return round(1.0 - (s - 0.60) * (0.10 / 0.05), 4)
        elif 0.20 <= s < 0.35: return round(0.30 + (s - 0.20) * (0.60 / 0.15), 4)
        elif 0.65 < s <= 0.80: return round(0.90 - (s - 0.65) * (0.50 / 0.15), 4)
        elif s > 0.80: return round(max(0.05, 0.40 - (s - 0.80) * (0.35 / 0.20)), 4)
        else:
            if s <= 0.0: return 0.0
            return round(s * (0.30 / 0.20), 4)

    async def compute_bimodal_scores(self, seed_topic: str, concept_counts: Dict[str, dict]) -> List[dict]:
        vec_seed = await self.embed_async(seed_topic)
        if vec_seed is None:
            return []
            
        counts = [c["count"] for c in concept_counts.values()]
        if not counts: return []
        
        import math
        log_counts = [math.log(1 + c) for c in counts]
        min_c = min(log_counts)
        max_c = max(log_counts)
        range_c = max_c - min_c if max_c > min_c else 1.0
        
        scored = []
        for c in concept_counts.values():
            freq_weight = (math.log(1 + c["count"]) - min_c) / range_c
            vec_concept = await self.embed_async(c["display_name"])
            cos_sim = await self.cosine_similarity_async(vec_seed, vec_concept)
            
            tier = None
            if 0.65 <= cos_sim < 0.85: tier = DiscoveryTier.ADJACENT
            elif 0.45 <= cos_sim < 0.65: tier = DiscoveryTier.TRANSLATIONAL
            elif 0.20 <= cos_sim < 0.45: tier = DiscoveryTier.FRONTIER
            else: continue
            
            g_score = self.calculate_goldilocks_score(cos_sim)
            bimodal_score = (0.4 * freq_weight) + (0.6 * g_score)
            
            c["cos_sim"] = cos_sim
            c["bimodal_score"] = bimodal_score
            c["tier"] = tier
            scored.append(c)
            
        scored.sort(key=lambda x: x["bimodal_score"], reverse=True)
        return scored



class TopicCrossBreeder:
    """
    Algorithmic cross-breeding engine with Data-Driven OpenAlex Concepts and Local Vector Intelligence.
    """

    def __init__(self, seed: Optional[int] = None, vector_engine: Optional[SemanticCorridorEngine] = None, mode: str = DiscoveryMode.BALANCED):
        self.rng = random.Random(seed)
        self.vector_engine = vector_engine or SemanticCorridorEngine.get_instance()
        self.mode = mode

    async def cross_breed_async(self, topics: List[str], count: int = 10) -> List[GeneratedTopic]:
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

        candidates: List[GeneratedTopic] = []

        composite_topic = " and ".join([format_topic_for_title(t) for t in resolved_topics])
        openalex_query = " AND ".join([f'"{format_topic_for_title(t)}"' for t in resolved_topics])
        
        # Fetch dynamic concepts for composite query
        concept_counts = await DynamicConceptMiner.fetch_seed_concepts(openalex_query)
        scored_concepts = await self.vector_engine.compute_bimodal_scores(composite_topic, concept_counts)
        
        is_multi = len(resolved_topics) > 1
        
        for concept in scored_concepts:
            c_name = concept["display_name"]
            c_id = concept["id"]
            tier = concept["tier"]
            bimodal = concept["bimodal_score"]

            if is_multi:
                titles = [
                    f"Integrating {composite_topic}: The Role of {c_name}",
                    f"Bridging {composite_topic} via {c_name}",
                    f"{c_name} Dynamics at the Intersection of {composite_topic}"
                ]
                rat = f"Explores the intersection of {composite_topic} focusing on the shared concept of {c_name}."
                op_name = "Composite Topic Synthesis"
            else:
                fmt_topic = format_topic_for_title(resolved_topics[0])
                if tier == DiscoveryTier.ADJACENT:
                    titles = [f"{c_name} in {fmt_topic}: Key Mechanisms", f"Targeting {c_name} in {fmt_topic}"]
                    rat = f"Investigates {fmt_topic} focusing on the dynamic adjacent concept of {c_name}."
                    op_name = "Adjacent Concept Intersection"
                elif tier == DiscoveryTier.TRANSLATIONAL:
                    titles = [f"Engineering {fmt_topic} with {c_name}", f"Translational Strategies: {c_name} in {fmt_topic}"]
                    rat = f"Translates findings in {fmt_topic} using {c_name}."
                    op_name = "Translational Concept Approach"
                else:
                    titles = [f"{fmt_topic} Through the Lens of {c_name}", f"Frontier Dynamics: {c_name} and {fmt_topic}"]
                    rat = f"Explores fundamental frontier concepts of {c_name} applied to {fmt_topic}."
                    op_name = "Frontier Epistemic Leap"

            title = self.rng.choice(titles)
            target_query = f'{openalex_query} AND "{c_name}"'
            
            candidates.append(GeneratedTopic(
                index=0,
                title=title,
                primary_topic=composite_topic,
                secondary_topic=None,
                concept_id=c_id,
                concept_name=c_name,
                concept_level=concept["level"],
                bimodal_score=bimodal,
                tier=tier,
                operator_name=op_name,
                rationale=rat,
                target_query=target_query
            ))

        # Enforce Quota Requirements
        tier_groups = {
            DiscoveryTier.ADJACENT: [],
            DiscoveryTier.TRANSLATIONAL: [],
            DiscoveryTier.FRONTIER: []
        }
        for cand in candidates:
            tier_groups[cand.tier].append(cand)
        
        for k in tier_groups:
            tier_groups[k].sort(key=lambda c: c.bimodal_score, reverse=True)
            
        if self.mode == DiscoveryMode.BALANCED:
            quotas = {DiscoveryTier.ADJACENT: 6, DiscoveryTier.TRANSLATIONAL: 3, DiscoveryTier.FRONTIER: 1}
        elif self.mode == DiscoveryMode.ADJACENT:
            quotas = {DiscoveryTier.ADJACENT: count, DiscoveryTier.TRANSLATIONAL: 0, DiscoveryTier.FRONTIER: 0}
        elif self.mode == DiscoveryMode.TRANSLATIONAL:
            quotas = {DiscoveryTier.ADJACENT: 0, DiscoveryTier.TRANSLATIONAL: count, DiscoveryTier.FRONTIER: 0}
        elif self.mode == DiscoveryMode.FRONTIER:
            quotas = {DiscoveryTier.ADJACENT: 0, DiscoveryTier.TRANSLATIONAL: 0, DiscoveryTier.FRONTIER: count}
        else:
            quotas = {DiscoveryTier.ADJACENT: count, DiscoveryTier.TRANSLATIONAL: 0, DiscoveryTier.FRONTIER: 0}
            
        final_list = []
        pool = []
        
        # Pull by quota
        for tier, q in quotas.items():
            pulled = tier_groups[tier][:q]
            final_list.extend(pulled)
            # leftover into pool
            pool.extend(tier_groups[tier][q:])
            
        # Backfill if not enough
        shortfall = count - len(final_list)
        if shortfall > 0:
            pool.sort(key=lambda c: c.bimodal_score, reverse=True)
            final_list.extend(pool[:shortfall])
            
        final_list.sort(key=lambda c: c.bimodal_score, reverse=True)
        # Ensure we don't exceed count if pool is huge
        final_list = final_list[:count]
        
        for i, cand in enumerate(final_list):
            cand.index = i + 1
        return final_list

    def cross_breed(self, topics: List[str], count: int = 10) -> List[GeneratedTopic]:
        return asyncio.run(self.cross_breed_async(topics, count))


@dataclass
class RetrievedPaper:
    title: str
    authors: List[str]
    pub_year: Optional[str]
    venue: Optional[str]
    citations: Optional[int]
    abstract: Optional[str]
    url: Optional[str]
    query_used: str = ""
    is_fallback: bool = False
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
    Tri-Axial Academic Literature Retrieval Engine (Async OpenAlex).
    Coordinates OpenAlex Concept ID queries across three axes: Foundation, Frontier, Review.
    Provides granular axis-specific Scholar fallback.
    """

    @classmethod
    async def fetch_triad_async(cls, topic: GeneratedTopic, silent: bool = False) -> LiteratureTriad:
        if not silent and HAS_RICH and rich_console:
            rich_console.print(f"[dim]Initiating Async OpenAlex Data-Driven Retrieval for {topic.concept_name}...[/dim]")
            
        c_id = topic.concept_id.split("/")[-1] if "/" in topic.concept_id else topic.concept_id
        
        async with aiohttp.ClientSession() as session:
            # The base target_query from our multi-topic synthesis
            base_search = topic.target_query if topic.target_query else topic.primary_topic

            # 1. Foundation: Highly cited
            url = "https://api.openalex.org/works"
            params_foundation = {
                "filter": f"concepts.id:{c_id}",
                "search": base_search,
                "sort": "cited_by_count:desc",
                "per-page": 5,
                "mailto": "test@example.com"
            }
            
            # 2. Frontier: Recent
            params_frontier = {
                "filter": f"concepts.id:{c_id},from_publication_date:2024-01-01",
                "search": base_search,
                "sort": "publication_date:desc",
                "per-page": 5,
                "mailto": "test@example.com"
            }
            
            # 3. Review: type:review
            params_review = {
                "filter": f"concepts.id:{c_id},type:review",
                "search": base_search,
                "sort": "cited_by_count:desc",
                "per-page": 5,
                "mailto": "test@example.com"
            }
            
            async def fetch_paper(u, s, p):
                async with s.get(u, params=p) as resp:
                    if resp.status == 200:
                        return await cls._parse_response_data(await resp.json())
                    return None

            tasks = [
                fetch_paper(url, session, params_foundation),
                fetch_paper(url, session, params_frontier),
                fetch_paper(url, session, params_review)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            foundation = responses[0] if not isinstance(responses[0], Exception) else None
            frontier = responses[1] if not isinstance(responses[1], Exception) else None
            review = responses[2] if not isinstance(responses[2], Exception) else None

            # Fallback for Review if type:review fails
            if not review:
                # Text-based fallback
                query_fallback = f'{base_search} AND ("review" OR "survey" OR "meta-analysis" OR "progress in" OR "advances in")'
                params_review_fallback = {
                    "filter": f"concepts.id:{c_id}",
                    "search": query_fallback,
                    "sort": "cited_by_count:desc",
                    "per-page": 5,
                    "mailto": "test@example.com"
                }
                review = await fetch_paper(url, session, params_review_fallback)
                
            # Granular Scholar Fallback
            if not foundation:
                foundation = await asyncio.to_thread(cls._scholar_fallback, topic.primary_topic)
            if not frontier:
                frontier = await asyncio.to_thread(cls._scholar_fallback, f"{topic.primary_topic} AND {topic.concept_name}", year_low=2024)
            if not review:
                review = await asyncio.to_thread(cls._scholar_fallback, f"{topic.primary_topic} AND (review OR survey)")
                
            return LiteratureTriad(
                foundation=foundation,
                frontier=frontier,
                review=review
            )
            
    @classmethod
    async def _parse_response_data(cls, data: dict) -> Optional[RetrievedPaper]:
        results = data.get("results", [])
        if results:
            work = results[0]
            authors = [a["author"]["display_name"] for a in work.get("authorships", [])]
            venue = work.get("primary_location", {}).get("source", {})
            venue_name = venue.get("display_name") if venue else "Unknown"
            return RetrievedPaper(
                title=work.get("title", "Unknown Title"),
                authors=authors,
                venue=venue_name,
                pub_year=str(work.get("publication_year", "")),
                citations=work.get("cited_by_count", 0),
                url=work.get("id"),
                abstract=reconstruct_abstract(work.get("abstract_inverted_index"))
            )
        return None

    @classmethod
    def _scholar_fallback(cls, query: str, year_low: Optional[int] = None) -> Optional[RetrievedPaper]:
        try:
            search_query = scholarly.search_pubs(query)
            for i in range(3):
                try:
                    pub = next(search_query)
                    bib = pub.get('bib', {})
                    if year_low and bib.get('pub_year'):
                        try:
                            if int(bib.get('pub_year')) < year_low:
                                continue
                        except ValueError:
                            pass
                    return RetrievedPaper(
                        title=bib.get('title', 'Unknown'),
                        authors=[bib.get('author', 'Unknown')],
                        venue=bib.get('venue', 'Unknown'),
                        pub_year=bib.get('pub_year', ''),
                        citations=pub.get('num_citations', 0),
                        url=pub.get('pub_url'),
                        abstract=bib.get('abstract')
                    )
                except StopIteration:
                    break
        except Exception:
            pass
        return None

    @classmethod
    def fetch_triad(cls, topic: GeneratedTopic, silent: bool = False) -> LiteratureTriad:
        return asyncio.run(cls.fetch_triad_async(topic, silent))

def generate_research_brief(topic: GeneratedTopic, triad: LiteratureTriad) -> str:
    # Just a placeholder implementation to avoid parsing errors in the string if it relied on domain.name
    # Since we replaced domain with concept_name
    lines = [
        f"# Research Brief: {topic.title}",
        f"**Primary Topic**: {topic.primary_topic}",
        f"**Dynamic Concept**: {topic.concept_name} (ID: {topic.concept_id}, Tier: {topic.tier})",
        f"**Bimodal Score**: {topic.bimodal_score}",
        f"**Rationale**: {topic.rationale}",
        "",
        "## Literature Triad",
    ]
    if triad.foundation:
        lines.append(f"### Foundation\\n- {triad.foundation.title}")
    if triad.frontier:
        lines.append(f"### Frontier\\n- {triad.frontier.title}")
    if triad.review:
        lines.append(f"### Review\\n- {triad.review.title}")
        
    return "\\n".join(lines)



def generate_research_brief(
    topic: GeneratedTopic,
    triad: LiteratureTriad,
    all_topics: List[GeneratedTopic],
    raw_topics: List[str]
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds_str = ", ".join(raw_topics)
    cat_name = classify_topic(topic.primary_topic).value.replace("_", " ").title()

    table_rows = []
    for t in all_topics:
        sel_mark = " **(Selected)**" if t.index == topic.index else ""
        row = f"| {t.index:02d} | {t.title}{sel_mark} | {t.concept_name} | {t.bimodal_score:.2f} | {t.operator_name} |"
        table_rows.append(row)
    landscape_table = "\\n".join(table_rows)

    def format_paper_section(p: Optional[RetrievedPaper]) -> str:
        if not p:
            return "*No indexed publication retrieved for this axis.*"
        md = [
            f"- **Title**: {p.title}",
            f"- **Authors**: {', '.join(p.authors) if p.authors else 'Unknown'}",
            f"- **Year**: {p.pub_year} | **Venue**: {p.venue} | **Citations**: {p.citations if p.citations is not None else 'N/A'}",
            f"- **URL**: {p.url if p.url else 'N/A'}",
        ]
        if p.abstract:
            md.append(f"\\n> **Abstract**: {p.abstract}\\n")
        return "\\n".join(md)

    markdown = f"""# Science Expander: Dynamic Research Brief
**Generated On**: {now_str}  
**Seed Topic(s)**: {seeds_str}  
**Primary Science Category**: {cat_name}

---

## 1. Executive Summary: The Selected Synthesis
**Topic #{topic.index:02d}: {topic.title}**

- **Primary Seed Domain**: {topic.primary_topic}
- **Dynamic Concept**: {topic.concept_name} (ID: {topic.concept_id})
- **Discovery Tier**: {topic.tier.value.upper() if topic.tier else 'N/A'}
- **Bimodal Score**: {topic.bimodal_score:.2f}
- **Epistemic Operator**: {topic.operator_name}

**Algorithmic Rationale**:  
{topic.rationale}

---

## 2. The Tri-Axial Literature Foundation

### Axis I: The Foundation (Seminal / Highly Cited)
{format_paper_section(triad.foundation)}

### Axis II: The Frontier (Recent / Cutting-Edge)
{format_paper_section(triad.frontier)}

### Axis III: The Review (State-of-the-Art Survey)
{format_paper_section(triad.review)}

---

## 3. Alternative Conceptual Corridors (Generated Landscape)
| # | Topic Title | Dynamic Concept | Score | Epistemic Operator |
|---|---|---|---|---|
{landscape_table}

---
*Generated autonomously by Science Expander.*
"""
    return markdown

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
        if t.concept_name not in unique_domains:
            unique_domains[t.concept_name] = t.concept_name

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
                "faculty": "Dynamic Ontology",
                "methods": "Data-Driven Conceptual Extraction",
                "phenomena": "Semantic Corridor Traversal",
                "description": "Legitimate interdisciplinary scientific field mined dynamically from recent OpenAlex citation graphs."
            }
        })

    # 3. Idea Nodes (10 Generated Research Topics) - Emerald Green (#10b981 / #00f59b)
    for t in all_topics:
        iid = f"idea_{t.index}"
        is_sel = (t.index == topic.index)
        aff_pct = int(t.bimodal_score * 100)
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
                "bridge": t.concept_name,
                "faculty": "Dynamic Ontology",
                "affinity": aff_badge,
                "lens": t.operator_name,
                "rationale": t.rationale,
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
        b_id = bridge_ids[t.concept_name]
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
        b_id = bridge_ids[t.concept_name]
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
            aff_pct = int(t.bimodal_score * 100)
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

            topic_cell = f"[bold]{t.title}[/]\n[dim]{source_repr} ↔ {t.concept_name}[/]"
            bridge_cell = f"{t.concept_name}\n[dim]({"Dynamic Ontology"})[/]"
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
            
            aff_pct = int(t.bimodal_score * 100)
            aff_color = Style.GREEN if aff_pct >= 85 else Style.YELLOW
            vec_info = f"Vector: Sim {t.cosine_similarity:.2f} ({t.vector_zone})" if t.cosine_similarity is not None else "Rule-Based"
            print(f"     {color('Bridge:', Style.CYAN)} {source_repr} ↔ {color(t.concept_name, Style.YELLOW)}  [{color(f'Affinity: {aff_pct}%', aff_color)}] [{color(vec_info, Style.GREEN)}]")
            print(f"     {color('Lens  :', Style.MAGENTA)} {t.operator_name} | {color(t.rationale, Style.DIM)}")
            print()


def display_triad(topic: GeneratedTopic, triad: LiteratureTriad) -> None:
    """Display the Tri-Axial literature triad (Foundation, Frontier, Review) with Rich panels & syntax highlighting."""
    if HAS_RICH and rich_console:
        header_text = (
            f"[bold white]Topic #{topic.index}: \"{topic.title}\"[/]\n"
            f"[dim cyan]Bridge:[/] [magenta]{topic.concept_name}[/] ({"Dynamic Ontology"})  |  "
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
async def main_async() -> None:
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["balanced", "adjacent", "translational", "frontier"],
        default="balanced",
        help="Discovery mode: balanced (default), adjacent, translational, or frontier."
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
    breeder = TopicCrossBreeder(seed=args.seed, mode=args.mode)
    generated = await breeder.cross_breed_async(input_en_topics, count=10)

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
    triad = await ScholarLiteratureClient.fetch_triad_async(selected_topic, silent=args.json)
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
                    "domain": t.concept_name,
                    "category": "Dynamic Ontology",
                    "lens": t.operator_name,
                    "rationale": t.rationale,
                    "affinity": t.bimodal_score,
                    "cosine_similarity": t.cosine_similarity,
                    "goldilocks_score": t.goldilocks_score,
                    "vector_zone": t.vector_zone
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
                    out_f.write(f"[{t.index:02d}] {t.title}\n    Bridge: {t.concept_name} (Affinity: {int(t.bimodal_score*100)}%)\n    Lens: {t.operator_name}\n\n")
                if paper:
                    out_f.write(f"\nRetrieved Literature for Topic #{selected_topic.index}:\n")
                    out_f.write(f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.pub_year}\nVenue: {paper.venue}\nURL: {paper.url}\nCitations: {paper.citations}\nSource: {paper.source}\n\nAbstract:\n{paper.abstract}\n\nBibTeX:\n{paper.bibtex}\n")
            print(color(f"[✓] Output saved to: {args.output}\n", Style.GREEN))
        except Exception as e:
            sys.stderr.write(color(f"Error saving to file '{args.output}': {e}\n", Style.RED))


if __name__ == "__main__":
    exit_code = 0
    try:
        asyncio.run(main_async())
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

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

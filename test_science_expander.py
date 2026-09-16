#!/usr/bin/env python3
"""
Automated Test Suite for Science Expander.
Covers:
1. Scientific Taxonomy Classification & Affinity Matrix
2. Domain Sanitization (no pseudo-scientific DL buzzwords)
3. Algorithmic Cross-Breeding (1, 2, and 3 topics)
4. Diversity & Grammar Invariants
5. Scholarly Literature Fetching & Google Scholar Integration
6. OpenAlex Resilient Fallback & Open Access PDF extraction
7. BibTeX Formatting and File Export
8. CLI Execution (interactive flags, auto, JSON schema, bibtex flag)
"""

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from science_expander import (
    TopicCrossBreeder,
    ScholarLiteratureClient,
    OpenAlexClient,
    ScienceCategory,
    RetrievedPaper,
    LiteratureTriad,
    MultilingualBridge,
    classify_topic,
    clean_topic_name,
    extract_core_keywords,
    format_topic_for_title,
    generate_bibtex,
    generate_research_brief,
    generate_knowledge_graph,
    save_bibtex,
    save_research_brief,
    save_knowledge_graph,
    is_same_paper,
    DOMAINS,
    SemanticCorridorEngine,
)


class TestTaxonomyAndAffinity(unittest.TestCase):
    """Verifies scientific topic classification and domain affinity invariants."""

    def test_topic_classification(self):
        """Verify accurate categorization across primary science disciplines."""
        self.assertEqual(classify_topic("cell biology"), ScienceCategory.BIOLOGY)
        self.assertEqual(classify_topic("CRISPR gene editing"), ScienceCategory.BIOLOGY)
        self.assertEqual(classify_topic("superconductivity"), ScienceCategory.PHYSICS)
        self.assertEqual(classify_topic("quantum optics"), ScienceCategory.PHYSICS)
        self.assertEqual(classify_topic("graphene metamaterials"), ScienceCategory.MATERIALS)
        self.assertEqual(classify_topic("differential geometry"), ScienceCategory.MATHEMATICS)
        self.assertEqual(classify_topic("distributed consensus algorithms"), ScienceCategory.COMPUTER_SCIENCE)
        self.assertEqual(classify_topic("behavioral economics"), ScienceCategory.SOCIAL_COGNITIVE)

    def test_cell_biology_affinity_and_sanitization(self):
        """Verify cell biology receives high-affinity scientific domains and 0 DL buzzwords."""
        breeder = TopicCrossBreeder(seed=42)
        generated = breeder.cross_breed(["cell biology"], count=10)

        # Forbidden pseudo-scientific/DL buzzwords
        banned_terms = [
            "grokking", "delayed generalization", "causal faithfulness",
            "monosemantic", "induction head", "polysemanticity",
            "activation patching", "representation engineering"
        ]

        for t in generated:
            # 1. High biological affinity
            self.assertGreaterEqual(t.affinity, 0.80, f"Domain '{t.domain.name}' has insufficient affinity for biology: {t.affinity}")
            
            # 2. No pseudo-scientific buzzwords in title or rationale
            text_corpus = f"{t.title} {t.rationale}".lower()
            for banned in banned_terms:
                self.assertNotIn(banned, text_corpus, f"Found prohibited buzzword '{banned}' in: {t.title}")


class TestMultilingualBridge(unittest.TestCase):
    """Verifies Zero-Token Multilingual Bridge (RU <-> EN) and bilingual formatting."""

    def test_cyrillic_detection(self):
        self.assertTrue(MultilingualBridge.is_cyrillic("Клеточная биология"))
        self.assertTrue(MultilingualBridge.is_cyrillic("Квантовые точки"))
        self.assertTrue(MultilingualBridge.is_cyrillic("CRISPR редактирование генома"))
        self.assertFalse(MultilingualBridge.is_cyrillic("Cell Biology"))
        self.assertFalse(MultilingualBridge.is_cyrillic("Quantum Dots"))
        self.assertFalse(MultilingualBridge.is_cyrillic(""))

    def test_translation_to_english(self):
        en_topic = MultilingualBridge.translate_to_en("Клеточная биология")
        self.assertIn(en_topic.lower(), ["cell biology", "cellular biology"])

        en_dots = MultilingualBridge.translate_to_en("Квантовые точки")
        self.assertIn("quantum", en_dots.lower())

    def test_format_bilingual(self):
        bilingual = MultilingualBridge.format_bilingual("Клеточная биология", "Cell Biology")
        self.assertEqual(bilingual, "Клеточная биология (Cell Biology)")

        en_only = MultilingualBridge.format_bilingual("Cell Biology", "Cell Biology")
        self.assertEqual(en_only, "Cell Biology")

    def test_classify_cyrillic_topic(self):
        cat = classify_topic("Клеточная биология")
        self.assertEqual(cat, ScienceCategory.BIOLOGY)

        cat_phys = classify_topic("Квантовые метаматериалы")
        self.assertEqual(cat_phys, ScienceCategory.PHYSICS)

    def test_breeder_with_cyrillic_inputs(self):
        breeder = TopicCrossBreeder(seed=42)
        generated = breeder.cross_breed(["Клеточная биология"], count=10)
        self.assertEqual(len(generated), 10)
        for t in generated:
            self.assertGreaterEqual(t.affinity, 0.80)
            self.assertIn("Клеточная биология", t.display_primary_topic or "")


class TestAlgorithmicEngine(unittest.TestCase):
    """Verifies the core conceptual blending engine."""

    def setUp(self):
        self.breeder = TopicCrossBreeder(seed=42)

    def test_single_topic_cross_breed(self):
        """Verify 1 input topic generates exactly 10 diverse topics."""
        topics = ["Superconductivity"]
        generated = self.breeder.cross_breed(topics, count=10)

        self.assertEqual(len(generated), 10)
        self.assertEqual([t.index for t in generated], list(range(1, 11)))
        self.assertEqual(len(set(t.title for t in generated)), 10)

        # Domain diversity
        domains = [t.domain.name for t in generated]
        self.assertGreaterEqual(len(set(domains)), 8)

        for t in generated:
            self.assertTrue(t.primary_query)
            self.assertTrue(t.fallback_query)

    def test_dual_topic_cross_breed(self):
        """Verify 2 input topics generate dual-topic synthesis blends."""
        topics = ["Quantum Computing", "Synthetic Biology"]
        generated = self.breeder.cross_breed(topics, count=10)

        self.assertEqual(len(generated), 10)
        operators = [t.operator_name for t in generated]
        self.assertIn("Dual-Topic Synthesis", operators)

    def test_triple_topic_cross_breed(self):
        """Verify 3 input topics generate 10 diverse topics."""
        topics = ["Quantum Metamaterials", "Neuroplasticity", "CRISPR"]
        generated = self.breeder.cross_breed(topics, count=10)

        self.assertEqual(len(generated), 10)
        self.assertEqual(len(set(t.title for t in generated)), 10)

    def test_grammar_and_articles(self):
        """Verify proper 'A' vs 'An' article usage in generated framework titles."""
        for seed in range(3):
            breeder = TopicCrossBreeder(seed=seed)
            generated = breeder.cross_breed(["Astrophysics"], count=10)
            for t in generated:
                if t.title.startswith("A "):
                    second_word = t.title.split()[1]
                    self.assertNotIn(second_word[0].lower(), "aeiou", f"Invalid article 'A' before vowel: {t.title}")
                elif t.title.startswith("An "):
                    second_word = t.title.split()[1]
                    self.assertIn(second_word[0].lower(), "aeiou", f"Invalid article 'An' before consonant: {t.title}")

    def test_empty_input_handling(self):
        """Verify empty input topic raises appropriate ValueError."""
        with self.assertRaises(ValueError):
            self.breeder.cross_breed([])
        with self.assertRaises(ValueError):
            self.breeder.cross_breed(["   ", ""])


class TestBibTeXAndOpenAlex(unittest.TestCase):
    """Verifies BibTeX formatting and OpenAlex fallback retrieval."""

    def test_generate_bibtex(self):
        """Verify clean BibTeX generation from paper metadata."""
        paper = RetrievedPaper(
            title="Atomic force microscopy-based mechanobiology",
            authors=["Michael Krieg", "Gotthold Fläschner", "David Alsteens"],
            pub_year="2018",
            venue="Nature Reviews Physics",
            citations=825,
            abstract="Sample abstract snippet.",
            url="https://doi.org/10.1038/s42254-018-0001-7",
            query_used="mechanobiology cell biology",
            is_fallback=False,
            doi="10.1038/s42254-018-0001-7",
            oa_url="https://www.nature.com/articles/s42254-018-0001-7.pdf"
        )
        bib = generate_bibtex(paper)
        self.assertTrue(bib.startswith("@article{krieg2018atomic,"))
        self.assertIn("title     = {Atomic force microscopy-based mechanobiology}", bib)
        self.assertIn("Michael Krieg and Gotthold Fläschner and David Alsteens", bib)
        self.assertIn("journal   = {Nature Reviews Physics}", bib)
        self.assertIn("doi       = {10.1038/s42254-018-0001-7}", bib)
        self.assertIn("Open Access: https://www.nature.com/articles/s42254-018-0001-7.pdf", bib)

    def test_save_bibtex_file(self):
        """Verify saving and appending BibTeX entry to a file."""
        test_file = "test_refs.bib"
        if os.path.exists(test_file):
            os.remove(test_file)

        sample_entry = "@article{test2026,\n  title = {Test Publication}\n}"
        success = save_bibtex(sample_entry, test_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(test_file))

        with open(test_file, "r") as fh:
            content = fh.read()
        self.assertIn("@article{test2026", content)
        os.remove(test_file)

    def test_openalex_direct_query(self):
        """Verify querying OpenAlex API directly extracts metadata and OA PDF URL."""
        paper = OpenAlexClient.search_work("mechanobiology cell biology")
        self.assertIsNotNone(paper, "OpenAlex API should return work for standard query")
        self.assertTrue(paper.title)
        self.assertTrue(paper.pub_year)
        self.assertTrue(any(s in paper.source for s in ["OpenAlex", "Crossref"]))
        self.assertTrue(paper.bibtex, "BibTeX should be auto-generated for OpenAlex result")

    def test_scholar_fallback_to_openalex(self):
        """Verify that when Google Scholar raises an error / rate-limit, OpenAlex takes over seamlessly."""
        breeder = TopicCrossBreeder(seed=1)
        topics = breeder.cross_breed(["cell biology"], count=10)
        selected = topics[0]

        # Mock Scholar to simulate rate limit / block (raises Exception)
        with patch("scholarly.scholarly.search_pubs", side_effect=Exception("Simulated Google 429 Rate Limit")):
            paper = ScholarLiteratureClient.fetch_paper(selected, silent=True)
            self.assertIsNotNone(paper, "OpenAlex fallback must retrieve paper when Scholar fails")
            self.assertTrue(any(s in paper.source for s in ["OpenAlex", "Crossref"]))
            self.assertTrue(paper.bibtex)


class TestScholarlyIntegration(unittest.TestCase):
    """Verifies live integration with literature fetcher."""

    def test_real_paper_fetch_cell_biology(self):
        """Fetch a real publication for cell biology high-affinity crossover."""
        breeder = TopicCrossBreeder(seed=12)
        topics = breeder.cross_breed(["cell biology"], count=10)
        selected_topic = topics[0]

        paper = ScholarLiteratureClient.fetch_paper(selected_topic, silent=True)
        self.assertIsNotNone(paper, "Should retrieve a valid paper for cell biology")
        self.assertTrue(paper.title, "Paper must have a non-empty title")
        self.assertTrue(paper.url, "Paper must have a URL")
        self.assertTrue(paper.bibtex, "Paper must have an auto-generated BibTeX entry")


class TestTriAxialRetrieval(unittest.TestCase):
    """Verifies 3-Dimensional Literature Slice (Foundation, Frontier 2024-2026, Review)."""

    def setUp(self):
        breeder = TopicCrossBreeder(seed=42)
        self.topics = breeder.cross_breed(["cell biology"], count=10)
        self.selected_topic = self.topics[0]

    def test_openalex_frontier_and_review_filters(self):
        """Verify OpenAlex queries with min_year and is_review filters."""
        # 1. Frontier: 2024-2026
        frontier_works = OpenAlexClient.search_works("mechanobiology cell biology", min_year=2024, per_page=3)
        self.assertTrue(len(frontier_works) > 0, "OpenAlex should return frontier works for 2024+")
        for w in frontier_works:
            if w.pub_year and w.pub_year.isdigit():
                self.assertGreaterEqual(int(w.pub_year), 2024, f"Frontier paper year must be >= 2024: {w.pub_year}")

        # 2. Review: Comprehensive survey
        review_works = OpenAlexClient.search_works("mechanobiology cell biology review", is_review=True, per_page=3)
        self.assertTrue(len(review_works) > 0, "OpenAlex should return review works")
        self.assertTrue(any(
            "review" in (w.title + " " + (w.abstract or "")).lower() or
            "survey" in (w.title + " " + (w.abstract or "")).lower() or
            w.venue
            for w in review_works
        ))

    def test_openalex_frontier_relevance_not_archaeology(self):
        """Verify OpenAlex relevance sorting retrieves cellular/biological works rather than unrelated domains."""
        frontier_works = OpenAlexClient.search_works("mechanobiology cell biology", min_year=2024, sort="relevance_score:desc", per_page=3)
        self.assertTrue(len(frontier_works) > 0)
        top_text = f"{frontier_works[0].title} {frontier_works[0].abstract or ''}".lower()
        bio_matches = [kw for kw in ["cell", "mechan", "tissue", "biolog", "biophys", "cancer", "matrix", "force", "overview", "interact"] if kw in top_text]
        self.assertTrue(len(bio_matches) > 0, f"Top frontier paper should relate to mechanobiology/cell biology, got: {frontier_works[0].title}")

    def test_fetch_triad_distinctness_and_structure(self):
        """Verify ScholarLiteratureClient.fetch_triad returns 3 distinct papers."""
        triad = ScholarLiteratureClient.fetch_triad(self.selected_topic, silent=True)
        self.assertIsInstance(triad, LiteratureTriad)
        self.assertIsNotNone(triad.foundation, "Foundation paper must be retrieved")
        self.assertIsNotNone(triad.frontier, "Frontier paper must be retrieved")
        self.assertIsNotNone(triad.review, "Review paper must be retrieved")

        # Check year bound on Frontier paper
        if triad.frontier.pub_year and triad.frontier.pub_year.isdigit():
            self.assertGreaterEqual(int(triad.frontier.pub_year), 2024, f"Frontier year must be >= 2024: {triad.frontier.pub_year}")

        # Check distinctness
        self.assertFalse(
            is_same_paper(triad.foundation, triad.frontier),
            f"Frontier paper duplicates Foundation paper: {triad.frontier.title}"
        )
        self.assertFalse(
            is_same_paper(triad.foundation, triad.review),
            f"Review paper duplicates Foundation paper: {triad.review.title}"
        )

        # Check BibTeX consolidation
        bib_block = triad.consolidated_bibtex
        self.assertIn("@article{", bib_block)
        self.assertGreaterEqual(bib_block.count("@article{"), 2)

    def test_scholar_fallback_to_openalex_triad(self):
        """Verify that triad retrieval falls back smoothly to OpenAlex if Scholar fails."""
        with patch("scholarly.scholarly.search_pubs", side_effect=Exception("Simulated Google Scholar 429 Rate Limit")):
            triad = ScholarLiteratureClient.fetch_triad(self.selected_topic, silent=True)
            self.assertIsNotNone(triad.foundation)
            self.assertIsNotNone(triad.frontier)
            self.assertIsNotNone(triad.review)
            for p in triad.papers:
                self.assertTrue(any(s in p.source for s in ["OpenAlex", "Crossref"]))


class TestResearchBriefGenerator(unittest.TestCase):
    """Verifies publication-ready Markdown Research Brief generation."""

    def setUp(self):
        self.breeder = TopicCrossBreeder(seed=42)
        self.raw_topics = ["Cell Biology"]
        self.topics = self.breeder.cross_breed(self.raw_topics, count=10)
        self.selected_topic = self.topics[0]
        self.triad = LiteratureTriad(
            foundation=RetrievedPaper(
                title="Landmark Mechanobiology Foundations",
                authors=["Jane Doe", "John Smith"],
                pub_year="2015",
                venue="Cell Mechanics",
                citations=1200,
                abstract="Seminal study on mechanotransduction.",
                url="https://doi.org/10.1000/foundation",
                query_used="cell biology mechanobiology",
                is_fallback=False,
                doi="10.1000/foundation"
            ),
            frontier=RetrievedPaper(
                title="Frontier Microfluidic Cell Rheology in 2025",
                authors=["Alice Green"],
                pub_year="2025",
                venue="Nature Physics",
                citations=15,
                abstract="Cutting-edge 2025 single-cell measurements.",
                url="https://doi.org/10.1000/frontier",
                query_used="cell biology mechanobiology",
                is_fallback=True,
                source="OpenAlex (Open Access)",
                doi="10.1000/frontier",
                oa_url="https://oa.org/frontier.pdf"
            ),
            review=RetrievedPaper(
                title="A Comprehensive Review of Cytoskeletal Dynamics",
                authors=["Robert Brown", "Emily White"],
                pub_year="2023",
                venue="Annual Review of Biophysics",
                citations=350,
                abstract="Exhaustive review synthesizing the field.",
                url="https://doi.org/10.1000/review",
                query_used="cell biology review",
                is_fallback=True,
                source="OpenAlex",
                doi="10.1000/review"
            )
        )
        for p in self.triad.papers:
            p.bibtex = generate_bibtex(p)

    def test_brief_markdown_structure(self):
        """Verify generated Markdown report contains all required sections and formatting."""
        brief = generate_research_brief(self.selected_topic, self.triad, self.topics, self.raw_topics)

        # 1. Executive Summary & Metadata
        self.assertIn(f"# Research Brief: {self.selected_topic.title}", brief)
        self.assertIn("## Executive Summary & Conceptual Formulation", brief)
        self.assertIn("Cell Biology", brief)
        self.assertIn(self.selected_topic.domain.name, brief)
        self.assertIn(self.selected_topic.operator_name, brief)

        # 2. Interdisciplinary Landscape Map (Table)
        self.assertIn("## Interdisciplinary Landscape Map", brief)
        self.assertIn("| # | Proposed Research Direction | Interdisciplinary Bridge | Affinity | Epistemic Lens |", brief)
        self.assertIn("| 01 |", brief)
        self.assertIn("| 10 |", brief)
        self.assertIn("(Selected)", brief)

        # 3. Deep Dive: Literature Triad
        self.assertIn("## Deep Dive: The Literature Triad", brief)
        self.assertIn("🏛️ The Foundation: Seminal Landmark Publication", brief)
        self.assertIn("⚡ The Frontier (2024–2026): Emerging Edge & Preprints", brief)
        self.assertIn("📚 The Review: Comprehensive Survey of the State-of-the-Art", brief)
        self.assertIn("Landmark Mechanobiology Foundations", brief)
        self.assertIn("Frontier Microfluidic Cell Rheology in 2025", brief)
        self.assertIn("Direct Open Access PDF", brief)
        self.assertIn("https://oa.org/frontier.pdf", brief)
        self.assertIn("A Comprehensive Review of Cytoskeletal Dynamics", brief)

        # 4. Integrated BibTeX Appendix
        self.assertIn("## Integrated BibTeX Appendix", brief)
        self.assertIn("```bibtex", brief)
        self.assertIn("@article{doe2015landmark,", brief)
        self.assertIn("@article{green2025frontier,", brief)
        self.assertIn("@article{brown2023comprehensive,", brief)

    def test_save_research_brief(self):
        """Verify saving Research Brief to file."""
        test_file = "test_Research_Brief.md"
        if os.path.exists(test_file):
            os.remove(test_file)

        brief = generate_research_brief(self.selected_topic, self.triad, self.topics, self.raw_topics)
        success = save_research_brief(brief, test_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(test_file))

        with open(test_file, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("# Research Brief:", content)
        self.assertIn("🏛️ The Foundation", content)
        os.remove(test_file)


class TestKnowledgeGraph(unittest.TestCase):
    """Verifies interactive Vis.js HTML knowledge graph generation."""

    def setUp(self):
        breeder = TopicCrossBreeder(seed=42)
        self.raw_topics = ["Cell Biology"]
        self.topics = breeder.cross_breed(self.raw_topics, count=10)
        self.selected_topic = self.topics[0]
        self.triad = LiteratureTriad(
            foundation=RetrievedPaper(
                title="Landmark Mechanobiology Foundations",
                authors=["Jane Doe"],
                pub_year="2015",
                venue="Cell Mechanics",
                citations=1200,
                abstract="Seminal study.",
                url="https://doi.org/10.1000/foundation",
                query_used="cell biology mechanobiology",
                is_fallback=False,
                doi="10.1000/foundation"
            ),
            frontier=RetrievedPaper(
                title="Frontier Rheology in 2025",
                authors=["Alice Green"],
                pub_year="2025",
                venue="Nature Physics",
                citations=15,
                abstract="2025 frontier research.",
                url="https://doi.org/10.1000/frontier",
                query_used="cell biology mechanobiology",
                is_fallback=True,
                source="OpenAlex (Open Access)",
                doi="10.1000/frontier",
                oa_url="https://oa.org/frontier.pdf"
            ),
            review=RetrievedPaper(
                title="Comprehensive Review",
                authors=["Bob White"],
                pub_year="2024",
                venue="Reviews",
                citations=300,
                abstract="Comprehensive review.",
                url="https://doi.org/10.1000/review",
                query_used="cell biology review",
                is_fallback=True,
                source="OpenAlex",
                doi="10.1000/review"
            )
        )

    def test_generate_knowledge_graph_html(self):
        """Verify HTML output contains Vis.js library, network containers, nodes and edges data."""
        html = generate_knowledge_graph(self.selected_topic, self.triad, self.topics, self.raw_topics)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("vis-network.min.js", html)
        self.assertIn("network-container", html)
        self.assertIn("Core Input Seed Topics", html)
        self.assertIn("Interdisciplinary Bridges", html)
        self.assertIn("Research Directions", html)
        self.assertIn("Literature Triad", html)
        self.assertIn("Landmark Mechanobiology Foundations", html)
        self.assertIn("Frontier Rheology in 2025", html)
        self.assertIn("https://oa.org/frontier.pdf", html)
        self.assertIn("new vis.Network", html)

    def test_save_knowledge_graph(self):
        """Verify saving HTML graph to a file."""
        test_file = "test_graph.html"
        if os.path.exists(test_file):
            os.remove(test_file)
        html = generate_knowledge_graph(self.selected_topic, self.triad, self.topics, self.raw_topics)
        success = save_knowledge_graph(html, test_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("vis-network", content)
        os.remove(test_file)


class TestCLIExecution(unittest.TestCase):
    """Verifies the CLI interface via end-to-end subprocess runs."""

    def test_cli_demo_auto(self):
        """CLI: python science_expander.py --demo --auto"""
        proc = subprocess.run(
            [sys.executable, "science_expander.py", "--demo", "--auto"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertIn("GENOMICS OF IDEAS", proc.stdout)
        self.assertIn("THE LITERATURE TRIAD FOR TOPIC #1", proc.stdout)
        self.assertIn("BibTeX Entry:", proc.stdout)

    def test_cli_cell_biology_bibtex_export(self):
        """CLI: python science_expander.py -t 'cell biology' --select 1 --bibtex test_cli.bib"""
        test_bib = "test_cli.bib"
        if os.path.exists(test_bib):
            os.remove(test_bib)

        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "cell biology", "--select", "1", "--bibtex", test_bib],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertTrue(os.path.exists(test_bib), "BibTeX file should be created by CLI")
        
        with open(test_bib, "r") as fh:
            content = fh.read()
        self.assertIn("@article{", content)
        os.remove(test_bib)

    def test_cli_cell_biology_report_export(self):
        """CLI: python science_expander.py -t 'Cell Biology' --select 1 --report test_cb_brief.md"""
        test_report = "test_cb_brief.md"
        if os.path.exists(test_report):
            os.remove(test_report)

        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "Cell Biology", "--select", "1", "--report", test_report],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertTrue(os.path.exists(test_report), "Report file should be generated by CLI")

        with open(test_report, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("# Research Brief:", content)
        self.assertIn("## Interdisciplinary Landscape Map", content)
        self.assertIn("## Deep Dive: The Literature Triad", content)
        self.assertIn("🏛️ The Foundation", content)
        self.assertIn("⚡ The Frontier", content)
        self.assertIn("📚 The Review", content)
        self.assertIn("## Integrated BibTeX Appendix", content)
        os.remove(test_report)

    def test_cli_cell_biology_graph_export(self):
        """CLI: python science_expander.py -t 'Cell Biology' --select 1 --graph test_cb_graph.html"""
        test_graph = "test_cb_graph.html"
        if os.path.exists(test_graph):
            os.remove(test_graph)

        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "Cell Biology", "--select", "1", "--graph", test_graph],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertTrue(os.path.exists(test_graph), "Knowledge graph HTML should be generated by CLI")

        with open(test_graph, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("vis-network", content)
        self.assertIn("Cell Biology", content)
        os.remove(test_graph)

    def test_cli_json_mode_with_triad(self):
        """CLI: python science_expander.py -t 'cell biology' --json --select 1"""
        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "cell biology", "--json", "--select", "1"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        
        data = json.loads(proc.stdout)
        self.assertEqual(len(data["generated_topics"]), 10)
        self.assertEqual(data["selected_topic_index"], 1)
        self.assertIn("retrieved_paper", data)
        self.assertIn("bibtex", data["retrieved_paper"])
        self.assertTrue(data["retrieved_paper"]["bibtex"].startswith("@article{"))
        self.assertIn("literature_triad", data)
        self.assertIn("foundation", data["literature_triad"])
        self.assertIn("frontier", data["literature_triad"])
        self.assertIn("review", data["literature_triad"])

    def test_cli_russian_topic_bilingual_export(self):
        """CLI: python science_expander.py -t 'Клеточная биология' --select 1 --report test_ru_brief.md --graph test_ru_graph.html"""
        test_report = "test_ru_brief.md"
        test_graph = "test_ru_graph.html"
        for p in [test_report, test_graph]:
            if os.path.exists(p):
                os.remove(p)

        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "Клеточная биология", "--select", "1", "--report", test_report, "--graph", test_graph],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertTrue(os.path.exists(test_report), "Bilingual report should be generated")
        self.assertTrue(os.path.exists(test_graph), "Bilingual knowledge graph should be generated")

        with open(test_report, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("Клеточная биология", content)
        self.assertIn("Cell Biology", content)

        with open(test_graph, "r", encoding="utf-8") as fh:
            graph_html = fh.read()
        self.assertIn("Клеточная биология", graph_html)

        for p in [test_report, test_graph]:
            if os.path.exists(p):
                os.remove(p)

    def test_cli_json_mode_bilingual(self):
        """CLI: python science_expander.py -t 'Клеточная биология' --json --select 1"""
        proc = subprocess.run(
            [sys.executable, "science_expander.py", "-t", "Клеточная биология", "--json", "--select", "1"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        data = json.loads(proc.stdout)
        self.assertIn("bilingual_topics", data)
        self.assertTrue(any("Клеточная биология" in bt for bt in data["bilingual_topics"]))
        self.assertIn("display_primary_topic", data["generated_topics"][0])
        self.assertIn("Клеточная биология", data["generated_topics"][0]["display_primary_topic"])


class TestStreamlitApp(unittest.TestCase):
    """Verifies the Streamlit Web Application components and execution helpers."""

    def test_streamlit_app_import(self):
        """Verify that app.py can be cleanly imported without errors."""
        import app
        self.assertTrue(hasattr(app, "run_cross_breeding_core"))
        self.assertTrue(hasattr(app, "compute_display_seeds"))
        self.assertTrue(hasattr(app, "render_app"))

    def test_compute_display_seeds(self):
        """Verify bilingual and English seed label computation in app.py."""
        import app
        en_seeds = app.compute_display_seeds(["Cell Biology", "Quantum Physics"])
        self.assertEqual(en_seeds, ["Cell Biology", "Quantum Physics"])

        ru_seeds = app.compute_display_seeds(["Клеточная биология"])
        self.assertEqual(len(ru_seeds), 1)
        self.assertIn("Клеточная биология", ru_seeds[0])
        self.assertIn("Cell Biology", ru_seeds[0])

    def test_run_cross_breeding_core_english(self):
        """Verify core synthesis in app.py with English topics."""
        import app
        raw_s, disp_s, topics, g_html = app.run_cross_breeding_core("Cell Biology", count=5)
        self.assertEqual(raw_s, ["Cell Biology"])
        self.assertEqual(disp_s, ["Cell Biology"])
        self.assertEqual(len(topics), 5)
        self.assertTrue(g_html.startswith("<!DOCTYPE html>"))
        self.assertIn("vis-network", g_html)

    def test_run_cross_breeding_core_bilingual(self):
        """Verify core synthesis in app.py with Russian topics."""
        import app
        raw_s, disp_s, topics, g_html = app.run_cross_breeding_core("Клеточная биология", count=5)
        self.assertEqual(raw_s, ["Клеточная биология"])
        self.assertIn("Клеточная биология", disp_s[0])
        self.assertIn("Cell Biology", disp_s[0])
        self.assertEqual(len(topics), 5)
        self.assertIn("Клеточная биология", g_html)


class TestSemanticCorridorEngine(unittest.TestCase):
    """Verifies local vector intelligence, ONNX embeddings, cosine math, and Goldilocks scoring."""

    def test_cosine_similarity_math(self):
        """Verify vector cosine similarity calculation across canonical geometric cases."""
        import numpy as np
        # 1. Identical vectors
        u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(SemanticCorridorEngine.cosine_similarity(u, u), 1.0, places=5)

        # 2. Orthogonal vectors
        v_orth = np.array([-2.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(SemanticCorridorEngine.cosine_similarity(u, v_orth), 0.0, places=5)

        # 3. Parallel scaled vectors
        v_scaled = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        self.assertAlmostEqual(SemanticCorridorEngine.cosine_similarity(u, v_scaled), 1.0, places=5)

        # 4. Zero and None vectors
        v_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(SemanticCorridorEngine.cosine_similarity(v_zero, u), 0.0)
        self.assertEqual(SemanticCorridorEngine.cosine_similarity(None, u), 0.0)
        self.assertEqual(SemanticCorridorEngine.cosine_similarity(u, None), 0.0)

    def test_goldilocks_scoring_curve(self):
        """Verify the Interdisciplinary Goldilocks Sweet Spot scoring curve."""
        # 1. Peak Sweet Spot [0.35, 0.65]
        score_50 = SemanticCorridorEngine.calculate_goldilocks_score(0.50)
        self.assertEqual(score_50, 1.0)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.50), "Sweet Spot")

        score_35 = SemanticCorridorEngine.calculate_goldilocks_score(0.35)
        self.assertGreaterEqual(score_35, 0.90)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.35), "Sweet Spot")

        score_65 = SemanticCorridorEngine.calculate_goldilocks_score(0.65)
        self.assertGreaterEqual(score_65, 0.90)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.65), "Sweet Spot")

        # 2. Trivial Overlap Penalty (> 0.80)
        score_85 = SemanticCorridorEngine.calculate_goldilocks_score(0.85)
        self.assertLess(score_85, 0.40)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.85), "Trivial Overlap")

        score_95 = SemanticCorridorEngine.calculate_goldilocks_score(0.95)
        self.assertLess(score_95, 0.20)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.95), "Trivial Overlap")

        # 3. Conceptual Disconnect Penalty (< 0.20)
        score_10 = SemanticCorridorEngine.calculate_goldilocks_score(0.10)
        self.assertLess(score_10, 0.25)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(0.10), "Conceptual Disconnect")

        score_neg = SemanticCorridorEngine.calculate_goldilocks_score(-0.1)
        self.assertEqual(score_neg, 0.0)
        self.assertEqual(SemanticCorridorEngine.get_vector_zone(-0.1), "Conceptual Disconnect")

    def test_vector_engine_embedding_and_cache(self):
        """Verify ONNX embedding generation and in-memory caching."""
        engine = SemanticCorridorEngine.get_instance()
        if not engine.is_available:
            self.skipTest("FastEmbed is not available in current environment")

        vec1 = engine.embed("Cell Biology")
        self.assertIsNotNone(vec1)
        self.assertEqual(len(vec1.shape), 1)

        # Cache check: second call should be instantaneous and return identical array
        vec2 = engine.embed("Cell Biology")
        self.assertIs(vec1, vec2)

        # Semantic similarity between related fields
        vec_mech = engine.embed("Biophysics & Mechanobiology")
        sim = engine.cosine_similarity(vec1, vec_mech)
        self.assertGreater(sim, 0.40)
        self.assertLessEqual(sim, 1.0)

    def test_engine_graceful_fallback(self):
        """Verify graceful degradation to rule-based heuristics when vector engine is disabled."""
        # Create an engine instance forced to unavailable
        fallback_engine = SemanticCorridorEngine.__new__(SemanticCorridorEngine)
        fallback_engine.model_name = "dummy"
        fallback_engine.model = None
        fallback_engine.is_available = False
        fallback_engine._cache = {}

        self.assertIsNone(fallback_engine.embed("Cell Biology"))

        # TopicCrossBreeder should seamlessly function using heuristic rules without crashing
        breeder = TopicCrossBreeder(seed=42, vector_engine=fallback_engine)
        topics = breeder.cross_breed(["Cell Biology"], count=5)
        self.assertEqual(len(topics), 5)
        for t in topics:
            self.assertGreaterEqual(t.affinity, 0.80)
            self.assertIsNone(t.cosine_similarity)

    def test_cross_breeder_vector_attributes(self):
        """Verify that topics generated with active vector engine contain valid vector metrics."""
        engine = SemanticCorridorEngine.get_instance()
        if not engine.is_available:
            self.skipTest("FastEmbed is not available in current environment")

        breeder = TopicCrossBreeder(seed=42)
        topics = breeder.cross_breed(["Cell Biology"], count=5)
        self.assertEqual(len(topics), 5)
        for t in topics:
            self.assertIsNotNone(t.cosine_similarity)
            self.assertIsNotNone(t.goldilocks_score)
            self.assertIn(t.vector_zone, ["Sweet Spot", "Moderate Overlap", "Trivial Overlap", "Distant Analogy", "Conceptual Disconnect"])
            self.assertGreaterEqual(t.affinity, 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)



#!/usr/bin/env python3
import asyncio
import json
import os
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from science_expander import (
    TopicCrossBreeder,
    ScholarLiteratureClient,
    DynamicConceptMiner,
    SemanticCorridorEngine,
    ScienceCategory,
    RetrievedPaper,
    LiteratureTriad,
    classify_topic,
    GeneratedTopic,
    DiscoveryTier
)

class TestTaxonomyAndClassification(unittest.TestCase):
    def test_topic_classification(self):
        self.assertEqual(classify_topic("cell biology"), ScienceCategory.BIOLOGY)
        self.assertEqual(classify_topic("quantum optics"), ScienceCategory.PHYSICS)

import re
from aioresponses import aioresponses
from aiohttp.client_reqrep import ClientResponse

class MockStreamWriter:
    output_size = 0

class PatchedClientResponse(ClientResponse):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('stream_writer', MockStreamWriter())
        super().__init__(*args, **kwargs)
@pytest.mark.asyncio
class TestDynamicConceptMiner:
    async def test_fetch_seed_concepts(self):
        with aioresponses() as m:
            m.get(re.compile(r'^https://api\.openalex\.org/.*'), payload={
                "results": [
                    {
                        "concepts": [
                            {"id": "https://openalex.org/C123", "display_name": "Test Concept", "level": 2}
                        ]
                    }
                ]
            }, repeat=True, response_class=PatchedClientResponse)
            
            concepts = await DynamicConceptMiner.fetch_seed_concepts("biology")
            assert "https://openalex.org/C123" in concepts
            assert concepts["https://openalex.org/C123"]["count"] == 2

@pytest.mark.asyncio
class TestSemanticCorridorEngine:
    async def test_bimodal_scoring(self):
        engine = SemanticCorridorEngine.get_instance()
        engine.embed_async = AsyncMock(return_value=[0.1]*384)
        engine.cosine_similarity_async = AsyncMock(return_value=0.7)
        
        concepts = {
            "C1": {"id": "C1", "display_name": "Test1", "level": 2, "count": 10}
        }
        
        scored = await engine.compute_bimodal_scores("seed", concepts)
        assert len(scored) == 1
        assert scored[0]["tier"] == DiscoveryTier.ADJACENT

@pytest.mark.asyncio
class TestTopicCrossBreeder:
    @patch('science_expander.DynamicConceptMiner.fetch_seed_concepts')
    @patch('science_expander.SemanticCorridorEngine.compute_bimodal_scores')
    async def test_cross_breed_async(self, mock_compute, mock_fetch):
        mock_fetch.return_value = {"C1": {"id": "C1", "display_name": "Test1", "level": 2, "count": 10}}
        mock_compute.return_value = [{
            "id": "C1",
            "display_name": "Test1",
            "level": 2,
            "count": 10,
            "cos_sim": 0.7,
            "bimodal_score": 0.8,
            "tier": DiscoveryTier.ADJACENT
        }]
        
        breeder = TopicCrossBreeder()
        results = await breeder.cross_breed_async(["biology"], count=1)
        assert len(results) >= 1
        assert results[0].concept_name == "Test1"
        assert results[0].tier == DiscoveryTier.ADJACENT

@pytest.mark.asyncio
class TestScholarLiteratureClient:
    async def test_fetch_triad_async(self):
        with aioresponses() as m:
            m.get(re.compile(r'^https://api\.openalex\.org/.*'), payload={
                "results": [{
                    "title": "Test Paper",
                    "authorships": [{"author": {"display_name": "John Doe"}}],
                    "publication_year": 2024,
                    "cited_by_count": 100,
                    "id": "W123"
                }]
            }, repeat=True, response_class=PatchedClientResponse)
            
            topic = GeneratedTopic(
                index=1, title="Test", primary_topic="biology", secondary_topic=None,
                concept_id="C1", concept_name="Test Concept", concept_level=2,
                bimodal_score=0.8, tier=DiscoveryTier.ADJACENT, operator_name="Test", rationale="Test"
            )
            
            triad = await ScholarLiteratureClient.fetch_triad_async(topic, silent=True)
            assert triad.foundation is not None
            assert triad.foundation.title == "Test Paper"
            assert triad.frontier is not None
            assert triad.review is not None

"""Tests for Web Search Engine."""

import pytest
from src.search_engine import WebSearchEngine
from src.models import SearchResult


def test_query_decomposition():
    engine = WebSearchEngine()
    queries = engine.decompose_query("Quantum Computing vs Classical Computing", depth="deep")
    assert len(queries) >= 1
    assert any("quantum" in q.lower() for q in queries)


def test_url_cleaning_and_normalization():
    dirty_url = "https://www.example.com/article?utm_source=twitter&utm_medium=social&ref=123"
    cleaned = WebSearchEngine._clean_url(dirty_url)
    assert "utm_source" not in cleaned
    assert "ref" not in cleaned
    assert "example.com/article" in cleaned

    normalized = WebSearchEngine._normalize_url(dirty_url)
    assert normalized == "example.com/article"


def test_wikipedia_fallback():
    engine = WebSearchEngine()
    results = engine.search_wikipedia("Artificial intelligence", max_results=2)
    assert len(results) > 0
    assert "wikipedia.org" in results[0].url
    assert results[0].source_engine == "wikipedia"


def test_unsupported_extensions():
    assert WebSearchEngine._is_unsupported("https://example.com/paper.pdf") is True
    assert WebSearchEngine._is_unsupported("https://example.com/archive.zip") is True
    assert WebSearchEngine._is_unsupported("https://example.com/article.html") is False
    assert WebSearchEngine._is_unsupported("https://example.com/news/tech") is False

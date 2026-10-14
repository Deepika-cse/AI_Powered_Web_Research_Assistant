"""Tests for Synthesizer module."""

from src.synthesizer import ResearchSynthesizer
from src.models import RelevantChunk, SourceCitation


def test_markdown_parsing():
    synth = ResearchSynthesizer()
    sample_md = """
    ## Executive Summary
    Quantum computers utilize quantum mechanical phenomena such as superposition and entanglement [1].

    ## In-Depth Analysis
    Recent breakthroughs by IBM demonstrate quantum utility with 127+ qubits [1]. Meanwhile Google focuses on error correction [2].

    ## Key Findings & Takeaways
    - Superposition allows parallel state representation
    - Error correction remains the critical bottleneck

    ## Cross-Source Consensus & Perspectives
    Both sources agree that fault-tolerant computing is still several years away.
    """

    exec_s, detailed, findings, consensus = synth._parse_markdown_sections(sample_md)
    assert "Quantum computers utilize" in exec_s
    assert len(findings) == 2
    assert "Superposition" in findings[0]
    assert consensus is not None
    assert "fault-tolerant" in consensus


def test_extractive_synthesis_fallback():
    synth = ResearchSynthesizer(api_key=None)
    chunk = RelevantChunk(
        source_index=1,
        title="Fusion Energy Advances",
        url="https://nature.com/fusion",
        domain="nature.com",
        text="Net energy gain was achieved in experimental tokamak reactors using high-field magnets.",
        relevance_score=1.5,
        credibility_badge="🟢 Academic",
        credibility_score=95,
    )
    citation = SourceCitation(
        id=1,
        title="Fusion Energy Advances",
        url="https://nature.com/fusion",
        domain="nature.com",
        snippet="Net energy gain was achieved...",
        credibility_badge="🟢 Academic",
        credibility_score=95,
        relevance_score=1.5,
    )

    report = synth._extractive_synthesis(
        query="Nuclear fusion net energy gain",
        sub_queries=["fusion energy status"],
        chunks=[chunk],
        citations=[citation],
        research_depth="quick",
        start_time=0.0,
    )

    assert report.total_sources_found == 1
    assert "Net energy gain" in report.executive_summary
    assert len(report.citations) == 1

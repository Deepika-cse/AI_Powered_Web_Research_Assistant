"""Automated Evaluation & Benchmark Suite for AI Web Research System.

Tests cover:
- HTML cleaning and boilerplate stripping
- Paragraph deduplication
- Date extraction with provenance
- Claim verification (supported, negated, numerical mismatch, unrelated excerpt)
- Empty claim list grounding score
- Invalid citation detection
"""

import pytest
from src.scraper import WebScraper
from src.retrieval import InformationRetriever
from src.verifier import ClaimVerifier
from src.models import (
    SearchResult,
    ScrapedDocument,
    RelevantChunk,
    SourceCitation,
    FactualClaim,
)
from src.pipeline import WebResearchPipeline


def test_html_cleaning_and_ad_stripping():
    scraper = WebScraper()
    dirty_html = """
    <html>
        <body>
            <header><h1>Site Brand</h1></header>
            <div class="advertisement">Buy our product now!</div>
            <div class="cookie-banner">Accept cookies</div>
            <article>
                <h1>Solid State Battery Milestone</h1>
                <p>Researchers have developed a solid-state electrolyte with ionic conductivity of 12 mS/cm at room temperature.</p>
                <p>This allows rapid charging to 80% within 15 minutes while preventing dendrite formation across lithium anodes.</p>
            </article>
            <div class="newsletter-signup">Sign up for emails</div>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    clean_text, paragraphs, date, date_type = scraper._parse_html(dirty_html)
    assert "Buy our product" not in clean_text
    assert "Accept cookies" not in clean_text
    assert "Sign up for emails" not in clean_text
    assert "ionic conductivity" in clean_text
    assert len(paragraphs) == 2


def test_paragraph_deduplication():
    scraper = WebScraper()
    repeated_paragraphs = [
        "Researchers at MIT developed a novel superconducting magnet capable of 20 tesla field strength.",
        "Researchers at MIT developed a novel superconducting magnet capable of 20 tesla field strength.",  # Duplicate
        "This magnet enables compact tokamak designs with significantly reduced capital expenditure.",
    ]
    deduped = scraper._deduplicate_paragraphs(repeated_paragraphs)
    assert len(deduped) == 2
    assert "20 tesla" in deduped[0]
    assert "compact tokamak" in deduped[1]


def test_date_extraction():
    scraper = WebScraper()
    html_with_date = """
    <html>
        <head>
            <meta property="article:published_time" content="2026-03-24T14:30:00Z" />
        </head>
        <body><article><p>Article content here with substantive length.</p></article></body>
    </html>
    """
    clean_text, paragraphs, pub_date, date_type = scraper._parse_html(html_with_date)
    assert pub_date == "2026-03-24"


def test_claim_verification_supported():
    """A claim directly supported by the source sentence must be marked verified."""
    verifier = ClaimVerifier()
    chunk = RelevantChunk(
        source_index=1,
        title="IBM Heron Announcement",
        url="https://ibm.com/heron",
        domain="ibm.com",
        text="IBM deployed the 156-qubit Heron R2 processor offering a 16x speedup over previous Eagle systems.",
        relevance_score=1.8,
        credibility_badge="🔵 Press",
        credibility_score=88,
    )
    citation = SourceCitation(
        id=1,
        title="IBM Heron Announcement",
        url="https://ibm.com/heron",
        domain="ibm.com",
        snippet="IBM deployed the 156-qubit Heron R2...",
        credibility_badge="🔵 Press",
        credibility_score=88,
    )

    claims = [
        FactualClaim(
            claim_text="IBM deployed the 156-qubit Heron R2 processor offering 16x speedup [1]",
            cited_source_ids=[1],
            verification_status="unverified",
        )
    ]

    verified_claims, score = verifier.verify_claims(claims, [chunk], [citation])
    assert len(verified_claims) == 1
    assert verified_claims[0].verification_status == "verified"
    assert verified_claims[0].confidence_score >= 0.8
    assert score == 100.0


def test_claim_verification_numerical_mismatch():
    """A claim with a false number not found in the source must NOT be marked verified."""
    verifier = ClaimVerifier()
    chunk = RelevantChunk(
        source_index=1,
        title="Fusion Experiment",
        url="https://example.com/fusion",
        domain="example.com",
        text="The laser delivered 2.05 megajoules of energy, yielding 3.15 megajoules in a 5-nanosecond pulse.",
        relevance_score=1.5,
        credibility_badge="⚪ General",
        credibility_score=60,
    )
    citation = SourceCitation(
        id=1,
        title="Fusion Experiment",
        url="https://example.com/fusion",
        domain="example.com",
        snippet="The laser delivered 2.05 megajoules...",
        credibility_badge="⚪ General",
        credibility_score=60,
    )

    # Claim containing "30 seconds" — not in source at all
    claims = [
        FactualClaim(
            claim_text="Scientists sustained net energy for 30 seconds continuously [1]",
            cited_source_ids=[1],
            verification_status="unverified",
        )
    ]

    verified_claims, score = verifier.verify_claims(claims, [chunk], [citation])
    assert len(verified_claims) == 1
    # Must NOT be verified — "30 seconds" is not in the source
    assert verified_claims[0].verification_status != "verified", (
        "REGRESSION: claim with false number was marked verified"
    )
    # Should be unsupported or qualified
    assert verified_claims[0].verification_status in ("qualified", "unsupported")


def test_claim_verification_negation():
    """
    REGRESSION TEST: A negated claim ('X was NOT the CEO') must never receive a
    'verified' badge from a positive excerpt about that person.

    This reproduces the critical bug identified in the audit:
    'Sam Altman was not the CEO of OpenAI on November 17, 2023' was being marked
    Verified against an excerpt that confirms he was fired on that date — which
    actually contradicts the negated claim.
    """
    verifier = ClaimVerifier()
    chunk = RelevantChunk(
        source_index=1,
        title="OpenAI Board Crisis",
        url="https://example.com/openai",
        domain="example.com",
        text=(
            "On Friday 17 November 2023, OpenAI's board of directors fired Sam Altman, "
            "stating that he had not been consistently candid with the board. "
            "Altman was reinstated as CEO on November 21, 2023."
        ),
        relevance_score=2.0,
        credibility_badge="⚪ General",
        credibility_score=60,
    )
    citation = SourceCitation(
        id=1,
        title="OpenAI Board Crisis",
        url="https://example.com/openai",
        domain="example.com",
        snippet="On Friday 17 November 2023, OpenAI's board fired Sam Altman...",
        credibility_badge="⚪ General",
        credibility_score=60,
    )

    # This claim is FALSE relative to the excerpt — the excerpt says he WAS fired (temporarily not CEO)
    # but the claim says he was NOT the CEO, phrasing it as a permanent/ongoing state.
    # More importantly: ANY negated claim should never be marked "verified" via positive entailment.
    negated_claim = FactualClaim(
        claim_text="Sam Altman was not the CEO of OpenAI on November 17, 2023 [1]",
        cited_source_ids=[1],
        verification_status="unverified",
    )

    verified_claims, score = verifier.verify_claims([negated_claim], [chunk], [citation])
    assert len(verified_claims) == 1
    # CRITICAL: must never be "verified" — negation requires special handling
    assert verified_claims[0].verification_status != "verified", (
        "REGRESSION: negated claim was marked 'verified' by a positive source excerpt. "
        "Entity overlap alone must not confirm a negated statement."
    )
    assert verified_claims[0].entailment_type != "direct", (
        "REGRESSION: negated claim was labeled 'Direct Entailment'"
    )


def test_claim_verification_unrelated_excerpt():
    """
    REGRESSION TEST: A claim about Topic A must not be verified by an excerpt about Topic B,
    even if the same entities appear in both (e.g. same person mentioned in different contexts).
    """
    verifier = ClaimVerifier()
    chunk = RelevantChunk(
        source_index=1,
        title="Sam Altman Biography",
        url="https://example.com/altman",
        domain="example.com",
        text=(
            "Sam Altman was born in Chicago, Illinois. He attended Stanford University "
            "before dropping out to pursue entrepreneurship. He later became President "
            "of Y Combinator from 2014 to 2019."
        ),
        relevance_score=1.0,
        credibility_badge="⚪ General",
        credibility_score=55,
    )
    citation = SourceCitation(
        id=1,
        title="Sam Altman Biography",
        url="https://example.com/altman",
        domain="example.com",
        snippet="Sam Altman was born in Chicago...",
        credibility_badge="⚪ General",
        credibility_score=55,
    )

    # This claim is about OpenAI's IPO — a completely different topic from the biography excerpt
    unrelated_claim = FactualClaim(
        claim_text="Sam Altman filed for OpenAI's IPO in June 2026 [1]",
        cited_source_ids=[1],
        verification_status="unverified",
    )

    verified_claims, score = verifier.verify_claims([unrelated_claim], [chunk], [citation])
    assert len(verified_claims) == 1
    # The biography excerpt does NOT mention the IPO — should not be verified
    assert verified_claims[0].verification_status != "verified", (
        "REGRESSION: claim about OpenAI IPO was verified against a biography excerpt "
        "that contains no information about the IPO. Topic overlap ≠ entailment."
    )


def test_empty_claim_list_grounding_score():
    """
    REGRESSION TEST: An empty claim list must NOT return a grounding score of 100.0.
    There are no claims, so there is no evidence of grounding.
    """
    verifier = ClaimVerifier()
    verified_claims, score = verifier.verify_claims([], [], [])
    assert verified_claims == []
    assert score == 0.0, (
        f"REGRESSION: empty claim list returned grounding score {score} instead of 0.0. "
        "No claims means no verified evidence — score must not default to 100."
    )


def test_invalid_citation_id_detection():
    verifier = ClaimVerifier()
    citation = SourceCitation(
        id=1,
        title="Source 1",
        url="https://example.com/1",
        domain="example.com",
        snippet="Text",
        credibility_badge="⚪ General",
        credibility_score=60,
    )

    claim = FactualClaim(
        claim_text="Quantum computers achieved room temperature coherence [9]",
        cited_source_ids=[9],  # Source 9 does not exist!
        verification_status="unverified",
    )

    verified_claims, score = verifier.verify_claims([claim], [], [citation])
    assert verified_claims[0].verification_status == "unsupported"
    assert "do not exist" in verified_claims[0].reasoning

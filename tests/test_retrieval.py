"""Tests for Information Retrieval & Source Credibility."""

from src.retrieval import InformationRetriever
from src.models import ScrapedDocument


def test_source_credibility_evaluation():
    retriever = InformationRetriever()

    # Academic — exact .edu subdomain
    score, badge = retriever.evaluate_source_credibility("cs.stanford.edu")
    assert score >= 90
    assert "Academic" in badge

    # News organization — exact match
    score, badge = retriever.evaluate_source_credibility("reuters.com")
    assert score >= 80
    assert "News" in badge  # Now labeled '📰 News Organization'

    # General Web
    score, badge = retriever.evaluate_source_credibility("randomblog.net")
    assert score <= 70


def test_domain_spoofing_prevention():
    """
    REGRESSION TESTS: Verify that domain spoofing attacks are blocked.

    Previously, substring matching allowed:
      - evilnature.com  → matched 'nature.com' → Academic 90%  (FIXED)
      - example.gov.attacker.com → matched '.gov.' → Government 95%  (FIXED)
      - beginnersinai.org → labeled 'Verified Organization'  (FIXED → now 'General Organization Domain')
    """
    retriever = InformationRetriever()

    # ATTACK 1: Subdomain spoofing — evilnature.com must NOT get academic rating
    score, badge = retriever.evaluate_source_credibility("evilnature.com")
    assert score < 90, (
        f"REGRESSION: evilnature.com received Academic score {score}. "
        "Substring matching of 'nature.com' must not grant academic credibility."
    )
    assert "Academic" not in badge, (
        f"REGRESSION: evilnature.com received Academic badge '{badge}'. "
        "Only genuine subdomains of nature.com should get Academic rating."
    )

    # ATTACK 2: Government spoofing — .gov. in middle of hostname must NOT get government rating
    score, badge = retriever.evaluate_source_credibility("example.gov.attacker.com")
    assert score < 90, (
        f"REGRESSION: example.gov.attacker.com received Government score {score}. "
        "Only domains ENDING in a known .gov TLD should get government rating."
    )
    assert "Government" not in badge, (
        f"REGRESSION: example.gov.attacker.com received Government badge '{badge}'."
    )

    # ATTACK 3: Legitimate .gov domain must still work
    score, badge = retriever.evaluate_source_credibility("pmindia.gov.in")
    assert score >= 90, "Legitimate government domain pmindia.gov.in must get high score"
    assert "Government" in badge

    # ATTACK 4: .org must be labeled 'General Organization Domain', not 'Verified Organization'
    score, badge = retriever.evaluate_source_credibility("beginnersinai.org")
    assert "Verified" not in badge, (
        f"REGRESSION: .org domain labeled '{badge}'. "
        ".org suffix does not imply the organization is verified or trustworthy."
    )
    assert "Organization" in badge or "General" in badge

    # ATTACK 5: Genuine Wikipedia subdomain must still get Encyclopedia rating
    score, badge = retriever.evaluate_source_credibility("en.wikipedia.org")
    assert "Encyclopedia" in badge

    # ATTACK 6: A domain containing 'wikipedia' but not a real subdomain must NOT get Encyclopedia rating
    score, badge = retriever.evaluate_source_credibility("fakewikipedia.org")
    assert "Encyclopedia" not in badge, (
        f"REGRESSION: fakewikipedia.org received Encyclopedia badge '{badge}'."
    )


def test_chunking_and_relevance():
    retriever = InformationRetriever()
    doc = ScrapedDocument(
        url="https://nature.com/articles/ai-research",
        title="AI in Scientific Discovery",
        domain="nature.com",
        clean_text=" ".join(["Artificial intelligence is accelerating scientific discovery across biology, chemistry, and physics."] * 50),
        word_count=500,
    )

    chunks, citations = retriever.retrieve_and_rank("Artificial intelligence discovery in physics", [doc], max_chunks=3)
    assert len(chunks) > 0
    assert len(citations) == 1
    assert citations[0].domain == "nature.com"
    assert chunks[0].relevance_score > 0

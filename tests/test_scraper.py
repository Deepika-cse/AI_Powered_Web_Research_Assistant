"""Tests for Web Scraper."""

from src.scraper import WebScraper
from src.models import SearchResult


def test_html_parsing_and_cleaning():
    scraper = WebScraper()
    html_sample = """
    <html>
        <head><title>Test Article</title></head>
        <body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <script>alert("run");</script>
            <article>
                <h1>Deep Learning Advances</h1>
                <p>Deep learning models have demonstrated extraordinary performance in understanding multimodal data.</p>
                <p>These breakthroughs leverage large-scale transformer architectures trained over vast corpora.</p>
            </article>
            <footer><p>Copyright 2026. All rights reserved. Cookie policy.</p></footer>
        </body>
    </html>
    """
    clean_text, paragraphs, pub_date, date_type = scraper._parse_html(html_sample)
    assert "Deep learning models" in clean_text
    assert "transformer architectures" in clean_text
    assert "alert" not in clean_text  # Script stripped
    assert "Home" not in clean_text   # Nav stripped
    assert "Cookie policy" not in clean_text  # Boilerplate stripped


def test_domain_extraction():
    assert WebScraper._extract_domain("https://www.reuters.com/tech/ai") == "reuters.com"
    assert WebScraper._extract_domain("https://sub.domain.edu/page") == "sub.domain.edu"


def test_fallback_on_unreachable_url():
    scraper = WebScraper(timeout=2)
    fake_item = SearchResult(
        title="Unreachable Web Page",
        url="https://invalid-non-existent-domain-12345.xyz/article",
        snippet="This is a verified search snippet fallback.",
    )
    doc = scraper.scrape_url(fake_item)
    assert doc.status == "success"
    assert "fallback" in doc.error_message.lower() or "network" in doc.error_message.lower()
    assert "verified search snippet fallback" in doc.clean_text

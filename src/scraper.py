"""Web Scraper module with Advanced Article Extraction, Boilerplate Stripping,
Date Extraction (with provenance label), and Deduplication."""

import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Set
import requests
from bs4 import BeautifulSoup
import trafilatura

from src.models import SearchResult, ScrapedDocument
from src.config import (
    SCRAPER_TIMEOUT_SECONDS,
    SCRAPER_MAX_WORKERS,
    MAX_PAGE_CHARS,
    USER_AGENTS,
)


class WebScraper:
    """Scrapes, extracts core article text, extracts publication dates (with provenance), and deduplicates passages."""

    def __init__(self, timeout: int = SCRAPER_TIMEOUT_SECONDS, max_workers: int = SCRAPER_MAX_WORKERS):
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()

    def scrape_url(self, search_item: SearchResult) -> ScrapedDocument:
        """Scrapes an individual URL and extracts clean article text and publication date."""
        url = search_item.url
        title = search_item.title
        domain = self._extract_domain(url)

        try:
            headers = {
                "User-Agent": USER_AGENTS[0],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
            }
            resp = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
            )

            if resp.status_code == 200 and resp.text:
                clean_text, paragraphs, pub_date, date_type = self._parse_html(resp.text, url)

                # Only accept date from search_item if we found nothing from the HTML
                if not pub_date and search_item.published_date:
                    pub_date = search_item.published_date
                    date_type = "url_pattern"

                if len(clean_text) >= 180 and len(paragraphs) >= 1:
                    # Distinguish event dates mentioned in content from webpage publication dates
                    event_date = None
                    event_match = re.search(
                        r"\b(?:landed|launched|occurred|happened|announced|signed|took place|image captured on|still image captured)\s+(?:on|in)?\s*"
                        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Feb|Jan|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}|(?:19|20)\d{2})",
                        clean_text, re.IGNORECASE
                    )
                    if event_match:
                        event_date = event_match.group(1).strip()

                    return ScrapedDocument(
                        url=url,
                        title=title,
                        domain=domain,
                        clean_text=clean_text[:MAX_PAGE_CHARS],
                        paragraphs=paragraphs,
                        word_count=len(clean_text.split()),
                        published_date=pub_date,
                        event_date=event_date,
                        date_type=date_type,
                        status="success",
                    )

            # Fallback to search snippet
            fallback_text = f"{title}. {search_item.snippet}"
            return ScrapedDocument(
                url=url,
                title=title,
                domain=domain,
                clean_text=fallback_text,
                paragraphs=[fallback_text],
                word_count=len(fallback_text.split()),
                published_date=search_item.published_date,
                date_type="unknown",
                status="success",
                error_message=f"HTTP {resp.status_code if 'resp' in dir() else 'N/A'} - used snippet",
            )

        except Exception as e:
            fallback_text = f"{title}. {search_item.snippet}"
            return ScrapedDocument(
                url=url,
                title=title,
                domain=domain,
                clean_text=fallback_text,
                paragraphs=[fallback_text],
                word_count=len(fallback_text.split()),
                published_date=search_item.published_date,
                date_type="unknown",
                status="success",
                error_message=f"Network error ({type(e).__name__}) - used snippet",
            )

    def scrape_batch(self, search_results: List[SearchResult]) -> List[ScrapedDocument]:
        """Scrapes a batch of search results concurrently and eliminates duplicate URLs."""
        scraped_docs: List[ScrapedDocument] = []
        if not search_results:
            return scraped_docs

        seen_urls: Set[str] = set()
        deduped_results: List[SearchResult] = []
        for r in search_results:
            norm = self._normalize_url(r.url)
            if norm not in seen_urls:
                seen_urls.add(norm)
                deduped_results.append(r)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self.scrape_url, item): item for item in deduped_results
            }
            for future in as_completed(future_to_item):
                try:
                    doc = future.result()
                    scraped_docs.append(doc)
                except Exception:
                    item = future_to_item[future]
                    domain = self._extract_domain(item.url)
                    scraped_docs.append(
                        ScrapedDocument(
                            url=item.url,
                            title=item.title,
                            domain=domain,
                            clean_text=f"{item.title}. {item.snippet}",
                            paragraphs=[f"{item.title}. {item.snippet}"],
                            word_count=len(item.snippet.split()),
                            published_date=item.published_date,
                            date_type="unknown",
                            status="success",
                            error_message="Fallback to search snippet",
                        )
                    )

        return scraped_docs

    def _parse_html(self, html: str, url: str = "") -> Tuple[str, List[str], Optional[str], str]:
        """
        Extracts clean article content using trafilatura (primary) or BeautifulSoup (fallback).
        Returns: (clean_text, paragraphs, date_value, date_type)
        date_type: "publication" | "last_modified" | "page_created" | "url_pattern" | "unknown"
        """
        pub_date: Optional[str] = None
        date_type: str = "unknown"

        # --- Try trafilatura first ---
        try:
            traf_text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
            )
            metadata = trafilatura.extract_metadata(html, default_url=url)
            if metadata and metadata.date:
                pub_date = self._clean_date_string(metadata.date)
                # Trafilatura preferentially finds datePublished from JSON-LD/meta → publication date
                date_type = "publication"

            if traf_text and len(traf_text.strip()) >= 160:
                paragraphs = [
                    p.strip()
                    for p in traf_text.split("\n")
                    if len(p.strip()) >= 40 and not self._is_boilerplate(p)
                ]
                deduped = self._deduplicate_paragraphs(paragraphs)
                if len(deduped) >= 1:
                    return "\n\n".join(deduped), deduped, pub_date, date_type
        except Exception:
            pass

        # --- Fallback: BeautifulSoup ---
        soup = BeautifulSoup(html, "html.parser")
        pub_date, date_type = self._extract_date_with_type(soup, url)

        for element in soup([
            "script", "style", "nav", "header", "footer", "aside", "form",
            "iframe", "noscript", "svg", "dialog", "menu", "canvas"
        ]):
            element.decompose()

        bad_selectors = [
            ".advertisement", ".ad-container", ".cookie-banner", ".newsletter-signup",
            ".social-share", ".related-posts", ".sidebar", ".comments", "#sidebar",
            "#comments", "#footer", ".author-bio", ".promo-box"
        ]
        for sel in bad_selectors:
            for bad in soup.select(sel):
                bad.decompose()

        article_container = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(attrs={"itemprop": "articleBody"})
            or soup.find("div", class_=re.compile(r"article[-_]body|entry[-_]content|post[-_]content", re.I))
            or soup.body
            or soup
        )

        raw_paragraphs: List[str] = []
        for tag in article_container.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 45 and not self._is_boilerplate(text):
                raw_paragraphs.append(text)

        deduped_paragraphs = self._deduplicate_paragraphs(raw_paragraphs)
        full_text = "\n\n".join(deduped_paragraphs)

        return full_text, deduped_paragraphs, pub_date, date_type

    def _extract_date_with_type(self, soup: BeautifulSoup, url: str) -> Tuple[Optional[str], str]:
        """
        Extracts date AND its type from a page.
        Returns: (date_string, date_type)
        date_type: "publication" | "last_modified" | "page_created" | "url_pattern" | "unknown"
        """
        # 1. Explicit publication meta tags → most reliable
        pub_meta_keys = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"property": "og:published_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"name": "dc.date.issued"}),
            ("meta", {"name": "parsely-pub-date"}),
        ]
        for tag_name, attrs in pub_meta_keys:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                date_str = self._clean_date_string(tag["content"])
                if date_str:
                    return date_str, "publication"

        # 2. Last-modified meta tag → labelled separately
        mod_tag = soup.find("meta", attrs={"name": "date"}) or soup.find("meta", attrs={"property": "og:updated_time"})
        if mod_tag and mod_tag.get("content"):
            date_str = self._clean_date_string(mod_tag["content"])
            if date_str:
                return date_str, "last_modified"

        # 3. <time> element
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            date_str = self._clean_date_string(time_tag["datetime"])
            if date_str:
                return date_str, "publication"

        # 4. JSON-LD Schema.org
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict):
                        if item.get("datePublished"):
                            date_str = self._clean_date_string(str(item["datePublished"]))
                            if date_str:
                                return date_str, "publication"
                        if item.get("dateModified"):
                            date_str = self._clean_date_string(str(item["dateModified"]))
                            if date_str:
                                return date_str, "last_modified"
            except Exception:
                continue

        # 5. URL date pattern (e.g. /2026/04/12/) — we know this is approximate
        url_match = re.search(r"/(202[0-9])[-/](0[1-9]|1[0-2])[-/]([0-3][0-9])", url)
        if url_match:
            d = f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}"
            return d, "url_pattern"

        return None, "unknown"

    @staticmethod
    def _clean_date_string(raw_date: str) -> Optional[str]:
        """Cleans and standardizes date string to YYYY-MM-DD."""
        if not raw_date:
            return None
        m = re.search(r"(20[12][0-9]-[0-1][0-9]-[0-3][0-9])", raw_date)
        if m:
            return m.group(1)
        m2 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? (20[12][0-9])", raw_date, re.I)
        if m2:
            return m2.group(0)
        return raw_date[:10] if len(raw_date) >= 10 else None

    @staticmethod
    def _deduplicate_paragraphs(paragraphs: List[str]) -> List[str]:
        """Removes duplicate and near-duplicate passages using token Jaccard overlap."""
        unique_paragraphs: List[str] = []
        seen_token_sets = []

        for p in paragraphs:
            tokens = set(re.findall(r"\w{4,}", p.lower()))
            if not tokens:
                unique_paragraphs.append(p)
                continue

            is_duplicate = False
            for prev_tokens in seen_token_sets:
                intersection = len(tokens & prev_tokens)
                union = len(tokens | prev_tokens)
                if union > 0 and (intersection / union) > 0.70:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_token_sets.append(tokens)
                unique_paragraphs.append(p)

        return unique_paragraphs

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        """Detects navigational, cookie, subscription, and social media notices."""
        lower = text.lower()
        patterns = [
            "cookie policy", "privacy policy", "all rights reserved",
            "subscribe to our newsletter", "sign in to your account",
            "terms of service", "accept all cookies", "manage preferences",
            "advertisement", "share this article", "follow us on twitter",
            "read more:", "click here to", "sponsored content",
            "copyright ©", "related stories", "leave a comment"
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extracts clean domain name from URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return "web-source"

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalizes URL for deduplication."""
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc.lower().replace("www.", "")
            path = parsed.path.rstrip("/")
            return f"{netloc}{path}"
        except Exception:
            return url.lower()

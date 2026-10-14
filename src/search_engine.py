"""Web Search Engine with Multi-Query Decomposition and Fallbacks."""

import re
import urllib.parse
from typing import List, Dict, Set, Optional
import requests
from ddgs import DDGS

from src.models import SearchResult
from src.config import USER_AGENTS


class WebSearchEngine:
    """Handles query decomposition, live web searches, and result deduplication."""

    def __init__(self):
        self.headers = {"User-Agent": USER_AGENTS[0]}

    def decompose_query(
        self, query: str, depth: str = "quick", groq_client=None
    ) -> List[str]:
        """
        Decomposes complex queries into multiple focused search queries.
        Uses Groq LLM if available, otherwise heuristics.
        """
        clean_q = query.strip()
        if depth == "quick":
            return [clean_q]

        # In deep research mode:
        if groq_client:
            try:
                system_prompt = (
                    "You are a search query formulation expert. Given a user query, "
                    "generate 2 to 3 distinct, high-impact search queries to explore different angles "
                    "(e.g., recent developments, underlying mechanisms, practical limitations/critique). "
                    "Output ONLY the queries separated by newlines, no numbers, no quotes, no extra text."
                )
                from src.config import DEFAULT_MODEL
                response = groq_client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Deconstruct this research topic: {clean_q}"},
                    ],
                    max_tokens=150,
                    temperature=0.3,
                )
                lines = response.choices[0].message.content.strip().split("\n")
                queries = [l.strip().lstrip("0123456789.-* ") for l in lines if l.strip()]
                queries = [q for q in queries if len(q) > 3][:3]
                if clean_q not in queries:
                    queries.insert(0, clean_q)
                return queries[:3]
            except Exception:
                pass  # Fall back to heuristic

        # Heuristic query decomposition
        queries = [clean_q]
        if " vs " in clean_q.lower() or " versus " in clean_q.lower():
            parts = re.split(r"\s+(?:vs|versus)\s+", clean_q, flags=re.IGNORECASE)
            if len(parts) == 2:
                queries.append(f"{parts[0].strip()} overview performance")
                queries.append(f"{parts[1].strip()} overview performance")
        elif any(w in clean_q.lower() for w in ["how", "why", "what", "compare", "impact"]):
            queries.append(f"{clean_q} recent advancements 2024 2025")
            queries.append(f"{clean_q} limitations and challenges")
        else:
            queries.append(f"{clean_q} comprehensive overview analysis")

        if any(term in clean_q.lower() for term in ["nasa", "perseverance", "curiosity", "mars rover", "jpl"]):
            queries.insert(1, f"site:nasa.gov {clean_q}")
            queries.insert(2, f"site:jpl.nasa.gov {clean_q}")

        return list(dict.fromkeys(queries))[:3]

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Executes a live search on DuckDuckGo using ddgs."""
        results: List[SearchResult] = []
        try:
            ddgs_client = DDGS()
            raw_hits = list(ddgs_client.text(query, max_results=max_results))
            for hit in raw_hits:
                url = hit.get("href", "")
                title = hit.get("title", "")
                snippet = hit.get("body", "")
                if url and title:
                    results.append(
                        SearchResult(
                            title=title.strip(),
                            url=self._clean_url(url),
                            snippet=snippet.strip(),
                            source_engine="duckduckgo",
                        )
                    )
        except Exception as e:
            # Fallback to duckduckgo news or wikipedia if text search encounters rate limiting
            try:
                ddgs_client = DDGS()
                news_hits = list(ddgs_client.news(query, max_results=max_results))
                for hit in news_hits:
                    url = hit.get("url") or hit.get("href", "")
                    title = hit.get("title", "")
                    snippet = hit.get("body", "")
                    if url and title:
                        results.append(
                            SearchResult(
                                title=title.strip(),
                                url=self._clean_url(url),
                                snippet=snippet.strip(),
                                source_engine="duckduckgo_news",
                                published_date=hit.get("date"),
                            )
                        )
            except Exception:
                pass

        return results

    def search_wikipedia(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """Queries the Wikipedia OpenSearch API as an authoritative fallback."""
        results: List[SearchResult] = []
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "namespace": 0,
                "format": "json",
            }
            resp = requests.get(api_url, params=params, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                titles = data[1] if len(data) > 1 else []
                snippets = data[2] if len(data) > 2 else []
                urls = data[3] if len(data) > 3 else []

                for t, s, u in zip(titles, snippets, urls):
                    results.append(
                        SearchResult(
                            title=t,
                            url=u,
                            snippet=s if s else f"Wikipedia entry for {t}",
                            source_engine="wikipedia",
                        )
                    )
        except Exception:
            pass

        return results

    def search_multi_query(
        self, queries: List[str], max_per_query: int = 4, total_limit: int = 8
    ) -> List[SearchResult]:
        """Executes search across multiple sub-queries, deduplicating URLs."""
        seen_urls: Set[str] = set()
        all_results: List[SearchResult] = []

        for q in queries:
            results = self.search_duckduckgo(q, max_results=max_per_query)
            if not results:
                # Try Wikipedia fallback
                results = self.search_wikipedia(q, max_results=2)

            for item in results:
                normalized = self._normalize_url(item.url)
                if normalized and normalized not in seen_urls and not self._is_unsupported(item.url):
                    seen_urls.add(normalized)
                    all_results.append(item)
                    if len(all_results) >= total_limit:
                        return all_results

        # If still empty, try broad wikipedia query
        if not all_results and queries:
            wiki_results = self.search_wikipedia(queries[0], max_results=3)
            for item in wiki_results:
                normalized = self._normalize_url(item.url)
                if normalized not in seen_urls:
                    seen_urls.add(normalized)
                    all_results.append(item)

        return all_results

    @staticmethod
    def _clean_url(url: str) -> str:
        """Strips tracking query parameters from URLs."""
        try:
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)
            # Remove tracking params
            filtered_params = {
                k: v for k, v in query_params.items()
                if not k.startswith("utm_") and k not in {"ref", "source", "fbclid", "gclid"}
            }
            new_query = urllib.parse.urlencode(filtered_params, doseq=True)
            return urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, "")
            )
        except Exception:
            return url

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

    @staticmethod
    def _is_unsupported(url: str) -> bool:
        """Checks if URL points to an unsupported binary file."""
        unsupported_exts = (
            ".pdf", ".zip", ".tar", ".gz", ".exe", ".dmg", ".mp4",
            ".mp3", ".avi", ".jpg", ".png", ".gif", ".iso"
        )
        return any(url.lower().endswith(ext) for ext in unsupported_exts)

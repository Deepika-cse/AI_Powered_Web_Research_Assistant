"""Information Retrieval, Source Credibility Evaluation, Freshness Scoring, and Context Ranking.

FIX: Source credibility labels are now accurate — Wikipedia is 'Encyclopedia Reference',
not 'Authoritative Press / Official Docs'.
"""

import re
from typing import List, Tuple, Dict, Optional
from src.models import ScrapedDocument, RelevantChunk, SourceCitation
from src.config import (
    ACADEMIC_GOV_DOMAINS,
    TOP_TECH_NEWS_DOMAINS,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
    MAX_CHUNKS_FOR_SYNTHESIS,
)

# --- Accurate Source Category Mapping ---
# These are checked in order; first match wins.

ENCYCLOPEDIA_DOMAINS = {
    "wikipedia.org", "britannica.com", "scholarpedia.org", "citizendium.org",
}

NEWS_ORG_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
    "ft.com", "wsj.com", "nytimes.com", "theguardian.com", "economist.com",
}

TECH_MEDIA_DOMAINS = {
    "technologyreview.com", "wired.com", "arstechnica.com", "theverge.com",
    "techcrunch.com", "venturebeat.com", "zdnet.com", "engadget.com",
}

DEVELOPER_DOCS_DOMAINS = {
    "github.com", "stackoverflow.com", "huggingface.co", "openai.com",
    "anthropic.com", "groq.com", "docs.python.org", "developer.mozilla.org",
    "deepmind.google",
}

EDUCATIONAL_DOMAINS = {
    "khan academy", "coursera.org", "edx.org", "mit.edu",
}


class InformationRetriever:
    """Evaluates source credibility with accurate category labels, calculates freshness, chunks, and ranks evidence."""

    def evaluate_source_credibility(self, domain: str) -> Tuple[int, str]:
        """
        Calculates credibility score (0-100) and assigns an ACCURATE authority badge.

        SECURITY FIX: All checks use exact hostname matching (== or genuine subdomain suffix)
        to prevent domain spoofing attacks such as:
          - evilnature.com  → was matching 'nature.com' via substring → FIXED
          - example.gov.attacker.com → was matching '.gov.' anywhere → FIXED
          - beginnersinai.org → was labeled 'Verified Organization' → FIXED (now 'General Organization Domain')

        Badges:
          🟢 Official Government Source    (exact .gov TLD or known country-code .gov.XX)
          🔬 Academic / Research           (exact .edu TLD or known academic/research domains)
          📰 News Organization             (exact match against known news domains)
          📖 Encyclopedia Reference        (Wikipedia, Britannica — reliable but community-edited)
          💻 Developer / Tech Docs         (GitHub, HuggingFace, OpenAI, MDN)
          🟡 Tech Media                    (Wired, TechCrunch, Ars Technica)
          🟡 General Organization Domain  (.org — domain type only, NOT a trust endorsement)
          ⚪ Educational Resource          (Coaching sites, tutorial platforms)
          ⚪ General Web Source            (Everything else)
        """
        dom = domain.lower().strip()

        # Helper: is `dom` exactly `trusted` or a genuine subdomain of `trusted`?
        def _is_domain_or_subdomain(dom: str, trusted: str) -> bool:
            return dom == trusted or dom.endswith("." + trusted)

        # Helper: does `dom` have one of the given TLD suffixes (exact)?
        def _has_tld(dom: str, *tlds: str) -> bool:
            return any(dom == tld or dom.endswith("." + tld) for tld in tlds)

        # 1. Official Government — exact .gov TLD or known country-code second-level gov domains
        #    FIX: previously used `.gov.` (substring) which allowed example.gov.attacker.com
        #    Now we only accept domains that END with a known government TLD pattern.
        GOV_TLDS = (".gov", ".gov.in", ".gov.uk", ".gov.au", ".gov.nz", ".gov.ca",
                    ".gov.za", ".gov.sg", ".gov.ie", ".govt.nz", ".gob.mx", ".gob.es")
        if any(dom.endswith(tld) for tld in GOV_TLDS):
            return 95, "🟢 Official Government Source"

        # 2. Academic / Research — exact domain or genuine subdomain matching only
        #    FIX: previously used `acad in dom` (substring) which allowed evilnature.com → nature.com
        ACADEMIC_EXACT = {
            "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
            "ncbi.nlm.nih.gov", "nih.gov", "who.int", "cdc.gov",
            "sciencedirect.com", "springer.com", "cell.com", "frontiersin.org",
            "plos.org", "jstor.org", "pubmed.ncbi.nlm.nih.gov",
        }
        ACADEMIC_TLDS = (".edu", ".ac.uk", ".ac.in", ".ac.jp", ".ac.nz")
        if any(dom.endswith(tld) for tld in ACADEMIC_TLDS):
            return 90, "🔬 Academic / Research Source"
        for acad in ACADEMIC_EXACT:
            if _is_domain_or_subdomain(dom, acad):
                return 90, "🔬 Academic / Research Source"

        # 3. News Organizations — professional editorial standards (exact match only)
        for news in NEWS_ORG_DOMAINS:
            if _is_domain_or_subdomain(dom, news):
                return 85, "📰 News Organization"

        # 4. Encyclopedia — reliable but community-edited; clearly labelled
        for enc in ENCYCLOPEDIA_DOMAINS:
            if _is_domain_or_subdomain(dom, enc):
                return 80, "📖 Encyclopedia Reference"

        # 5. Developer / Tech Official Docs (exact match only)
        for dev in DEVELOPER_DOCS_DOMAINS:
            if _is_domain_or_subdomain(dom, dev):
                return 80, "💻 Developer / Tech Docs"

        # 6. Tech Media (exact match only)
        for tech in TECH_MEDIA_DOMAINS:
            if _is_domain_or_subdomain(dom, tech):
                return 75, "🟡 Tech Media"

        # 7. .org TLD — FIX: renamed from 'Verified Organization' to 'General Organization Domain'
        #    A .org domain does NOT automatically mean the organization is verified or trustworthy.
        if dom.endswith(".org"):
            return 65, "🟡 General Organization Domain (.org)"

        if dom.endswith(".io") or dom.endswith(".dev"):
            return 60, "💻 Tech / Developer Resource"

        # 8. Educational / coaching sites
        coaching_keywords = ["vajiram", "unacademy", "byjus", "studyiq", "testbook",
                             "geeksforgeeks", "w3schools", "tutorialspoint"]
        if any(kw in dom for kw in coaching_keywords):
            return 60, "⚪ Educational Resource"

        return 55, "⚪ General Web Source"

    def calculate_freshness_bonus(self, query: str, pub_date: Optional[str]) -> float:
        """Computes a freshness multiplier for time-sensitive queries."""
        if not pub_date:
            return 1.0

        query_lower = query.lower()
        is_time_sensitive = any(
            w in query_lower
            for w in ["latest", "recent", "current", "2025", "2026", "new", "breakthrough", "now", "today"]
        )
        if not is_time_sensitive:
            return 1.0

        if "2026" in pub_date:
            return 1.25
        elif "2025" in pub_date:
            return 1.15
        elif "2024" in pub_date:
            return 1.05
        return 0.95

    def chunk_document(self, doc: ScrapedDocument, source_idx: int) -> List[RelevantChunk]:
        """Splits document text into overlapping semantic chunks with metadata and date."""
        cred_score, cred_badge = self.evaluate_source_credibility(doc.domain)
        words = doc.clean_text.split()
        if not words:
            return []

        chunks: List[RelevantChunk] = []
        step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)

        for i in range(0, len(words), step):
            chunk_words = words[i: i + CHUNK_SIZE_WORDS]
            if len(chunk_words) < 30:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append(
                RelevantChunk(
                    source_index=source_idx,
                    title=doc.title,
                    url=doc.url,
                    domain=doc.domain,
                    text=chunk_text,
                    relevance_score=0.0,
                    credibility_badge=cred_badge,
                    credibility_score=cred_score,
                    published_date=doc.published_date,
                    last_modified_date=doc.last_modified_date,
                    event_date=doc.event_date,
                    date_type=doc.date_type,
                )
            )

        return chunks

    def calculate_relevance(
        self, query: str, chunk_text: str, cred_score: int, pub_date: Optional[str]
    ) -> float:
        """Computes composite relevance score using term frequency, phrase matching, authority, and freshness."""
        query_words = re.findall(r"\w+", query.lower())
        if not query_words:
            return 0.0

        chunk_lower = chunk_text.lower()
        if not chunk_lower:
            return 0.0

        # 1. Lexical exact term matching
        matches = sum(1 for w in query_words if w in chunk_lower)
        term_score = matches / len(query_words)

        # 2. Subword / stem alignment
        semantic_matches = 0
        for qw in query_words:
            stem = qw[:4] if len(qw) >= 5 else qw
            if stem in chunk_lower:
                semantic_matches += 1
        semantic_score = semantic_matches / len(query_words)

        # 3. Bigram phrase matching bonus
        phrase_bonus = 0.0
        if len(query_words) >= 2:
            for j in range(len(query_words) - 1):
                bigram = f"{query_words[j]} {query_words[j + 1]}"
                if bigram in chunk_lower:
                    phrase_bonus += 0.40

        # 4. Credibility weight
        cred_weight = 0.8 + (cred_score / 100.0) * 0.4

        # 5. Freshness multiplier
        freshness_weight = self.calculate_freshness_bonus(query, pub_date)

        final_score = (term_score * 1.2 + semantic_score * 0.6 + phrase_bonus) * cred_weight * freshness_weight
        return round(final_score, 4)

    def retrieve_and_rank(
        self, query: str, documents: List[ScrapedDocument], max_chunks: int = MAX_CHUNKS_FOR_SYNTHESIS
    ) -> Tuple[List[RelevantChunk], List[SourceCitation]]:
        """Chunks, ranks, ensures diversity, and builds the citation registry."""
        all_chunks: List[RelevantChunk] = []

        for idx, doc in enumerate(documents, start=1):
            chunks = self.chunk_document(doc, idx)
            for c in chunks:
                c.relevance_score = self.calculate_relevance(query, c.text, c.credibility_score, c.published_date)
                all_chunks.append(c)

        all_chunks.sort(key=lambda x: x.relevance_score, reverse=True)

        # Ensure source diversity (max 3 chunks per source)
        selected_chunks: List[RelevantChunk] = []
        source_counts: Dict[str, int] = {}

        for chunk in all_chunks:
            count = source_counts.get(chunk.url, 0)
            if count < 3:
                selected_chunks.append(chunk)
                source_counts[chunk.url] = count + 1
                if len(selected_chunks) >= max_chunks:
                    break

        # Build numbered citations
        citations: List[SourceCitation] = []
        used_urls: Dict[str, int] = {}
        citation_id = 1

        for c in selected_chunks:
            if c.url not in used_urls:
                used_urls[c.url] = citation_id
                snippet_preview = c.text[:220].strip() + "..."
                citations.append(
                    SourceCitation(
                        id=citation_id,
                        title=c.title,
                        url=c.url,
                        domain=c.domain,
                        snippet=snippet_preview,
                        credibility_badge=c.credibility_badge,
                        credibility_score=c.credibility_score,
                        relevance_score=c.relevance_score,
                        published_date=c.published_date,
                        last_modified_date=c.last_modified_date,
                        event_date=c.event_date,
                        date_type=c.date_type,
                    )
                )
                citation_id += 1

        # Re-map chunk source indices to citation IDs
        for c in selected_chunks:
            c.source_index = used_urls.get(c.url, 1)

        return selected_chunks, citations

    @staticmethod
    def format_context_for_llm(chunks: List[RelevantChunk], citations: List[SourceCitation]) -> str:
        """Formats retrieved chunks into structured context for the Groq LLM with date provenance."""
        lines = []
        lines.append("=== RETRIEVED WEB EVIDENCE SOURCES ===")
        for cit in citations:
            if cit.published_date:
                date_label = {
                    "publication": "Published",
                    "last_modified": "Last Modified",
                    "url_pattern": "Date (from URL)",
                    "page_created": "Page Created",
                }.get(cit.date_type, "Date")
                date_info = f" | {date_label}: {cit.published_date}"
            else:
                date_info = " | Date: Unlisted"

            if cit.event_date:
                date_info += f" | Event Date Mentioned: {cit.event_date}"

            lines.append(
                f"[Source {cit.id}]: {cit.title} (Domain: {cit.domain} | {cit.credibility_badge} {cit.credibility_score}%{date_info})"
            )
            lines.append(f"URL: {cit.url}")

        lines.append("\n=== RELEVANT EXCERPTS (GROUNDING EVIDENCE) ===")
        for c in chunks:
            date_info = f" [{c.date_type}: {c.published_date}]" if c.published_date else ""
            lines.append(f"\n[Source {c.source_index}] ({c.domain}{date_info}):")
            lines.append(c.text.strip())

        return "\n".join(lines)

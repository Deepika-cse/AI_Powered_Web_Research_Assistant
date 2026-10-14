"""End-to-end Pipeline orchestrator with Stage-by-Stage Latency Instrumentation."""

import time
from typing import Callable, Optional, List, Tuple, Dict, Any
from src.models import (
    ResearchReport,
    SearchResult,
    ScrapedDocument,
    RelevantChunk,
    SourceCitation,
)
from src.search_engine import WebSearchEngine
from src.scraper import WebScraper
from src.retrieval import InformationRetriever
from src.synthesizer import ResearchSynthesizer
from src.database import save_report_to_db
from src.config import (
    SEARCH_MAX_RESULTS_QUICK,
    SEARCH_MAX_RESULTS_DEEP,
    DEFAULT_MODEL,
)


class WebResearchPipeline:
    """Coordinates search, scraping, retrieval, credibility analysis, claim verification, and LLM synthesis."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.search_engine = WebSearchEngine()
        self.scraper = WebScraper()
        self.retriever = InformationRetriever()
        self.synthesizer = ResearchSynthesizer(api_key=groq_api_key, model=model)

    def set_api_key(self, api_key: str, model: Optional[str] = None):
        """Updates Groq API key and model dynamically."""
        self.synthesizer = ResearchSynthesizer(
            api_key=api_key, model=model or self.synthesizer.model
        )

    def run_research(
        self,
        query: str,
        depth: str = "quick",
        max_sources: Optional[int] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[ResearchReport, str, List[ScrapedDocument], List[RelevantChunk]]:
        """
        Runs the full research pipeline with granular stage latency tracking:
        1. Decompose query
        2. Search the live web (Multi-Source)
        3. Scrape and clean web pages (Deduplicated passages)
        4. Evaluate credibility, freshness & retrieve top chunks
        5. Synthesize with Groq LLM & Verify Factual Claims

        Returns:
            Tuple of (ResearchReport, formatted_context_text, scraped_documents, top_chunks)
        """
        def update_status(step: str, detail: str):
            if progress_callback:
                progress_callback(step, detail)

        timings: Dict[str, float] = {}
        t_start_total = time.time()

        # Step 1 & 2: Query Decomposition & Web Search
        t0 = time.time()
        update_status("Query Analysis", f"Formulating multi-angle search plan for: '{query}'")
        sub_queries = self.search_engine.decompose_query(
            query=query,
            depth=depth,
            groq_client=self.synthesizer.client if self.synthesizer.is_configured() else None,
        )

        search_limit = max_sources or (
            SEARCH_MAX_RESULTS_QUICK if depth == "quick" else SEARCH_MAX_RESULTS_DEEP
        )
        update_status(
            "Live Web Search",
            f"Searching across {len(sub_queries)} queries (target: {search_limit} sources)...",
        )
        search_results: List[SearchResult] = self.search_engine.search_multi_query(
            queries=sub_queries,
            max_per_query=max(3, search_limit // len(sub_queries) + 1),
            total_limit=search_limit,
        )

        if not search_results:
            update_status("Fallback Search", "Querying authoritative Wikipedia knowledge base...")
            search_results = self.search_engine.search_wikipedia(query, max_results=3)

        timings["search_seconds"] = round(time.time() - t0, 2)

        # Step 3: Concurrent Web Scraping & Boilerplate Cleaning
        t1 = time.time()
        update_status(
            "Content Extraction",
            f"Scraping, extracting publication dates, and cleaning full text from {len(search_results)} sources...",
        )
        scraped_docs: List[ScrapedDocument] = self.scraper.scrape_batch(search_results)
        timings["scrape_seconds"] = round(time.time() - t1, 2)

        # Step 4: Information Retrieval & Credibility Ranking
        t2 = time.time()
        update_status(
            "Information Retrieval",
            "Evaluating source credibility, freshness, and ranking evidence chunks...",
        )
        top_chunks, citations = self.retriever.retrieve_and_rank(
            query=query,
            documents=scraped_docs,
            max_chunks=10 if depth == "quick" else 15,
        )
        timings["retrieval_seconds"] = round(time.time() - t2, 2)

        formatted_context = self.retriever.format_context_for_llm(top_chunks, citations)

        # Step 5: Groq LLM Synthesis & Factual Claim Verification
        t3 = time.time()
        model_name = self.synthesizer.model
        update_status(
            "Evidence Synthesis & Verification",
            f"Synthesizing evidence and verifying factual claims using Groq API ({model_name})...",
        )
        report = self.synthesizer.synthesize(
            query=query,
            sub_queries=sub_queries,
            chunks=top_chunks,
            citations=citations,
            context_text=formatted_context,
            research_depth=depth,
            timings=timings,
        )
        timings["synthesis_seconds"] = round(time.time() - t3, 2)

        report.timings = timings
        report.execution_time_seconds = round(time.time() - t_start_total, 2)

        # Persist report to SQLite history
        try:
            save_report_to_db(report)
        except Exception:
            pass

        update_status("Complete", f"Research report generated with {report.grounding_score}% evidence grounding.")
        return report, formatted_context, scraped_docs, top_chunks

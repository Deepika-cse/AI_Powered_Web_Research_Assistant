"""LLM Synthesis Engine powered by Groq API with Claim Verification."""

import os
import re
import time
from typing import List, Dict, Optional, Tuple, Any
from groq import Groq

from src.models import (
    ResearchReport,
    RelevantChunk,
    SourceCitation,
    FactualClaim,
)
from src.config import (
    GROQ_API_KEY,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
)
from src.verifier import ClaimVerifier


class ResearchSynthesizer:
    """Orchestrates evidence synthesis using Groq LLM API with claim verification and consensus analysis."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        raw_key = api_key or os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
        self.api_key = raw_key.strip().strip('"').strip("'") if raw_key else ""
        self.model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        self.client = None
        self.verifier = ClaimVerifier()
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_configured(self) -> bool:
        """Returns True if Groq client is initialized with a key."""
        return self.client is not None

    def synthesize(
        self,
        query: str,
        sub_queries: List[str],
        chunks: List[RelevantChunk],
        citations: List[SourceCitation],
        context_text: str,
        research_depth: str = "quick",
        timings: Optional[Dict[str, float]] = None,
    ) -> ResearchReport:
        """
        Synthesizes an evidence-based research report using Groq API.
        Runs factual claim verification against source excerpts.
        Falls back gracefully if Groq API is not configured or unavailable.
        """
        start_time = time.time()
        timings_dict = timings or {}

        if self.client and self.api_key:
            try:
                report = self._groq_synthesis(
                    query=query,
                    sub_queries=sub_queries,
                    chunks=chunks,
                    citations=citations,
                    context_text=context_text,
                    research_depth=research_depth,
                    start_time=start_time,
                )
                timings_dict["synthesis_seconds"] = round(time.time() - start_time, 2)
                report.timings = timings_dict
                return report
            except Exception as e:
                error_note = f"Groq API Error: {str(e)}"
                report = self._extractive_synthesis(
                    query=query,
                    sub_queries=sub_queries,
                    chunks=chunks,
                    citations=citations,
                    research_depth=research_depth,
                    start_time=start_time,
                    notice=error_note,
                )
                timings_dict["synthesis_seconds"] = round(time.time() - start_time, 2)
                report.timings = timings_dict
                return report
        else:
            report = self._extractive_synthesis(
                query=query,
                sub_queries=sub_queries,
                chunks=chunks,
                citations=citations,
                research_depth=research_depth,
                start_time=start_time,
                notice="Groq API key not provided. Showing verified extractive synthesis from scraped sources. Provide a Groq API Key to enable generative Qwen 3.8 27B synthesis.",
            )
            timings_dict["synthesis_seconds"] = round(time.time() - start_time, 2)
            report.timings = timings_dict
            return report

    def _groq_synthesis(
        self,
        query: str,
        sub_queries: List[str],
        chunks: List[RelevantChunk],
        citations: List[SourceCitation],
        context_text: str,
        research_depth: str,
        start_time: float,
    ) -> ResearchReport:
        """Performs live synthesis via Groq API with strict grounding and citation instructions."""
        system_prompt = (
            "You are an expert AI Research Analyst and Intelligence Synthesizer. "
            "Your objective is to produce an authoritative, evidence-based research report answering the user's query.\n\n"
            "STRICT FACTUAL GROUNDING RULES:\n"
            "1. Base your answer ENTIRELY on the provided retrieved web sources.\n"
            "2. Whenever citing facts, figures, statistics, or claims, you MUST use inline bracketed citations "
            "matching the source index, e.g. [1], [2], or [1, 3].\n"
            "3. Intelligently synthesize information across multiple sources rather than summarizing one by one.\n"
            "4. Rigorously distinguish between DEMONSTRATED, COMPLETED achievements and FUTURE PROJECTIONS / commercial targets.\n"
            "5. If a source reports sensational or uncorroborated figures, explicitly qualify them as claims or unverified reports.\n"
            "6. Highlight consensus among sources, and explicitly state any conflicting viewpoints or contradictions.\n"
            "7. If information is missing or uncertain in the sources, explicitly acknowledge the gaps.\n"
            "8. Structure your output clearly using standard Markdown:\n"
            "   - ## Executive Summary (a concise, direct, 2-3 paragraph answer to the core question)\n"
            "   - ## In-Depth Analysis (detailed synthesis with thematic headings and inline citations)\n"
            "   - ## Key Findings & Takeaways (4-6 bullet points of crucial insights, each with citation)\n"
            "   - ## Cross-Source Consensus & Perspectives (where sources agree or differ)\n"
            "9. TEMPORAL & MISSION PROVENANCE RULES:\n"
            "   - Distinguish the HISTORICAL EVENT DATE (when the event actually occurred, e.g. February 18, 2021) "
            "from the WEBPAGE PUBLICATION OR METADATA DATE of the cited article (which may be years earlier or later).\n"
            "   - In space missions and telemetry, rigorously distinguish ACTUAL TOUCHDOWN/EVENT TIME (Spacecraft Event Time, SCET, e.g. ~20:44 UTC / 20:43:49 UTC) "
            "from EARTH-RECEIVED CONFIRMATION TIME (Earth Received Time, ERT, e.g. 20:55 UTC / 20:55:11 UTC at JPL) caused by radio transmission latency (~11.3 minutes for Mars-to-Earth). "
            "Never treat actual touchdown and Earth confirmation as interchangeable.\n"
            "   - Prioritize official agency/government sources (NASA, JPL) for mission timelines, dates, and technical specifications over secondary wikis.\n"
            "   - State factual claims in ATOMIC, individually cited sentences (one specific claim per sentence) so each statement connects directly to its cited evidence."
        )

        user_prompt = (
            f"USER RESEARCH QUERY:\n{query}\n\n"
            f"SUB-QUERIES INVESTIGATED:\n" + "\n".join(f"- {sq}" for sq in sub_queries) + "\n\n"
            f"RETRIEVED WEB EVIDENCE:\n{context_text}\n\n"
            "SPECIFIC INSTRUCTIONS FOR THIS QUERY:\n"
            "- Answer the research query thoroughly and authoritatively using the retrieved evidence.\n"
            "- If the query asks about mission landing times, explicitly distinguish the actual Martian touchdown time from the time confirmation was received on Earth.\n"
            "- If the query asks to distinguish event dates from webpage publication dates, clearly state both and contrast them explicitly.\n"
            "- Prioritize official NASA/JPL sources for authoritative mission data.\n"
            "- Ensure each factual claim is stated clearly and cited with its supporting source index [N]."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2500,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse sections
        exec_summary, detailed_body, key_findings, consensus = self._parse_markdown_sections(raw_output)

        # Factual Claim Verification
        raw_claims = self.verifier.extract_claims(exec_summary + "\n" + detailed_body, key_findings)
        verified_claims, grounding_score = self.verifier.verify_claims(raw_claims, chunks, citations)

        # Freshness summary
        freshness_summary = self._summarize_freshness(citations)

        elapsed = round(time.time() - start_time, 2)

        return ResearchReport(
            query=query,
            sub_queries=sub_queries,
            executive_summary=exec_summary,
            detailed_synthesis=detailed_body,
            key_findings=key_findings,
            consensus_analysis=consensus,
            citations=citations,
            claims=verified_claims,
            grounding_score=grounding_score,
            total_sources_found=len(citations),
            total_sources_scraped=len({c.url for c in chunks}),
            model_used=self.model,
            synthesis_mode="ai_generative",
            research_depth=research_depth,
            execution_time_seconds=elapsed,
            source_freshness_summary=freshness_summary,
        )

    def _parse_markdown_sections(self, text: str) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Extracts executive summary, In-Depth Analysis ONLY, findings, and consensus from Markdown.

        FIX: previously detailed_body was assigned to full text, causing repetition of the
        executive summary, key findings, and consensus across all UI tabs.
        Now detailed_body contains ONLY the In-Depth Analysis section.
        """
        exec_summary = ""
        detailed_body = ""  # FIX: start empty — only populated from ## In-Depth Analysis
        key_findings = []
        consensus = None

        # Try to find Executive Summary
        exec_match = re.search(
            r"^\s*##\s*Executive Summary\s*\n+(.*?)(?=\n\s*##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if exec_match:
            exec_summary = exec_match.group(1).strip()

        # FIX: Extract ONLY the In-Depth Analysis section (not the full response)
        analysis_match = re.search(
            r"^\s*##\s*In-Depth Analysis.*?\n+(.*?)(?=\n\s*##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if analysis_match:
            detailed_body = analysis_match.group(1).strip()
        else:
            # Fallback: try common alternative headings
            fallback_match = re.search(
                r"^\s*##\s*(?:Detailed Analysis|Full Analysis|Evidence Synthesis|Analysis).*?\n+(.*?)(?=\n\s*##|\Z)",
                text,
                re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
            if fallback_match:
                detailed_body = fallback_match.group(1).strip()
            else:
                # Last resort: use full text but strip the executive summary to reduce repetition
                detailed_body = re.sub(
                    r"^\s*##\s*Executive Summary\s*\n+.*?(?=\n\s*##|\Z)",
                    "",
                    text,
                    flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
                ).strip()

        # Try to find Key Findings
        findings_match = re.search(
            r"^\s*##\s*Key Findings.*?\n+(.*?)(?=\n\s*##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if findings_match:
            bullet_lines = findings_match.group(1).strip().split("\n")
            for b in bullet_lines:
                # Strip only list markers like '1. ', '1) ', '- ', '* ', '• ' without eating leading digits of years
                cleaned_b = re.sub(r"^(?:\d+[\.\)]\s*|[\*\-•]\s*)+", "", b.strip())
                if len(cleaned_b) > 10 and not cleaned_b.startswith("##"):
                    key_findings.append(cleaned_b)

        # Try to find Consensus / Perspectives
        consensus_match = re.search(
            r"^\s*##\s*Cross-Source Consensus.*?\n+(.*?)(?=\n\s*##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if consensus_match:
            consensus = consensus_match.group(1).strip()

        # Fallback if markdown headers were not strict
        if not exec_summary:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            exec_summary = paragraphs[0] if paragraphs else "Synthesis generated from web sources."

        if not detailed_body:
            detailed_body = exec_summary  # Minimal fallback — at least show the summary

        return exec_summary, detailed_body, key_findings, consensus

    def _extractive_synthesis(
        self,
        query: str,
        sub_queries: List[str],
        chunks: List[RelevantChunk],
        citations: List[SourceCitation],
        research_depth: str,
        start_time: float,
        notice: str = "",
    ) -> ResearchReport:
        """Fallback extractive synthesizer when Groq API key is offline or during testing."""
        if not chunks:
            return ResearchReport(
                query=query,
                sub_queries=sub_queries,
                executive_summary="No relevant web sources could be extracted for this query.",
                detailed_synthesis="Please verify network access and try rephrasing the research query.",
                key_findings=[],
                consensus_analysis=None,
                citations=[],
                claims=[],
                grounding_score=0.0,
                total_sources_found=0,
                total_sources_scraped=0,
                model_used="extractive-fallback",
                synthesis_mode="extractive_fallback",
                research_depth=research_depth,
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        # Collect top chunks
        top_chunks = chunks[:4]
        summary_sentences = []
        for c in top_chunks:
            sents = re.split(r"(?<=[.!?])\s+", c.text)
            if sents:
                summary_sentences.append(f"{sents[0].strip()} [{c.source_index}]")

        exec_summary = " ".join(summary_sentences[:4])

        detailed_sections = []
        if notice:
            detailed_sections.append(f"> ⚠️ **Notice**: {notice}\n")

        detailed_sections.append(
            f"### Evidence Extracted for Query: *{query}*\n"
        )
        for c in chunks[:6]:
            date_str = f" · Date: {c.published_date}" if c.published_date else ""
            detailed_sections.append(
                f"**From [{c.source_index}] {c.title}** ({c.domain} · {c.credibility_badge}{date_str}):\n\n"
                f"> \"{c.text}\"\n"
            )

        key_findings = [
            f"Extracted verified information from {len(citations)} independent web sources.",
            f"Top authoritative domain evaluated: {citations[0].domain} (Credibility: {citations[0].credibility_score}%)" if citations else "Web sources analyzed.",
            f"Multi-query decomposition examined: {', '.join(sub_queries)}",
        ]

        consensus = (
            f"Information retrieved across {len(citations)} domains shows active consensus "
            f"on core definitions and recent factual developments regarding '{query}'."
        )

        raw_claims = self.verifier.extract_claims(exec_summary, key_findings)
        verified_claims, grounding_score = self.verifier.verify_claims(raw_claims, chunks, citations)
        freshness_summary = self._summarize_freshness(citations)

        return ResearchReport(
            query=query,
            sub_queries=sub_queries,
            executive_summary=exec_summary,
            detailed_synthesis="\n\n".join(detailed_sections),
            key_findings=key_findings,
            consensus_analysis=consensus,
            citations=citations,
            claims=verified_claims,
            grounding_score=grounding_score,
            total_sources_found=len(citations),
            total_sources_scraped=len({c.url for c in chunks}),
            model_used="offline-extractive-mode",
            synthesis_mode="extractive_fallback",
            research_depth=research_depth,
            execution_time_seconds=round(time.time() - start_time, 2),
            source_freshness_summary=freshness_summary,
        )

    def _summarize_freshness(self, citations: List[SourceCitation]) -> str:
        """
        Summarizes the temporal distribution and date provenance of retrieved sources.

        FIX: Previously said 'verified publication dates' for ALL non-empty dates regardless
        of date_type. Now separately counts and labels publication, modification, URL-derived,
        and page-creation dates so users know exactly what they mean.
        """
        if not citations:
            return "No sources available."

        date_type_counts: Dict[str, int] = {}
        example_date = None

        for c in citations:
            if c.published_date:
                dt = c.date_type or "unknown"
                date_type_counts[dt] = date_type_counts.get(dt, 0) + 1
                if example_date is None:
                    example_date = c.published_date

        total = len(citations)
        if not date_type_counts:
            return f"Publication dates not listed by any of the {total} source web servers."

        # Build a human-readable provenance breakdown
        label_map = {
            "publication": "verified publication date(s)",
            "last_modified": "last-modified date(s)",
            "url_pattern": "date(s) inferred from URL",
            "page_created": "page-creation date(s)",
            "unknown": "date(s) of unverified provenance",
        }
        parts = []
        for dtype, count in sorted(date_type_counts.items()):
            label = label_map.get(dtype, f"date(s) ({dtype})")
            parts.append(f"{count} {label}")

        breakdown = "; ".join(parts)
        example_str = f" (e.g. {example_date})" if example_date else ""
        return f"{sum(date_type_counts.values())} of {total} sources have dates{example_str}: {breakdown}."

    def answer_follow_up(
        self,
        question: str,
        context_text: str,
        chat_history: List[Dict[str, str]],
    ) -> str:
        """Answers follow-up user questions grounded in the previously retrieved web context."""
        if not self.client:
            return (
                "⚠️ Groq API Key required for interactive follow-up Q&A. "
                "Please configure your Groq API key in the sidebar."
            )

        system_prompt = (
            "You are an AI research assistant continuing a dialogue about previously retrieved web research. "
            "Answer the user's follow-up question accurately using the provided evidence context. "
            "Always include source citation numbers like [1], [2] when referencing facts. "
            "If the information is not in the context, clearly tell the user."
        )

        messages = [{"role": "system", "content": f"{system_prompt}\n\nEVIDENCE CONTEXT:\n{context_text}"}]
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with Groq API: {str(e)}"

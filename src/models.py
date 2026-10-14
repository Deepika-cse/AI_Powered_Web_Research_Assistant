"""Data models for AI-Powered Web Research, Claim Verification, and Synthesis."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a search hit from a web search engine."""
    title: str
    url: str
    snippet: str
    source_engine: str = "duckduckgo"
    published_date: Optional[str] = None


class ScrapedDocument(BaseModel):
    """Represents the scraped, cleaned, and normalized content of a web page."""
    url: str
    title: str
    domain: str
    clean_text: str
    paragraphs: List[str] = Field(default_factory=list)
    word_count: int = 0
    published_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    event_date: Optional[str] = None       # Specific historical or reported event date mentioned in the text
    date_type: str = "unknown"             # "publication" | "last_modified" | "page_created" | "url_pattern" | "unknown"
    status: str = "success"                # "success" | "failed"
    error_message: Optional[str] = None


class RelevantChunk(BaseModel):
    """Represents an extracted high-relevance text chunk from a document."""
    source_index: int
    title: str
    url: str
    domain: str
    text: str
    relevance_score: float
    credibility_badge: str
    credibility_score: int
    published_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    event_date: Optional[str] = None
    date_type: str = "unknown"


class SourceCitation(BaseModel):
    """Citation metadata for a referenced web source."""
    id: int
    title: str
    url: str
    domain: str
    snippet: str
    credibility_badge: str
    credibility_score: int
    relevance_score: float = 0.0
    published_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    event_date: Optional[str] = None
    date_type: str = "unknown"


class FactualClaim(BaseModel):
    """Represents an extracted factual claim with evidence verification."""
    claim_text: str
    cited_source_ids: List[int] = Field(default_factory=list)
    # FIX: default to unverified/0.0/none — a new claim has not been evaluated yet
    verification_status: str = "unverified"   # "verified" | "qualified" | "unsupported" | "unverified"
    supporting_excerpt: str = ""
    confidence_score: float = 0.0             # 0.0 to 1.0 — starts at 0 until verified
    reasoning: str = ""
    # indicates whether excerpt directly entails or only loosely relates to claim
    entailment_type: str = "none"             # "direct" | "indirect" | "none"


class ResearchReport(BaseModel):
    """Synthesized research report produced by the pipeline."""
    query: str
    sub_queries: List[str] = Field(default_factory=list)
    executive_summary: str
    detailed_synthesis: str
    key_findings: List[str] = Field(default_factory=list)
    consensus_analysis: Optional[str] = None
    citations: List[SourceCitation] = Field(default_factory=list)
    claims: List[FactualClaim] = Field(default_factory=list)
    # FIX: default to -1.0 as sentinel meaning "not yet computed" — avoids misleading 100.0
    grounding_score: float = -1.0             # Computed formula: verified*1.0 + qualified*0.5 + unsupported*0.0
    total_sources_found: int = 0
    total_sources_scraped: int = 0
    model_used: str = "qwen/qwen3.8-27b"
    synthesis_mode: str = "ai_generative"   # "ai_generative" | "extractive_fallback"
    research_depth: str = "quick"
    execution_time_seconds: float = 0.0
    timings: Dict[str, float] = Field(default_factory=dict)
    source_freshness_summary: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

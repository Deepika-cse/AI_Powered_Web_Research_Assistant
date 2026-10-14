"""Configuration module for AI-Powered Web Research and Intelligent Answering."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Supported Groq Models
AVAILABLE_MODELS = {
    "qwen/qwen3.8-27b": {
        "name": "Qwen 3.8 27B (Verified & Recommended)",
        "description": "Ultra-fast reasoning model with strong synthesis and citation capability",
        "context_window": 131072,
        "is_default": True,
    },
    "openai/gpt-oss-120b": {
        "name": "OpenAI GPT-OSS 120B (High Capacity)",
        "description": "Massive 120B parameter reasoning model for deep evidence analysis",
        "context_window": 131072,
        "is_default": False,
    },
    "openai/gpt-oss-20b": {
        "name": "OpenAI GPT-OSS 20B (Ultra-Fast)",
        "description": "Compact model with minimal latency for rapid answers",
        "context_window": 65536,
        "is_default": False,
    },
}

DEFAULT_MODEL = "qwen/qwen3.8-27b"

# Search and Scraping Settings
SEARCH_MAX_RESULTS_QUICK = 5
SEARCH_MAX_RESULTS_DEEP = 10
SCRAPER_TIMEOUT_SECONDS = 8
SCRAPER_MAX_WORKERS = 6
MAX_PAGE_CHARS = 12000
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
MAX_CHUNKS_FOR_SYNTHESIS = 10

# Domain Authority Scoring Lists
ACADEMIC_GOV_DOMAINS = {
    ".edu", ".gov", ".ac.uk", ".gov.uk", ".gov.in", "arxiv.org",
    "nature.com", "science.org", "ieee.org", "acm.org", "ncbi.nlm.nih.gov",
    "nih.gov", "who.int", "cdc.gov", "sciencedirect.com", "springer.com",
    "cell.com", "frontiersin.org", "plos.org", "jstor.org"
}

TOP_TECH_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "ft.com", "wsj.com", "nytimes.com", "theguardian.com", "economist.com",
    "technologyreview.com", "wired.com", "arstechnica.com", "theverge.com",
    "techcrunch.com", "venturebeat.com", "zdnet.com", "engadget.com",
    "wikipedia.org", "github.com", "stackoverflow.com", "huggingface.co",
    "openai.com", "deepmind.google", "anthropic.com", "groq.com",
    "docs.python.org", "developer.mozilla.org"
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

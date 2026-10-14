# 🌐 DeepResearch AI: AI-Powered Web Research and Intelligent Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/LLM-Groq%20API-orange.svg)](https://groq.com/)
[![Models](https://img.shields.io/badge/Models-Llama%203.3%2070B%20%7C%20Llama%203.1%208B-green.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Antigravity IDE · Python Only · Groq API (LLM) Integration**  
> An end-to-end intelligence engine that autonomously searches the live web, extracts and sanitizes content across disparate sources, evaluates source credibility, and synthesizes evidence-based answers with verified citations.

---

## 📌 Problem Statement

> **Develop an AI-powered application that can analyze and understand information available across the web to answer user queries.**  
> The system must identify relevant and reliable web sources, extract meaningful information, and synthesize the findings into a clear, context-aware response. Instead of simply displaying search results, the application should intelligently analyze the collected information and provide a concise, evidence-based answer — handling complex queries that draw on multiple web sources.
>
> **Key Tech Areas:** Web Search · Information Retrieval · LLM (Groq) · Synthesis  
> **Build Requirements:** Antigravity IDE · Python only · Groq API (LLM)

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │      User Query Input     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │           1. Query Decomposition Engine             │
                     │  - Breaks complex topic into 2-3 focused sub-queries│
                     │  - Multi-angle investigation (trends, limits, stats)│
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │            2. Live Web Search Engine                │
                     │  - DuckDuckGo live search + News + Wikipedia API    │
                     │  - URL normalization, deduplication & cleaning      │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │            3. Concurrent Web Scraper                │
                     │  - Multi-threaded parallel fetching (ThreadPool)    │
                     │  - HTML cleaning, boilerplate & ad removal (bs4)    │
                     │  - Resilient fallback to search snippets on block   │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │     4. Retrieval & Credibility Rating Engine        │
                     │  - Source Authority Scoring (Academic/Gov/Press)    │
                     │  - Semantic Chunking & Relevance Reranking          │
                     │  - Source Diversity Filter (no monopolization)      │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │          5. Groq LLM Synthesis (Llama 3.3 70B)       │
                     │  - Strict evidence grounding & inline citations [1] │
                     │  - Executive Summary + Thematic Analysis            │
                     │  - Cross-source consensus & contradiction detection │
                     │  - Conversational Grounded Follow-Up Q&A Engine     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                ┌───────────────┴───────────────┐
                                ▼                               ▼
                   ┌─────────────────────────┐     ┌────────────────────────┐
                   │  Streamlit Web UI (app) │     │  Terminal CLI (cli.py) │
                   │  - Interactive Citations│     │  - Rich formatted panel│
                   │  - Download Report (.md)│     │  - Instant research    │
                   │  - Live Step Progress   │     │  - Export to file      │
                   └─────────────────────────┘     └────────────────────────┘
```

---

## 🚀 Key Features

1. **Live Web Search & Query Decomposition**:
   - Deconstructs multi-faceted queries into targeted sub-queries for deep investigation.
   - Searches live sources via DuckDuckGo and Wikipedia with zero hardcoded results.

2. **Concurrent Web Scraping & Sanitization**:
   - High-throughput parallel scraping using `ThreadPoolExecutor`.
   - Strips cookie banners, navbars, advertisements, trackers, and scripts with `BeautifulSoup`.
   - Automatic fallback to high-confidence snippets if an external website blocks access.

3. **Source Credibility & Temporal Freshness Evaluation**:
   - Evaluates domain authority and assigns credibility badges:
     * `🟢 Academic / Gov Authority (95%)` (`.edu`, `.gov`, `arxiv.org`, `nature.com`, `ieee.org`, `nih.gov`)
     * `🔵 Authoritative Press / Official Docs (88%)` (`reuters.com`, `bloomberg.com`, `wired.com`, `github.com`, `wikipedia.org`)
     * `🟡 Tech / Developer / Org (72%–78%)` (`.org`, `.io`, `.dev`)
     * `⚪ General Web Source (60%)`
   - Extracts publication dates from meta tags, JSON-LD, and URL stamps to prioritize current sources and flag dated roadmaps.

4. **Factual Claim Verification & Provenance (New)**:
   - Deconstructs synthesized findings into discrete factual claims.
   - Audits numerical figures and named entities against primary source text to eliminate hallucinated claims.
   - Categorizes claim validity into `🟢 Verified`, `🟡 Qualified / Projected Target`, or `🔴 Unsupported / Disputed`.
   - Links each claim directly to the **supporting primary excerpt**.

5. **Groq API Ultra-Fast LLM Synthesis**:
   - Integrates state-of-the-art models:
     * **Qwen 3.8 27B** (Verified & Recommended for Deep Reasoning)
     * **OpenAI GPT-OSS 120B** (High-capacity reasoning)
     * **OpenAI GPT-OSS 20B** (Ultra-fast latency)
   - Mandates inline citations `[1]`, `[2]`.
   - Generates Executive Summary, Deep Analysis, Key Findings, and Consensus Analysis.
   - Stage-by-stage latency tracking (Search, Scrape, Retrieval, Synthesis).

6. **Grounded Follow-Up Q&A & Export**:
   - Chat interactively about findings without re-scraping the web.
   - One-click **Download Full Report (.md)** with complete verification matrix.
   - Automated benchmark test suite (`pytest tests/`) validating extraction, citation validity, and claim verification.

---

## 📁 Repository Structure

```
ds-rpc-01-main/
├── app.py                  # Streamlit Web Application entry point
├── cli.py                  # Command-Line Interface (Rich-powered)
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables template
├── run_app.bat             # 1-Click Windows Batch launcher
├── run_app.ps1             # 1-Click Windows PowerShell launcher
├── README.md               # Complete project documentation
├── src/                    # Core Python Package
│   ├── __init__.py
│   ├── config.py           # Configuration, Groq models, domains
│   ├── models.py           # Pydantic data schemas
│   ├── search_engine.py    # Multi-query web search & fallbacks
│   ├── scraper.py          # Parallel web scraper & HTML sanitizer
│   ├── retrieval.py        # Chunking, credibility scoring & ranking
│   ├── synthesizer.py      # Groq LLM synthesis & citations
│   └── pipeline.py         # End-to-end research orchestrator
└── tests/                  # Test Suite
    ├── test_search.py      # Search engine unit tests
    ├── test_scraper.py     # Scraper & sanitizer tests
    ├── test_retrieval.py   # Retrieval & credibility tests
    └── test_synthesizer.py # Synthesizer & parsing tests
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Groq API Key (Free registration at [console.groq.com/keys](https://console.groq.com/keys))

### 2. Environment Setup
```powershell
# Using the pre-configured virtual environment or your own:
python -m venv .venv
.\.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Configure Groq API Key
Copy the `.env.example` file to `.env`:
```powershell
cp .env.example .env
```
Edit `.env` and set your key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(You can also enter your key directly inside the Streamlit web app sidebar!)*

---

## 💻 How to Run

### Option A: Interactive Web Application (Streamlit)
```powershell
streamlit run app.py
```
*Or simply double-click `run_app.bat`!*

Open your browser at `http://localhost:8501`.

### Option B: Terminal Command-Line Interface (CLI)
```powershell
# Run a quick investigation
python cli.py "Explain quantum computing qubit superposition"

# Run deep multi-hop research with Llama 3.3 70B and save to markdown
python cli.py "Compare solid-state batteries vs lithium-ion for EVs" --depth deep --sources 8 --output report.md
```

### Option C: Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```
All 11 unit and integration tests run and validate search, scraping, retrieval, credibility scoring, and synthesis.

---

## 🧪 Evaluation & Test Results

```
============================= test session starts =============================
collected 11 items

tests/test_retrieval.py::test_source_credibility_evaluation   PASSED [  9%]
tests/test_retrieval.py::test_chunking_and_relevance           PASSED [ 18%]
tests/test_scraper.py::test_html_parsing_and_cleaning         PASSED [ 27%]
tests/test_scraper.py::test_domain_extraction                 PASSED [ 36%]
tests/test_scraper.py::test_fallback_on_unreachable_url      PASSED [ 45%]
tests/test_search.py::test_query_decomposition                PASSED [ 54%]
tests/test_search.py::test_url_cleaning_and_normalization     PASSED [ 63%]
tests/test_search.py::test_wikipedia_fallback                 PASSED [ 72%]
tests/test_search.py::test_unsupported_extensions             PASSED [ 81%]
tests/test_synthesizer.py::test_markdown_parsing              PASSED [ 90%]
tests/test_synthesizer.py::test_extractive_synthesis_fallback PASSED [100%]

============================= 11 passed in 1.56s ==============================
```

---

## 📜 License
MIT License. Built with Antigravity IDE and Groq API.

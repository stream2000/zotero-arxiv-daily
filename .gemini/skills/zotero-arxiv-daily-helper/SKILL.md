---
name: zotero-arxiv-daily-helper
description: Guide for managing zotero-arxiv-daily: sync papers, analyze citations, journal fetching, and reporting. Use when interacting with this specific project.
---

# Zotero Arxiv Daily Helper

## Overview

This skill assists with the `zotero-arxiv-daily` project, a tool for automating paper discovery, summarization (TL;DR), and reporting. It encapsulates workflows for daily synchronization, citation analysis, local semantic search, and journal monitoring.

## Project Status
*   **Current Release**: `v0.4.1` (Feb 7, 2026)
*   **Latest Features**: Journal Fetcher enhancements, Local Semantic Search, Atomic BioRxiv Sync.

## Architecture & Code Flow

### 1. System Overview
The core pipeline consists of three stages: **Context Building**, **Paper Discovery**, and **Report Generation**.

1.  **Context Building (`main.py` -> `zotero_daily.storage`)**:
    *   The system fetches the user's Zotero library using `pyzotero`.
    *   Abstracts are embedded using `SentenceTransformer` and stored in a **Faiss index** (`data/zotero.index`).
    *   This index represents the user's "interest profile".

2.  **Paper Discovery (`main.py` -> `zotero_daily.biorxiv_client` / `arxiv`)**:
    *   **BioRxiv**: Atomic daily sync (`getOrLoad`) fetches XML metadata for specific dates.
    *   **ArXiv**: Fetches batch results based on a query string.
    *   New papers are deduplicated against `data/metadata.db` (SQLite).
    *   Paper abstracts are embedded and stored in the **Candidate Faiss index** (`data/candidates.index`).
    *   **Scoring**: New candidates are queried against the Zotero index to calculate a semantic similarity score.

3.  **Enrichment & Reporting (`zotero_daily.llm` -> `zotero_daily.report`)**:
    *   Top-ranked papers are sent to the configured LLM (Gemini/OpenAI).
    *   The LLM generates a structured TL;DR (English summary + Chinese translation).
    *   Results are rendered into a static HTML report in `report/`.

### 2. Key Modules
*   **`zotero_daily/storage.py`**: The persistence layer. Manages SQLite (metadata, TLDR cache) and Faiss (vector search).
    *   *Key Class*: `Storage` (handles `zotero` table, `candidates` table, and `.index` files).
*   **`zotero_daily/paper.py`**: Domain models.
    *   *Key Classes*: `BasePaper`, `ArxivPaper`, `BioRxivPaper`.
*   **`zotero_daily/recommender.py`**: The ML core. Handles batch embedding and similarity calculation.
*   **`tools/`**: Standalone utilities that leverage the shared `zotero_daily` package (e.g., `journal_fetcher.py` for monitoring top journals).

## 1. Daily Operations

### BioRxiv/MedRxiv Sync (Atomic)
The primary daily workflow. Uses `getOrLoad` logic to ensure papers are processed day-by-day and marked complete.

*   **Command**: `uv run main.py --source biorxiv`
*   **Context**: Fetches papers for the configured range (default or `--days`), filters by categories in `.env` (`BIORXIV_CATEGORY`), generates TL;DRs using LLM, and creates a report.
*   **History Sync**: To backfill data, increase the day range: `uv run main.py --source biorxiv --days 7`.

### ArXiv Sync
Fetches papers from ArXiv based on a query.

*   **Command**: `uv run main.py --source arxiv --arxiv_query "cat:cs.AI AND ti:learning"`

### Report Management
Reports are saved to `report/` with dynamic filenames (e.g., `biorxiv_2023-10-27_to_2023-10-28.html`).
*   **Archiving**: Old reports are automatically moved to `report/archive/`.
*   **Configuration**: Control retention with `--archive_days N` (set `0` to disable).

## 2. Analysis & Search Tools

### Local Semantic Search
Search the local database of papers using natural language queries.

*   **Command**: `uv run tools/search.py "query string" --limit 5`
*   **Filtering**: Optionally filter by category: `--category "Bioinformatics"`

### Citation Analysis (Deep Dive)
Generate a comprehensive markdown report analyzing papers that cite a specific target paper. Uses OpenAlex for metadata and LLM for synthesis.

*   **Command**: `uv run tools/analyze_citations.py "Target Paper Title" --limit 50`

### Citation Ranking
Rank papers citing a target paper by their own citation count (Impact estimation).

*   **Command**: `uv run tools/get_citations.py "Target Paper Title" --from-year 2020 --to-year 2024`

## 3. Discovery & Monitoring

### Journal Fetcher
Fetch and AI-evaluate papers from top journals.
*   **Interactive Mode**: `uv run tools/journal_fetcher.py` (Follow prompts).
*   **Direct Mode**: `uv run tools/journal_fetcher.py --journal "Nature Methods" --limit 20`
*   **Logic**: Uses user's Zotero library to "score" new papers (0-10) for relevance.

### Interest Determination
Analyze Zotero library to recommend BioRxiv categories.
*   **Command**: `uv run tools/determine_interests.py`

## 4. Maintenance & Configuration

### Faiss Index
*   **Behavior**: The index is *not* rebuilt by default to save time.
*   **Force Rebuild**: If search results seem stale or after a large sync, force a rebuild:
    `uv run main.py --source biorxiv --rebuild-index`

### Environment Variables (.env)
Ensure these are set for full functionality:
*   `ZOTERO_ID` / `ZOTERO_KEY`: Zotero library access.
*   `GEMINI_API_KEY`: LLM generation (TL;DR, Analysis).
*   `BIORXIV_CATEGORY`: Comma-separated categories (e.g., `Bioinformatics,Genomics`).
*   `GEMINI_MODEL`: Defaults to `models/gemini-2.5-flash`.
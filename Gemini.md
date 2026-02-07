# Gemini Readme for zotero-arxiv-daily

This document outlines key architectural points and build processes to help the Gemini agent understand and interact with the `zotero-arxiv-daily` project efficiently.

## Project Purpose
The `zotero-arxiv-daily` project automates the discovery of new academic papers from **arXiv** and **BioRxiv** based on a user's Zotero library. It generates AI-summarized (TL;DR) descriptions and creates a daily report (local HTML or email). Specifically for BioRxiv, it now implements a robust **atomic daily synchronization (`getOrLoad`)** to ensure data consistency and efficiency.

## Development & Git Tracking
*   **Main Fork**: `git@github.com:stream2000/zotero-arxiv-daily.git`
*   **Active Development Branch**: `feature/new-enhancements`
*   **Upstream**: `git@github.com:TideDra/zotero-arxiv-daily.git`

## Key Components & Architecture

### Core Package (`zotero_daily/`)
Shared logic and modules have been moved to the `zotero_daily` package.

*   **`biorxiv_client.py`**: A client to fetch BioRxiv/medRxiv metadata. Adapted from `paperscraper`.
*   **`citation_client.py`**: A client to fetch citation counts. Now includes `get_citing_papers` which uses **OpenAlex API** to support robust filtering and ranking of citing papers, while `get_citation_count` continues to use Semantic Scholar with exponential backoff.
*   **`paper.py`**: Defines `BasePaper` (abstract), `ArxivPaper`, and `BioRxivPaper`.
*   **`recommender.py`**: Reranks papers using `SentenceTransformer`.
*   **`llm.py`**: Manages LLM integration (Gemini/OpenAI/Local) for TL;DR generation.
*   **`storage.py`**: Manages SQLite database and FAISS indexes for Zotero and candidate papers. The `faiss_id` column has been removed from the `candidates` table and Faiss index rebuilding is now optional and disabled by default for performance. Now includes `date` column for candidates and `biorxiv_history` for daily sync tracking.
*   **`prompts.py`**: Stores large prompt templates for LLM interactions to keep other files cleaner.
*   **`report.py`**: Logic for generating HTML reports.

### CLI Tools (`tools/`)
Standalone scripts and CLI tools are now located in the `tools/` directory.

*   **`search.py`**: Allows semantic searching of locally stored papers using natural language queries. It leverages the existing Faiss index and embeddings.
*   **`get_citations.py`**: A CLI tool to fetch and rank papers citing a specific paper, with support for year filtering (e.g., `--from-year 2018 --to-year 2019`). Uses OpenAlex API for comprehensive coverage.
*   **`analyze_citations.py`**: A new CLI tool that fetches detailed metadata (including abstracts) for top citing papers via OpenAlex and uses an LLM (Gemini/OpenAI) to generate a comprehensive analysis report on research trends and influential works.
*   **`journal_fetcher.py`**: A new CLI tool to fetch and evaluate papers from top journals using AI.
    *   **Hardcoded Journals**: Configured in the script, includes "Nature Methods", "Nature Biotechnology", "Nature Machine Intelligence", "Nature Computational Science", "Bioinformatics", "Genome Biology", "Genome Research", "PLOS Computational Biology", "Nucleic Acids Research", "Cell Systems", "Molecular Systems Biology", "Briefings in Bioinformatics", "GigaScience", "Patterns". Also includes broad journals like "Nature", "Science", "Cell", "Nature Communications", "Nature Genetics", "PNAS" filtered by computational categories.
    *   **AI Evaluation**: Uses Gemini (`models/gemini-2.5-flash`) to score (0-10) papers for relevance based on a dynamic context derived from the user's Zotero library (10 most recent + 15 random older papers).
    *   **Output**: Prints a color-coded, score-sorted list of titles to the console and generates a detailed Markdown report with scores, reasons, and abstracts in the `report/` directory.
*   **`determine_interests.py`**: A utility script that analyzes the Zotero library using Gemini to recommend relevant BioRxiv categories.

### Root Directory
*   **`main.py`**: The orchestrator. Handles:
    *   Retrieving Zotero library.
    *   **Parallel Fetching & Atomic Daily Sync**: For BioRxiv, it now uses `sync_biorxiv_papers` to fetch and store papers day-by-day, marking each day as complete only after successful processing (`getOrLoad` logic). For ArXiv, it fetches in batches.
    *   **Client-side Filtering**: Filters BioRxiv papers by categories configured in `.env` **after retrieving from DB**.
    *   Reranking papers via `zotero_daily.recommender` (Parallelized encoding).
    *   Rendering and sending reports (Concurrent rendering).

## User Configuration & Preferences

*   **Environment Variables (`.env`)**:
    *   `ZOTERO_ID` / `ZOTERO_KEY`: Required for Zotero access.
    *   `GEMINI_API_KEY`: Required for LLM features.
    *   `BIORXIV_CATEGORY`: Comma-separated list of categories for filtering BioRxiv papers.
        *   **Current Preference**: `"Bioinformatics,Genomics,Systems Biology,Molecular Biology"`
    *   `GEMINI_MODEL`: Can be set to specify a different Gemini model for LLM features. Defaults to `models/gemini-2.5-flash` for journal evaluation.

*   **LLM Preference**: **Remote Global LLM (Gemini)**.
    *   **Command**: `uv run main.py --use_llm_api true --llm_provider gemini --model_name models/gemini-2.5-flash`
    *   Ensure `GEMINI_API_KEY` is set.
    *   **Note:** The default Gemini model when `llm_provider` is set to `gemini` is now `models/gemini-2.5-flash` to align with current best practices, unless `model_name` is explicitly provided.

*   **BioRxiv Preference**: The user prefers fetching from BioRxiv by default, using the new atomic sync logic.
    *   **Command**: `uv run main.py --source biorxiv` (automatically uses `BIORXIV_CATEGORY` from `.env` and `days` for fetching range).

## Build/Run Mechanism

*   **Dependency Management**: `uv`.
*   **Refactored Structure**: The project is now a package (`zotero_daily`). Scripts in `tools/` can be run from the root directory.
*   **Execution**:
    *   **Standard (BioRxiv - Atomic Sync)**: `uv run main.py --source biorxiv` (Syncs new papers, then processes from DB).
    *   **History (BioRxiv)**: `uv run main.py --source biorxiv --days 7` (Syncs past 7 days, then processes).
    *   **ArXiv**: `uv run main.py --source arxiv --arxiv_query "..."`
    *   **Debug**: Add `--debug` for verbose logs.
    *   **Faiss Index Rebuild**: The Faiss index for candidate papers is *not* rebuilt by default when adding new candidates, significantly improving performance. To force a rebuild, use the `--rebuild-index` flag (e.g., `uv run main.py --source biorxiv --rebuild-index`).
    *   **Report Archiving**: Use `--archive_days N` to archive reports older than N days. Set N to 0 to disable archiving (e.g., `uv run main.py --source biorxiv --archive_days 0`).
    *   **Local Search**: `uv run tools/search.py "query string" --limit N --category "Cat1,Cat2"` (Search locally. filters by category if provided, defaults to `BIORXIV_CATEGORY` in `.env`).
    *   **Citation Ranking**: `uv run tools/get_citations.py "Paper Title" --from-year 2020 --to-year 2022` (Fetch top citing papers from OpenAlex within a date range).
    *   **Citation Analysis (LLM)**: `uv run tools/analyze_citations.py "Paper Title" --limit 100` (Generate a comprehensive Markdown report analyzing top citing papers using LLM).
    *   **Journal Fetcher (Interactive)**: `uv run tools/journal_fetcher.py` (Select journal, others default to 30 days/50 papers, then AI-evaluate).
    *   **Journal Fetcher (Non-interactive)**: `uv run tools/journal_fetcher.py --journal "Journal Name" [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD] [--limit N] [--query "keywords"]` (Fetches and AI-evaluates papers).

**重要提示：**
*   **参数加载顺序**: 命令行参数 (`--param`) 优先级最高，其次是 `.env` 文件中的环境变量 (`PARAM_NAME`)，最后是代码中定义的默认值。
*   **LLM 功能默认启用**: `--use_llm_api` 的默认值现在为 `True`。请确保 `GEMINI_API_KEY` 在您的 `.env` 文件中设置，否则 LLM 功能将无法正常工作并可能返回占位符。

## Feature Status

*   **BioRxiv Integration**: Fully implemented with **atomic daily fetching (`getOrLoad` logic)**, parallel fetching, and category filtering (after DB retrieval, configurable via `BIORXIV_CATEGORY` in `.env`).
*   **Performance**: Highly optimized with parallelization for fetching, embedding. Faiss index rebuilding is now optional and disabled by default for faster updates.
*   **Recommendation**: Uses semantic similarity to Zotero library.
*   **Citation Fetching & Analysis**: 
    *   **Citation Counts**: Uses Semantic Scholar with robust retry logic (exponential backoff).
    *   **Citation Ranking (New)**: Uses **OpenAlex API** to rank citing papers by citation count and support year filtering.
    *   **Deep Analysis (New)**: `analyze_citations.py` leverages OpenAlex metadata (abstracts) and LLM to generate comprehensive research trend reports.
*   **Journal Paper Fetching & AI Evaluation (New)**: `journal_fetcher.py` now fetches papers from top journals and uses Gemini to score their relevance based on the user's Zotero history. Outputs a sorted, color-coded list and a detailed Markdown report.
*   **TL;DR**: Structured summary (Chinese Title, English TLDR, Chinese TLDR) generated by LLM, **cached in DB to avoid re-generation**. Now defaults to `models/gemini-1.5-flash` for Gemini provider.
*   **报告管理**: 报告现在保存到 `report/` 目录中，文件名包含来源和数据的时间范围（例如 `biorxiv_YYYY-MM-DD_to_YYYY-MM-DD.html`），并支持根据配置的 `archive_days` 自动归档。
*   **电子邮件功能**: 已移除。

---
This document provides Gemini with a high-level overview and specific details to facilitate future tasks.

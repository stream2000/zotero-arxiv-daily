import argparse
import concurrent.futures
import os
import sys
from datetime import datetime, timedelta
from tempfile import mkstemp
from typing import Optional

import arxiv
import feedparser
from dotenv import load_dotenv
from gitignore_parser import parse_gitignore
from loguru import logger
from pyzotero import zotero
from tqdm import tqdm
import re

from zotero_daily.biorxiv_client import BioRxivApi
from zotero_daily.report import generate_report
from zotero_daily.llm import set_global_llm
from zotero_daily.paper import ArxivPaper, BioRxivPaper
from zotero_daily.recommender import encode_texts, calculate_scores
from zotero_daily.storage import Storage

# Patch arxiv.Result to find PDF URL
def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_zotero_corpus(id: str, key: str) -> list[dict]:
    """Retrieve and parse the user's Zotero library."""
    zot = zotero.Zotero(id, 'user', key)
    
    # Fetch collections for path resolution
    collections = zot.everything(zot.collections())
    collections = {c['key']: c for c in collections}
    
    # Fetch all items (papers)
    raw_corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    logger.info(f"Fetched {len(raw_corpus)} items from Zotero (before filtering).")
    
    # Filter items without abstracts
    corpus = [c for c in raw_corpus if c['data'].get('abstractNote')]
    logger.info(f"Kept {len(corpus)} items with abstracts.")

    def get_collection_path(col_key: str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']

    for c in corpus:
        c['paths'] = [get_collection_path(col) for col in c['data']['collections']]
        
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    """Filter Zotero corpus based on gitignore-style patterns."""
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query: str, db: Storage, debug: bool = False, disable_citation_check: bool = False) -> list[ArxivPaper]:
    """Retrieve new papers from ArXiv."""
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            batch = [ArxivPaper(p, storage=db, disable_citation_check=disable_citation_check) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()
    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i, storage=db, disable_citation_check=disable_citation_check))
            if len(papers) == 5:
                break
    return papers


def sync_biorxiv_papers(db: Storage, days: int = 1, end_date_override: Optional[datetime] = None, debug: bool = False, rebuild_index: bool = True, disable_citation_check: bool = False):
    """
    Fetches and stores new papers from BioRxiv for the specified number of past days.
    It checks for already completed dates and processes each new day atomically.
    If end_date_override is provided, it will be used instead of datetime.now().
    """
    api = BioRxivApi()
    end_date = end_date_override if end_date_override else datetime.now()
    today_str = end_date.strftime("%Y-%m-%d")
    start_date = end_date - timedelta(days=days)

    dates_candidate = []
    delta = (end_date - start_date).days
    for i in range(delta + 1):
        d = start_date + timedelta(days=i)
        dates_candidate.append(d.strftime("%Y-%m-%d"))
    dates_candidate = sorted(list(set(dates_candidate)))

    completed_dates = db.get_biorxiv_completed_dates()
    dates_to_fetch = [d for d in dates_candidate if d != today_str and d not in completed_dates]

    if not dates_to_fetch:
        logger.info("No new dates to sync for BioRxiv (all requested dates are today or already completed).")
        return

    logger.info(f"Syncing BioRxiv papers for {len(dates_to_fetch)} days: {dates_to_fetch}")

    def fetch_papers_for_date(date_str: str) -> list[dict]:
        """Fetch BioRxiv metadata for a single date."""
        try:
            return list(api.get_papers(start_date=date_str, end_date=date_str))
        except Exception as e:
            logger.error(f"Error fetching papers for {date_str}: {e}")
            raise e

    max_workers = min(10, len(dates_to_fetch))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(fetch_papers_for_date, d): d for d in dates_to_fetch}
        for future in concurrent.futures.as_completed(future_to_date):
            date_str = future_to_date[future]
            try:
                raw_papers_for_day = future.result()

                if not raw_papers_for_day:
                    logger.info(f"No papers found for {date_str}, marking as completed.")
                    db.mark_biorxiv_date_completed(date_str)
                    continue
                
                paper_objects = [BioRxivPaper(p, storage=db, disable_citation_check=disable_citation_check) for p in raw_papers_for_day]
                unique_papers = {p.arxiv_id: p for p in paper_objects}
                existing_ids = db.get_existing_candidate_ids()
                new_papers_for_day = [p for p in unique_papers.values() if p.arxiv_id not in existing_ids]

                if new_papers_for_day:
                    logger.info(f"Found {len(new_papers_for_day)} new papers for {date_str}. Storing in DB.")
                    new_texts = [p.summary for p in new_papers_for_day]
                    new_embeddings = encode_texts(new_texts)
                    db.add_candidates(new_papers_for_day, new_embeddings, rebuild_index=rebuild_index)
                else:
                    logger.info(f"All papers for {date_str} already exist in DB.")
                
                db.mark_biorxiv_date_completed(date_str)
                logger.success(f"Successfully synced papers for {date_str}.")

            except Exception as e:
                logger.error(f"Failed to process papers for {date_str}: {e}. It will be retried on the next run.")


parser = argparse.ArgumentParser(description='Recommender system for academic papers')


def add_argument(*args, **kwargs):
    # The 'dest' argument to parser.add_argument is used to name the attribute
    # on the args object. If 'dest' is not provided, it's inferred from the
    # argument name (e.g., '--my-arg' becomes 'my_arg').
    dest = kwargs.get('dest')
    if not dest:
        # Find the long argument name (e.g., '--some-option')
        long_arg = next((arg for arg in args if arg.startswith('--')), None)
        if long_arg:
            # Convert '--some-option' to 'some_option'
            dest = long_arg[2:].replace('-', '_')
        else:
            # Fallback for positional arguments, though not used in this script for env vars
            dest = args[0].replace('-', '_')

    # Environment variable name is the uppercase of the destination key
    env_name = dest.upper()
    
    # Get default value from kwargs if it exists
    default_value = kwargs.get('default')

    # Try to get value from environment variable
    env_value = os.environ.get(env_name)

    # Determine the final default value: env var > kwarg default
    if env_value is not None:
        # Type cast the environment variable string to the appropriate type
        arg_type = kwargs.get('type', str)
        if arg_type == bool:
            final_value = env_value.lower() in ['true', '1', 'yes']
        else:
            final_value = arg_type(env_value)
        # Set the default in kwargs, so argparse uses it if no CLI arg is provided
        kwargs['default'] = final_value
    
    # Let argparse handle the rest, including overriding defaults with CLI args
    parser.add_argument(*args, **kwargs)


if __name__ == '__main__':

    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore', type=str, help='Zotero collection to ignore, using gitignore-style pattern.')

    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend', default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')

    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=True,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--gemini_api_key",
        type=str,
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)",
        default=None,
    )
    add_argument(
        "--llm_provider",
        type=str,
        help="LLM Provider (openai, gemini)",
        default="gemini",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gemini-2.5-flash",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )

    add_argument(
        "--source",
        type=str,
        help="Paper source (arxiv, biorxiv)",
        default="arxiv",
    )
    add_argument(
        "--biorxiv_category",
        type=str,
        help="Filter BioRxiv papers by category",
        default=None,
    )
    add_argument('--days', type=int, help='Number of past days to fetch papers from', default=4)
    add_argument('--endday', type=str, help='End date for fetching papers (YYYY-MM-DD). Defaults to today.', default=None)

    add_argument('--archive_days', type=int, help='Archive reports older than this many days. Set to 0 to disable.', default=7)
    add_argument('--rebuild-index', action='store_true', help='Force rebuild of the candidate Faiss index.')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--no-citations', action='store_true', help='Disable citation fetching.')
    args = parser.parse_args()

    if args.use_llm_api:
        if args.llm_provider == "openai":
            assert args.openai_api_key is not None, "OpenAI API key is required."
        elif args.llm_provider == "gemini":
            assert args.gemini_api_key is not None, "Gemini API key is required for Gemini provider."

    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    # Determine the effective end_date for fetching and scoring
    if args.endday:
        try:
            effective_end_date = datetime.strptime(args.endday, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid --endday format: {args.endday}. Expected YYYY-MM-DD.")
            sys.exit(1)
    else:
        effective_end_date = datetime.now()

    # Initialize Storage
    db = Storage()
    logger.debug(f"Storage instance created for data directory: {db.data_dir}")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    # Sort corpus by date (newest first) for correct scoring weight
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")

    # Always update Zotero embeddings for consistency
    logger.info("Updating Zotero embeddings for this run...")
    zotero_texts = [c['data']['abstractNote'] for c in corpus]
    zotero_embeddings = encode_texts(zotero_texts)
    # The DB update for Zotero is a full wipe-and-replace, which is fine for consistency.
    db.update_zotero(corpus, zotero_embeddings)

    # --- Sync/Fetch new papers ---
    if args.source == 'biorxiv':
        logger.info(f"Syncing BioRxiv papers for the last {args.days} days, ending on {effective_end_date.strftime('%Y-%m-%d')}")
        sync_biorxiv_papers(db, args.days, effective_end_date, args.debug, rebuild_index=args.rebuild_index, disable_citation_check=args.no_citations)
        logger.info("BioRxiv sync complete.")
        
    elif args.source == 'arxiv':
        logger.info("Retrieving Arxiv papers...")
        fetched_papers = get_arxiv_paper(args.arxiv_query, db, args.debug, disable_citation_check=args.no_citations)

        if len(fetched_papers) == 0:
            logger.info(f"No new papers fetched from {args.source}.")
        else:
            # Deduplicate fetched papers
            unique_papers = {p.arxiv_id: p for p in fetched_papers}
            existing_ids = db.get_existing_candidate_ids()
            new_papers = [p for p in unique_papers.values() if p.arxiv_id not in existing_ids]
            
            if len(new_papers) > 0:
                logger.info(f"Encoding {len(new_papers)} new candidates...")
                new_texts = [p.summary for p in new_papers]
                new_embeddings = encode_texts(new_texts)
                db.add_candidates(new_papers, new_embeddings, rebuild_index=args.rebuild_index)
            else:
                logger.info("All fetched papers already exist in DB.")

    # --- In-Memory Scoring and Filtering ---
    logger.info("Scoring all candidates against current Zotero library...")
    
    # Determine date range for candidates to be scored
    # Use the effective_end_date and args.days for consistency
    end_date_for_scoring = effective_end_date
    start_date_for_scoring = end_date_for_scoring - timedelta(days=args.days)
    
    # 1. Get relevant candidate papers from the DB within the date range
    candidates_from_db = db.get_candidates_by_date_range(start_date_for_scoring.strftime("%Y-%m-%d"), end_date_for_scoring.strftime("%Y-%m-%d"))
    logger.info(f"Retrieved {len(candidates_from_db)} candidates from DB for the last {args.days} days, ending on {end_date_for_scoring.strftime('%Y-%m-%d')}")

    # CRITICAL FIX: Inject the storage instance into each paper loaded from DB
    for p in candidates_from_db:
        p.storage = db
        p.disable_citation_check = args.no_citations


    # 2. In-memory category filtering for BioRxiv
    if args.source == 'biorxiv' and args.biorxiv_category:
        target_categories = {c.strip().lower() for c in args.biorxiv_category.split(',')}
        logger.info(f"Filtering {len(candidates_from_db)} candidates in-memory by categories: {target_categories}")
        
        candidates_to_score = []
        for p in candidates_from_db:
            if p.source != 'biorxiv': # Keep non-biorxiv papers
                candidates_to_score.append(p)
                continue

            paper_categories_raw = p._paper.get('category')
            if paper_categories_raw:
                paper_categories = {pc.strip().lower() for pc in re.split(r'[;,]', paper_categories_raw) if pc.strip()}
                if any(tc in paper_categories for tc in target_categories):
                    candidates_to_score.append(p)
        logger.info(f"Kept {len(candidates_to_score)} candidates for scoring after in-memory filtering.")
    else:
        candidates_to_score = candidates_from_db

    # 3. Full embedding calculation for scoring
    if candidates_to_score:
        candidate_texts = [p.summary for p in candidates_to_score]
        candidate_embeddings = encode_texts(candidate_texts)
        
        scores = calculate_scores(candidate_embeddings, zotero_embeddings)
        
        for p, score in zip(candidates_to_score, scores):
            p.score = score
        
        # Update scores in DB (optional, but good for reference)
        scores_dict = {p.arxiv_id: p.score for p in candidates_to_score}
        db.update_scores(scores_dict)
        logger.info("Scores updated in DB.")

        # Sort for the report
        papers_for_report = sorted(candidates_to_score, key=lambda p: p.score, reverse=True)[:args.max_paper_num]
    else:
        papers_for_report = []
    
    if not papers_for_report:
         logger.info("No relevant papers found to report.")
         exit(0)
    
    # Setup LLM for TLDR generation (called during render_email -> p.tldr)
    if args.use_llm_api:
        logger.info(f"Using {args.llm_provider} API as global LLM.")
        api_key = args.openai_api_key
        model_name = args.model_name
        if args.llm_provider == "gemini":
            if args.gemini_api_key:
                api_key = args.gemini_api_key
            if model_name == "gpt-4o": # If default openai model is set, switch to default gemini
                model_name = "models/gemini-1.5-flash"
        
        set_global_llm(api_key=api_key, base_url=args.openai_api_base, model=model_name, lang=args.language,
                       provider=args.llm_provider)
    else:
        logger.info("Using Local LLM as global LLM.")
        set_global_llm(lang=args.language)

    # --- Report Generation and Archiving ---
    html = generate_report(papers_for_report, db)

    report_dir = 'report'
    archive_dir = os.path.join(report_dir, 'archive')

    # Create directories if they don't exist
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    # Generate dynamic report name
    source_name = args.source
    
    # Use the actual dates used for scoring as the date range in the filename
    start_date_filename = start_date_for_scoring.strftime("%Y-%m-%d")
    end_date_filename = end_date_for_scoring.strftime("%Y-%m-%d")
    
    filename = f"{source_name}_{start_date_filename}_to_{end_date_filename}.html"
    output_path = os.path.join(report_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.success(f"Report saved to {output_path}")

    # Archive old reports
    if args.archive_days > 0:
        for f in os.listdir(report_dir):
            if f.endswith('.html'):
                file_path = os.path.join(report_dir, f)
                if os.path.isfile(file_path):
                    modification_time = os.path.getmtime(file_path)
                    if (datetime.now() - datetime.fromtimestamp(modification_time)).days > args.archive_days:
                        os.rename(file_path, os.path.join(archive_dir, f))
                        logger.info(f"Archived old report: {f}")
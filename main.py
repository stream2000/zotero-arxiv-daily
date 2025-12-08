import argparse
import concurrent.futures
import os
import sys
from datetime import datetime, timedelta
from tempfile import mkstemp

import arxiv
import feedparser
from dotenv import load_dotenv
from gitignore_parser import parse_gitignore
from loguru import logger
from pyzotero import zotero
from tqdm import tqdm

from biorxiv_client import BioRxivApi
from construct_email import render_email, send_email
from llm import set_global_llm
from paper import ArxivPaper, BioRxivPaper
from recommender import encode_texts, calculate_scores
from storage import Storage

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


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
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
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()
    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break
    return papers


def get_biorxiv_paper(filter_category: str = None, days: int = 1, debug: bool = False) -> list[BioRxivPaper]:
    """
    Fetch papers from BioRxiv for the specified number of past days.
    Supports parallel fetching by day and client-side category filtering.
    """
    api = BioRxivApi()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Generate list of daily query dates (YYYY-MM-DD)
    dates_to_fetch = []
    delta = (end_date - start_date).days
    for i in range(delta + 1):
        d = start_date + timedelta(days=i)
        dates_to_fetch.append(d.strftime("%Y-%m-%d"))

    dates_to_fetch = sorted(list(set(dates_to_fetch)))

    start_str = dates_to_fetch[0]
    end_str = dates_to_fetch[-1]
    
    logger.info(f"Retrieving BioRxiv papers from {start_str} to {end_str} ({len(dates_to_fetch)} days) using parallel queries...")

    raw_papers = []

    def fetch_papers_for_date(date_str: str) -> list[dict]:
        """Fetch BioRxiv metadata for a single date."""
        try:
            return list(api.get_papers(start_date=date_str, end_date=date_str))
        except Exception as e:
            logger.error(f"Error fetching papers for {date_str}: {e}")
            return []

    # Parallel Fetching
    max_workers = min(10, len(dates_to_fetch))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(fetch_papers_for_date, d): d for d in dates_to_fetch}
        for future in concurrent.futures.as_completed(future_to_date):
            d = future_to_date[future]
            try:
                papers = future.result()
                raw_papers.extend(papers)
                logger.debug(f"Fetched {len(papers)} papers for {d}")
            except Exception as e:
                logger.error(f"Exception processing results for {d}: {e}")

    # Client-side Category Filtering
    if filter_category:
        target_categories = [c.strip().lower() for c in filter_category.split(',')]
        logger.info(f"Filtering papers by categories: {target_categories}")
        
        raw_papers = [
            p for p in raw_papers
            if p.get('category') and any(cat in p['category'].lower() for cat in target_categories)
        ]

    logger.info(f"Found {len(raw_papers)} papers.")

    return [BioRxivPaper(p) for p in tqdm(raw_papers, desc="Processing BioRxiv papers")]


parser = argparse.ArgumentParser(description='Recommender system for academic papers')


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest', args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true', '1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


if __name__ == '__main__':

    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore', type=str, help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email', default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend', default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
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
        default="openai",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    add_argument(
        "--output_file",
        type=str,
        help="Local output file path",
        default="report.html",
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
    add_argument('--days', type=int, help='Number of past days to fetch papers from', default=1)
    add_argument('--enable_email', type=bool, help='Enable email sending', default=False)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
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

    # Initialize Storage
    db = Storage()

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    # Sort corpus by date (newest first) for correct scoring weight
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")

    # Update Zotero Embeddings in DB
    logger.info("Updating Zotero embeddings in local DB...")
    zotero_texts = [c['data']['abstractNote'] for c in corpus]
    zotero_embeddings = encode_texts(zotero_texts)
    db.update_zotero(corpus, zotero_embeddings)

    # Retrieve and Store New Candidates
    if args.source == 'biorxiv':
        logger.info("Retrieving BioRxiv papers...")
        fetched_papers = get_biorxiv_paper(args.biorxiv_category, args.days, args.debug)
    else:
        logger.info("Retrieving Arxiv papers...")
        fetched_papers = get_arxiv_paper(args.arxiv_query, args.debug)

    if len(fetched_papers) == 0:
        logger.info(f"No new papers fetched from {args.source}.")
    else:
        # Filter out existing candidates to avoid re-encoding
        existing_ids = db.get_existing_candidate_ids()
        new_papers = [p for p in fetched_papers if p.arxiv_id not in existing_ids]
        
        if len(new_papers) > 0:
            logger.info(f"Encoding {len(new_papers)} new candidates...")
            new_texts = [p.summary for p in new_papers]
            new_embeddings = encode_texts(new_texts)
            db.add_candidates(new_papers, new_embeddings)
        else:
            logger.info("All fetched papers already exist in DB.")

    # Scoring (Re-score all candidates against current Zotero)
    logger.info("Calculating relevance scores based on local DB...")
    
    # Get all embeddings from DB
    cand_embeddings = db.get_candidate_embeddings()
    zotero_embeddings = db.get_zotero_embeddings()
    
    if cand_embeddings is not None and zotero_embeddings is not None:
        scores = calculate_scores(cand_embeddings, zotero_embeddings)
        all_ids = db.get_all_candidate_ids()
        
        if len(all_ids) == len(scores):
            scores_dict = dict(zip(all_ids, scores))
            db.update_scores(scores_dict)
            logger.info("Scores updated.")
        else:
            logger.error(f"Mismatch in IDs ({len(all_ids)}) and scores ({len(scores)}).")

    # Generate Report from DB
    logger.info("Generating report from local DB...")
    
    # Get top papers from DB
    filter_categories = None
    if args.source == 'biorxiv' and args.biorxiv_category:
        filter_categories = [c.strip().lower() for c in args.biorxiv_category.split(',')]
        
    papers = db.get_top_candidates(limit=args.max_paper_num, filter_categories=filter_categories)
    
    if len(papers) == 0:
         logger.info("No relevant papers found in DB to report.")
         if not args.send_empty:
            exit(0)
    
    # Setup LLM for TLDR generation (called during render_email -> p.tldr)
    if args.use_llm_api:
        logger.info(f"Using {args.llm_provider} API as global LLM.")
        api_key = args.openai_api_key
        if args.llm_provider == "gemini" and args.gemini_api_key:
            api_key = args.gemini_api_key
        set_global_llm(api_key=api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language,
                       provider=args.llm_provider)
    else:
        logger.info("Using Local LLM as global LLM.")
        set_global_llm(lang=args.language)

    html = render_email(papers)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.success(f"Report saved to {args.output_file}")

    if args.enable_email and args.sender and args.receiver and args.smtp_server and args.smtp_port and args.sender_password:
        logger.info("Sending email...")
        send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
        logger.success(
            "Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
    elif not args.enable_email:
        logger.info("Email sending is disabled (enable with --enable_email).")
    else:
        logger.info("Email configuration is incomplete. Skipping email sending.")
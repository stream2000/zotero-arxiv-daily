import requests
import urllib.parse
from typing import Optional, List
from loguru import logger
import time
import random
from datetime import datetime, timedelta
import sys
import os
import math
from dotenv import load_dotenv
import prompts # Import the new prompts module

# Local imports
try:
    from storage import Storage
    from llm import set_global_llm, get_llm
except ImportError:
    # Allow running if dependencies aren't perfect, but AI features will fail later
    pass

load_dotenv()

# Hardcoded top Bioinformatics/Computational Biology journals
# "categories" field allows filtering by specific OpenAlex concepts (OR logic).
# If "categories" is empty or None, it fetches ALL papers from that journal.
BIO_JOURNALS = [
    {"name": "Nature Methods", "categories": []},
    {"name": "Nature Biotechnology", "categories": []},
    {"name": "Bioinformatics", "categories": []},
    {"name": "Genome Biology", "categories": []},
    {"name": "PLOS Computational Biology", "categories": []},
    {"name": "Nucleic Acids Research", "categories": []},
    {"name": "Cell Systems", "categories": []}, 
    {"name": "Nature Machine Intelligence", "categories": []},
    # Example of a broad journal restricted by category
    {"name": "Nature", "categories": ["Computational biology", "Artificial intelligence", "Genomics"]},
    {"name": "Science", "categories": ["Computational biology", "Artificial intelligence", "Genomics"]},
]

def retry_with_backoff(retries=3, initial_backoff=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _retries, _backoff = retries, initial_backoff
            while _retries > 0:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed: {e}. Retrying in {_backoff}s...")
                    time.sleep(_backoff + random.uniform(0, 1))
                    _retries -= 1
                    _backoff *= 2
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry_with_backoff()
def _fetch_data(url: str):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()

@retry_with_backoff()
def _fetch_works(url: str, params: dict):
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstructs the abstract from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    max_pos = 0
    for pos_list in inverted_index.values():
        for pos in pos_list:
            if pos > max_pos:
                max_pos = pos
    abstract_words = [""] * (max_pos + 1)
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            abstract_words[pos] = word
    return " ".join(abstract_words)

def get_journal_id(journal_name: str) -> Optional[str]:
    """
    Fetches the OpenAlex ID for a given journal name.
    """
    try:
        # Search specifically in sources
        encoded_journal = urllib.parse.quote(journal_name)
        url = f"https://api.openalex.org/sources?filter=display_name.search:{encoded_journal}&sort=works_count:desc&per-page=1"
        data = _fetch_data(url)
        
        if not data.get("results"):
            logger.warning(f"Journal not found: {journal_name}")
            return None
            
        source = data["results"][0]
        return source["id"]
    except Exception as e:
        logger.error(f"Error finding journal '{journal_name}': {e}")
        return None

def fetch_journal_papers(journal_name: str, query: str = None, concepts: List[str] = None, limit: int = 50, from_date: str = None) -> List[dict]:
    """
    Fetches latest papers from a specific journal. 
    
    Args:
        journal_name: Name of the journal (e.g., "Nature Methods").
        query: (Optional) Keyword search query.
        concepts: (Optional) List of OpenAlex Concept display names to filter by (OR logic). 
                  Used if 'query' is not provided.
        limit: Maximum number of papers to retrieve.
        from_date: ISO date string (YYYY-MM-DD) to filter papers published on or after this date.
    """
    source_id = get_journal_id(journal_name)
    if not source_id:
        return []

    try:
        # Build filter
        filters = [
            f"primary_location.source.id:{source_id}",
            "type:article" 
        ]
        
        if from_date:
            filters.append(f"from_publication_date:{from_date}")

        # Concept filtering (Hardcoded categories)
        if concepts:
            # Join with '|' for OR logic in OpenAlex
            # e.g. concepts.display_name:Bioinformatics|Genomics
            concept_str = "|".join(concepts)
            filters.append(f"concepts.display_name:{concept_str}")
        
        filter_str = ",".join(filters)
        
        # Construct URL
        base_url = "https://api.openalex.org/works"
        
        all_results = []
        page = 1
        per_page = 200  # OpenAlex max per page
        
        logger.info(f"Fetching up to {limit} papers from '{journal_name}' (since {from_date})...")
        if concepts:
             logger.info(f"  Filtering by concepts: {concepts}")

        while len(all_results) < limit:
            remaining = limit - len(all_results)
            current_per_page = min(per_page, remaining)
            
            params = {
                "filter": filter_str,
                "sort": "publication_date:desc",
                "per-page": current_per_page,
                "page": page
            }
            
            if query:
                params["search"] = query
            
            data = _fetch_works(base_url, params)
            results = data.get("results", [])
            
            if not results:
                break
                
            for work in results:
                authors = [
                    {"name": authorship.get("author", {}).get("display_name", "")}
                    for authorship in work.get("authorships", [])
                ]
                
                paper = {
                    "title": work.get("display_name"),
                    "date": work.get("publication_date"),
                    "url": work.get("doi") or work.get("id"),
                    "authors": authors,
                    "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                    "citation_count": work.get("cited_by_count")
                }
                all_results.append(paper)
                
            if len(results) < current_per_page:
                break
                
            page += 1
            
        return all_results

    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return []

def evaluate_papers_with_ai(papers: List[dict]):
    """
    Evaluates a list of papers based on Zotero history using LLM.
    Updates the 'papers' list in-place with 'score' and 'ai_reason'.
    """
    if not papers:
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found. Skipping AI evaluation.")
        return

    # Fetch Zotero Context (10 recent + 15 random)
    try:
        store = Storage()
        zotero_items = store.get_zotero_sample(recent_count=10, random_count=15)
        zotero_titles = [item['title'] for item in zotero_items]
        zotero_context = "\n".join([f"- {t}" for t in zotero_titles])
    except Exception as e:
        logger.error(f"Failed to fetch Zotero context: {e}")
        zotero_context = "No Zotero history available."

    # Ensure LLM is set up with the requested powerful model
    # User requested 'gemini-3-pro-preview', but it's not available via API yet. Falling back to 2.5 Flash which worked previously.
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    set_global_llm(api_key=api_key, model=model_name, provider="gemini")
    llm = get_llm()

    batch_size = 30
    total_batches = math.ceil(len(papers) / batch_size)
    
    print(f"\nAI Evaluating {len(papers)} papers in {total_batches} batches using {model_name}...")
    print(f"Context: {len(zotero_items)} Zotero papers used for reference.")
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_text = ""
        for idx, p in enumerate(batch):
            # Use relative index 1..30 for the prompt
            batch_text += f"Paper {idx+1}:\nTitle: {p['title']}\nAbstract: {p['abstract'][:500]}...\n\n"
            
        prompt_content = prompts.PAPER_EVALUATION_PROMPT.format(
            zotero_context=zotero_context,
            batch_size=len(batch),
            batch_text=batch_text
        )
        
        prompt = [{"role": "user", "content": prompt_content}]
        
        try:
            response = llm.generate(prompt)
            # Parse response
            lines = response.strip().split('\n')
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        p_idx = int(parts[0].strip()) - 1
                        score = int(parts[1].strip())
                        reason = parts[2].strip() if len(parts) > 2 else ""
                        
                        if 0 <= p_idx < len(batch):
                            batch[p_idx]['score'] = score
                            batch[p_idx]['ai_reason'] = reason
                    except ValueError:
                        continue
        except Exception as e:
            logger.error(f"Error in AI batch evaluation: {e}")

def interactive_mode():
    """
    Interactive CLI for browsing papers.
    """
    print("\n=== Bio/CompBio Journal Fetcher ===\n")
    
    # 1. Select Journal
    print("Available Journals:")
    for i, j_data in enumerate(BIO_JOURNALS, 1):
        cat_info = f" [Categories: {', '.join(j_data['categories'])}]" if j_data['categories'] else ""
        print(f"  {i}. {j_data['name']}{cat_info}")
    
    j_idx = -1
    while True:
        try:
            choice = input(f"\nSelect Journal (1-{len(BIO_JOURNALS)}): ").strip()
            j_idx = int(choice) - 1
            if 0 <= j_idx < len(BIO_JOURNALS):
                break
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
            
    selected_journal = BIO_JOURNALS[j_idx]
    journal_name = selected_journal["name"]
    categories = selected_journal["categories"]

    # Default settings for streamlined mode
    default_days = 30
    user_date = (datetime.now() - timedelta(days=default_days)).strftime("%Y-%m-%d")
    limit = 50
        
    # 2. Fetch Papers
    print(f"\nFetching papers from '{journal_name}' since {user_date}...")
    if categories:
        print(f"Using hardcoded categories: {categories}")
    else:
        print("No categories defined. Fetching all papers.")

    papers = fetch_journal_papers(
        journal_name, 
        concepts=categories, # Use concepts if available
        from_date=user_date, 
        limit=limit
    )
        
    if not papers:
        print(f"\nNo papers found in {journal_name} since {user_date}.")
        return

    # 3. AI Evaluation
    evaluate_papers_with_ai(papers)

    # Sort by Score (Desc) then Date (Desc)
    papers.sort(key=lambda x: (x.get('score', 0), x['date']), reverse=True)
    
    # 4. Output and Save
    print(f"\n--- Found {len(papers)} papers in {journal_name} ---\n")
    
    report_lines = []
    report_lines.append(f"# Papers from {journal_name} since {user_date}\n")

    for i, p in enumerate(papers, 1):
        # Console Output
        score = p.get('score', 0)
        title = p['title']
        if title and len(title) > 80:
            title = title[:77] + "..."
        
        # Colorize based on score
        if score >= 8:
            score_str = f"\033[92m[{score:2}/10]\033[0m" # Green
            title_str = f"\033[1m{title}\033[0m" # Bold
        elif score >= 5:
            score_str = f"\033[93m[{score:2}/10]\033[0m" # Yellow
            title_str = title
        else:
            score_str = f"\033[90m[{score:2}/10]\033[0m" # Grey
            title_str = f"\033[90m{title}\033[0m"
            
        print(f"{i:2}. {score_str} [{p['date']}] {title_str}")
        
        # File Content
        report_lines.append(f"## {i}. {p['title']}")
        report_lines.append(f"**AI Score:** {score}/10")
        report_lines.append(f"**AI Reason:** {p.get('ai_reason', 'N/A')}")
        report_lines.append(f"**Date:** {p['date']}")
        report_lines.append(f"**Authors:** {', '.join([a['name'] for a in p['authors']])}")
        abstract_text = p['abstract'] if p['abstract'] else "[Abstract not available via Open APIs]"
        report_lines.append(f"**Abstract:**\n{abstract_text}\n")
        report_lines.append("-" * 60 + "\n")

    # 5. Save to file
    import os
    os.makedirs("report", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_journal_name = journal_name.replace(" ", "_").lower()
    filename = f"report/fetched_{safe_journal_name}_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nDetailed report with abstracts saved to: {filename}")
    print("You can use this file for further AI analysis.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and analyze papers from scientific journals.")
    parser.add_argument("--journal", type=str, help="Specify a journal name to fetch papers from (e.g., 'Nature Methods').")
    parser.add_argument("--from-date", type=str, 
                        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        help="Specify a start date (YYYY-MM-DD) to fetch papers from. Defaults to 30 days ago.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of papers to fetch. Defaults to 50.")
    parser.add_argument("--query", type=str, help="(Optional) Keyword query to filter papers within the specified journal.")

    args = parser.parse_args()

    if args.journal:
        # Non-interactive mode
        journal_name = args.journal
        from_date = args.from_date
        query = args.query
        limit = args.limit

        # Find the journal data including categories
        selected_journal_data = next((j for j in BIO_JOURNALS if j["name"].lower() == journal_name.lower()), None)

        if not selected_journal_data:
            print(f"Warning: Journal '{journal_name}' not found in the hardcoded list. Fetching without category filters.")
            categories = None
        else:
            categories = selected_journal_data["categories"]

        print(f"\nFetching papers from '{journal_name}' since {from_date}...")
        if categories:
            print(f"Using hardcoded categories: {categories}")
        elif query:
            print(f"Using keyword query: '{query}'")
        else:
            print("No categories or query defined. Fetching all papers.")

        papers = fetch_journal_papers(
            journal_name,
            query=query,
            concepts=categories if not query else None, # Use concepts only if no keyword query is provided
            from_date=from_date,
            limit=limit
        )

        if not papers:
            print(f"\nNo papers found in {journal_name} since {from_date}.")
            sys.exit(0)

        # AI Evaluation
        evaluate_papers_with_ai(papers)

        # Sort by Score (Desc) then Date (Desc)
        papers.sort(key=lambda x: (x.get('score', 0), x['date']), reverse=True)

        # Output and Save Report (reusing logic from interactive_mode)
        print(f"\n--- Found {len(papers)} papers in {journal_name} ---\n")
        
        report_lines = []
        report_lines.append(f"# Papers from {journal_name} since {from_date}\n")

        for i, p in enumerate(papers, 1):
            # Console Output
            score = p.get('score', 0)
            title = p['title']
            if title and len(title) > 80:
                title = title[:77] + "..."
            
            # Colorize based on score
            if score >= 8:
                score_str = f"\033[92m[{score:2}/10]\033[0m" # Green
                title_str = f"\033[1m{title}\033[0m" # Bold
            elif score >= 5:
                score_str = f"\033[93m[{score:2}/10]\033[0m" # Yellow
                title_str = title
            else:
                score_str = f"\033[90m[{score:2}/10]\033[0m" # Grey
                title_str = f"\033[90m{title}\033[0m"
                
            print(f"{i:2}. {score_str} [{p['date']}] {title_str}")
            
            # File Content
            report_lines.append(f"## {i}. {p['title']}")
            report_lines.append(f"**AI Score:** {score}/10")
            report_lines.append(f"**AI Reason:** {p.get('ai_reason', 'N/A')}")
            report_lines.append(f"**Date:** {p['date']}")
            report_lines.append(f"**Authors:** {', '.join([a['name'] for a in p['authors']])}")
            report_lines.append(f"**URL:** {p['url']}")
            abstract_text = p['abstract'] if p['abstract'] else "[Abstract not available via Open APIs]"
            report_lines.append(f"**Abstract:**\n{abstract_text}\n")
            report_lines.append("-" * 60 + "\n")

        os.makedirs("report", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_journal_name = journal_name.replace(" ", "_").lower()
        filename = f"report/fetched_{safe_journal_name}_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            
        print(f"\nDetailed report with abstracts saved to: {filename}")
        print("You can use this file for further AI analysis.")

    else:
        # Interactive mode
        interactive_mode()

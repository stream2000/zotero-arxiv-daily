import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger

# Configure logger to suppress debug messages from imported modules
logger.remove()
logger.add(sys.stderr, level="INFO")

import argparse
import numpy as np
import sqlite3
import pickle
import os
import re
from dotenv import load_dotenv
from zotero_daily.storage import Storage
from zotero_daily.recommender import encode_texts
from zotero_daily.paper import BasePaper
from zotero_daily.report import generate_report
from zotero_daily.llm import set_global_llm, get_llm

load_dotenv()

def construct_search_abstract(user_intent: str) -> str:
    """
    Uses LLM to construct a professional academic abstract based on user intent.
    """
    print("\nConstructing academic abstract from your input...")
    llm = get_llm()
    prompt = f"""
You are a helpful research assistant. The user wants to find research papers based on the following idea or description:

"{user_intent}"

Your task is to write a high-quality, professional academic abstract (in English) that perfectly describes a hypothetical paper about this topic. 
Use specific terminology, relevant methodologies, and maintain a formal tone. 
This abstract will be used as a query for semantic search to find similar existing papers.

Strictly return ONLY the abstract text. Do not include any conversational filler.
"""
    response = llm.generate(
        messages=[
            {"role": "system", "content": "You are an expert academic writer."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.strip()

def search_papers(query: str, limit: int = 10, categories: str = None, db_path: str = "data/metadata.db", storage: Storage = None, disable_citation_check: bool = False):
    """
    Search for papers in the local database using semantic search with optional category filtering.
    """
    if storage is None:
        storage = Storage()
    
    if not storage.candidate_index or storage.candidate_index.ntotal == 0:
        print("Error: No candidate index found. Please run the main data collection process first to populate the database.")
        return []

    target_categories = set()
    if categories:
        target_categories = {c.strip().lower() for c in re.split(r'[;,]', categories) if c.strip()}
        print(f"Filtering by categories: {target_categories}")

    print(f"Encoding query...") 
    # Don't print the full query if it's super long (like the constructed abstract), maybe just first 100 chars
    print(f"Query (preview): '{query[:100]}...'" ) 

    try:
        query_embedding = encode_texts([query])
    except Exception as e:
        print(f"Error encoding query: {e}")
        return []
        
    query_embedding = query_embedding.astype(np.float32)

    search_limit = limit * 10 if target_categories else limit
    D, I = storage.candidate_index.search(query_embedding, search_limit)
    
    found_indices = I[0]
    distances = D[0]
    all_ids = storage.get_all_candidate_ids()
    
    target_ids = []
    id_to_score = {}
    
    for idx, dist in zip(found_indices, distances):
        if idx == -1 or idx >= len(all_ids):
            continue
        pid = all_ids[idx]
        target_ids.append(pid)
        id_to_score[pid] = dist

    if not target_ids:
        return []

    conn = sqlite3.connect(storage.db_path)
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(target_ids))
    query_sql = f"""
        SELECT id, raw_data, tldr, score, citation_count, citation_last_updated, category 
        FROM candidates 
        WHERE id IN ({placeholders})
    """
    c.execute(query_sql, target_ids)
    rows = c.fetchall()
    conn.close()
    
    paper_map = {}
    for row in rows:
        pid, raw_data, tldr, db_score, cit_count, cit_updated, category_str = row
        
        if target_categories:
            if not category_str:
                continue
            paper_cats = {pc.strip().lower() for pc in re.split(r'[;,]', category_str) if pc.strip()}
            if not paper_cats.intersection(target_categories):
                continue

        try:
            paper = pickle.loads(raw_data)
            paper.disable_citation_check = disable_citation_check
            if tldr: paper.set_tldr(tldr)
            if cit_count is not None: paper._citation_count_cache = cit_count
            if cit_updated is not None: paper._citation_last_updated = cit_updated
            paper_map[pid] = paper
        except Exception as e:
            print(f"Error loading paper {pid}: {e}")

    ordered_results = []
    for pid in target_ids:
        if pid in paper_map:
            sim = id_to_score[pid]
            mapped_score = (sim * 13.33) - 4
            paper_map[pid].score = mapped_score
            ordered_results.append((paper_map[pid], sim))
            if len(ordered_results) >= limit:
                break
            
    return [p for p, s in ordered_results]

def main():
    parser = argparse.ArgumentParser(description="Search locally stored papers using semantic search.")
    parser.add_argument("query", type=str, nargs='?', help="The search query text (optional if using --interactive).")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Maximum number of results to return (default: 10).")
    
    default_category = os.getenv("BIORXIV_CATEGORY")
    parser.add_argument("--category", "-c", type=str, default=default_category, 
                        help=f"Filter by category (comma-separated). Default: {default_category}")
    
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the candidate Faiss index before searching.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode: Input a description to auto-generate a search abstract.")
    parser.add_argument("--no-citations", action="store_true", help="Disable citation fetching.")

    args = parser.parse_args()

    # Initialize LLM early if needed
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
         set_global_llm(api_key=gemini_key, provider="gemini", model="gemini-2.5-flash")
    else:
         set_global_llm() # Default

    query_text = args.query

    if args.interactive:
        print("="*60)
        print("Interactive Search Mode")
        print("="*60)
        user_input = input("Please describe what you are looking for (in Chinese or English):\n> ")
        if not user_input.strip():
            print("Empty input. Exiting.")
            return
        
        query_text = construct_search_abstract(user_input)
        print("\n" + "-"*60)
        print("Generated Search Abstract:")
        print("-"*60)
        print(query_text)
        print("-"*60 + "\n")
    
    if not query_text:
        print("Error: No query provided. Please provide a query string or use --interactive.")
        return

    storage = Storage()
    if args.rebuild:
        print("Rebuilding candidate index... This may take a while.")
        storage.rebuild_candidate_index()
    
    if not storage.candidate_index or storage.candidate_index.ntotal == 0:
        print("Error: No candidate index found. Please run the main data collection process first to populate the database.")
        return

    papers = search_papers(query_text, args.limit, args.category, storage=storage, disable_citation_check=args.no_citations)
    
    if not papers:
        print("No results found.")
        return

    print(f"\nFound {len(papers)} relevant papers.")
    
    print("Generating HTML report...")
    html = generate_report(papers, storage)
    
    output_dir = "report"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "search_result.html")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"\nReport saved to: {output_path}")
    print(f"Open it with: open {output_path}")

if __name__ == "__main__":
    main()

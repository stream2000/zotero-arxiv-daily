import sys
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
from storage import Storage
from recommender import encode_texts
from paper import BasePaper
from report import generate_report # Import generate_report

load_dotenv()

def search_papers(query: str, limit: int = 10, categories: str = None, db_path: str = "data/metadata.db", storage: Storage = None):
    """
    Search for papers in the local database using semantic search with optional category filtering.
    """
    if storage is None:
        storage = Storage()
    
    # Check if index exists
    if not storage.candidate_index or storage.candidate_index.ntotal == 0:
        print("Error: No candidate index found. Please run the main data collection process first to populate the database.")
        return []

    # Parse categories if provided
    target_categories = set()
    if categories:
        # Split by comma or semicolon and normalize
        target_categories = {c.strip().lower() for c in re.split(r'[;,]', categories) if c.strip()}
        print(f"Filtering by categories: {target_categories}")

    # Encode the query
    print(f"Encoding query: '{query}'...")
    try:
        query_embedding = encode_texts([query])
    except Exception as e:
        print(f"Error encoding query: {e}")
        return []
        
    query_embedding = query_embedding.astype(np.float32)

    # Search the Faiss index
    # Fetch more candidates if filtering to ensure we have enough results
    search_limit = limit * 10 if target_categories else limit
    
    # D: Distances (scores), I: Indices
    D, I = storage.candidate_index.search(query_embedding, search_limit)
    
    found_indices = I[0]
    distances = D[0]
    
    # Map Faiss indices back to Paper IDs
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

    # Fetch paper details from SQLite
    conn = sqlite3.connect(storage.db_path)
    c = conn.cursor()
    
    # Dynamically build the IN clause
    placeholders = ','.join(['?'] * len(target_ids))
    query_sql = f"""
        SELECT id, raw_data, tldr, score, citation_count, citation_last_updated, category 
        FROM candidates 
        WHERE id IN ({placeholders})
    """
    
    c.execute(query_sql, target_ids)
    rows = c.fetchall()
    conn.close()
    
    # Process rows
    paper_map = {}
    for row in rows:
        pid, raw_data, tldr, db_score, cit_count, cit_updated, category_str = row
        
        # Category Filtering
        if target_categories:
            if not category_str:
                continue # Skip papers without category if filtering is enabled
            
            paper_cats = {pc.strip().lower() for pc in re.split(r'[;,]', category_str) if pc.strip()}
            # Check intersection
            if not paper_cats.intersection(target_categories):
                continue

        try:
            paper = pickle.loads(raw_data)
            # Update with latest metadata from DB
            # Use search similarity score instead of stored score for sorting relevance to query
            # But generate_report usually uses paper.score for stars.
            # We can override paper.score with similarity score for visualization
            
            if tldr:
                paper.set_tldr(tldr)
            
            if cit_count is not None:
                paper._citation_count_cache = cit_count
            if cit_updated is not None:
                paper._citation_last_updated = cit_updated
                
            paper_map[pid] = paper
        except Exception as e:
            print(f"Error loading paper {pid}: {e}")

    # Reconstruct list in order of search results
    ordered_results = []
    for pid in target_ids:
        if pid in paper_map:
            # Overwrite the paper's score with the similarity score so the report shows relevance to the search query
            # Scale it to be comparable to the 0-10 scale used in main.py if needed, 
            # but usually cosine similarity is 0-1.
            # get_stars expects roughly 6-8 range for stars.
            # Cosine sim is usually 0.7-0.9 for relevant papers. 
            # Let's map 0.75 -> 6, 0.9 -> 8
            # y = 13.33x - 4
            sim = id_to_score[pid]
            mapped_score = (sim * 13.33) - 4
            paper_map[pid].score = mapped_score
            
            ordered_results.append((paper_map[pid], sim))
            if len(ordered_results) >= limit:
                break
    
    # Return just the papers for report generation
    return [p for p, s in ordered_results]

def main():
    parser = argparse.ArgumentParser(description="Search locally stored papers using semantic search.")
    parser.add_argument("query", type=str, help="The search query text.")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Maximum number of results to return (default: 10).")
    
    # Default category from environment variable
    default_category = os.getenv("BIORXIV_CATEGORY")
    parser.add_argument("--category", "-c", type=str, default=default_category, 
                        help=f"Filter by category (comma-separated). Default: {default_category}")
    
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the candidate Faiss index before searching.")

    args = parser.parse_args()
    
    storage = Storage()
    if args.rebuild:
        print("Rebuilding candidate index... This may take a while.")
        storage.rebuild_candidate_index()
    
    # Check if index exists or is empty
    if not storage.candidate_index or storage.candidate_index.ntotal == 0:
        print("Error: No candidate index found.")
        print("The database might be empty, or the index has not been built.")
        print("Try running with --rebuild to generate the index from existing database records.")
        return

    papers = search_papers(args.query, args.limit, args.category, storage=storage)
    
    if not papers:
        print("No results found.")
        return

    print(f"\nFound {len(papers)} relevant papers for query: '{args.query}'")
    
    # Generate HTML Report
    print("Generating HTML report...")
    # Initialize global LLM if needed for TLDR generation
    # Since search.py is often run interactively, we might not want to force LLM init unless necessary.
    # But paper.tldr property calls get_llm().
    # Let's assume user has env vars set or we use default local/gemini.
    from llm import set_global_llm
    # Try to load keys from env
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
         set_global_llm(api_key=gemini_key, provider="gemini", model="gemini-2.5-flash")
    else:
         print("Warning: GEMINI_API_KEY not found. TLDR generation might fail or use local fallback.")
         set_global_llm() # Default
    
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
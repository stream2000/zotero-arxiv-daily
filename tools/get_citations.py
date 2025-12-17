import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import argparse
from zotero_daily.citation_client import get_citing_papers

def main():
    parser = argparse.ArgumentParser(description="Get a ranked list of papers citing a specific paper.")
    parser.add_argument("title", help="The title of the paper to analyze.")
    parser.add_argument("--limit", type=int, default=20, help="Number of results to show.")
    parser.add_argument("--from-year", type=int, help="Start year for filtering citations.")
    parser.add_argument("--to-year", type=int, help="End year for filtering citations.")
    
    args = parser.parse_args()
    
    year_msg = ""
    if args.from_year or args.to_year:
        year_msg = f" (Year: {args.from_year or '...'} - {args.to_year or '...'})"

    print(f"Analyzing citations for: '{args.title}'{year_msg}...")
    results = get_citing_papers(args.title, limit=args.limit, start_year=args.from_year, end_year=args.to_year)
    
    if not results:
        print("No citations found or paper not found.")
        return

    print(f"\nTop {len(results)} citing papers (ranked by citation count):")
    print("-" * 80)
    for i, paper in enumerate(results):
        title = paper.get('title', 'Unknown Title')
        count = paper.get('citationCount', 0)
        year = paper.get('year', 'Unknown Year')
        # Handle authors safely
        authors_list = paper.get('authors', [])
        if authors_list:
            authors = ", ".join([a['name'] for a in authors_list[:3]])
            if len(authors_list) > 3:
                authors += " et al."
        else:
            authors = "Unknown Authors"
            
        print(f"{i+1}. [{count} cites] {title} ({year})")
        print(f"    {authors}")
    print("-" * 80)

if __name__ == "__main__":
    main()


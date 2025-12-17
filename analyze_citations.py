import os
import sys
import argparse
from dotenv import load_dotenv
from citation_client import get_citing_papers
from llm import get_llm, set_global_llm
from loguru import logger

# Load environment variables
load_dotenv()

def init_llm():
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        # Context said "Default Gemini model... is now models/gemini-2.5-flash"
        set_global_llm(api_key=gemini_key, provider="gemini", model="models/gemini-3-pro-preview")
    elif openai_key:
        set_global_llm(api_key=openai_key, provider="openai", model="gpt-4o")
    else:
        logger.warning("No API key found in .env (GEMINI_API_KEY or OPENAI_API_KEY). LLM features will be disabled.")

def analyze_citations(title: str, limit: int = 100):
    init_llm()
    logger.info(f"Fetching top {limit} citing papers for: {title}")
    papers = get_citing_papers(title, limit=limit)
    
    if not papers:
        logger.warning("No citing papers found.")
        return

    logger.info(f"Found {len(papers)} papers. Preparing for analysis...")

    # Save raw citation data to Markdown
    safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
    data_md = f"# Citation Data for: {title}\n\n"
    data_md += f"**Total Papers Found:** {len(papers)}\n\n"
    
    for i, p in enumerate(papers):
        authors = ", ".join([a['name'] for a in p.get('authors', [])])
        abstract = p.get('abstract') or "No abstract available."
        
        data_md += f"## {i+1}. {p['title']}\n"
        data_md += f"- **Year:** {p['year']}\n"
        data_md += f"- **Citations:** {p['citationCount']}\n"
        data_md += f"- **Authors:** {authors}\n"
        data_md += f"- **URL:** {p['url']}\n"
        data_md += f"- **Abstract:**\n{abstract}\n\n"
        data_md += "---\n\n"

    data_filename = f"report/citation_data_{safe_title}.md"
    os.makedirs("report", exist_ok=True)
    
    with open(data_filename, "w") as f:
        f.write(data_md)
        
    logger.info(f"Raw citation data saved to: {data_filename}")
    
    # Prepare prompt for LLM
    # If there are too many papers, we might need to truncate or batch.
    # For now, let's try sending title + abstract for the top 50, and just titles for the rest if it fits.
    # Or just top 20-30 abstracts?
    # 100 abstracts is a lot. Let's start with providing full details for top 20 and titles/citations for the rest up to 100.
    
    # Constructing the context
    context_text = f"Target Paper: {title}\n\nTop Citing Papers:\n"
    
    for i, p in enumerate(papers):
        authors = ", ".join([a['name'] for a in p.get('authors', [])[:3]])
        abstract = p.get('abstract', 'No abstract available.')
        # Truncate abstract to save tokens if necessary
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
            
        entry = f"{i+1}. **{p['title']}** ({p['year']}) - Citations: {p['citationCount']}\n   Authors: {authors}\n"
        if i < 30: # Include abstract for top 30
            entry += f"   Abstract: {abstract}\n"
        
        context_text += entry + "\n"

    prompt = [
        {"role": "system", "content": "You are an expert academic researcher. Your task is to analyze the papers citing a specific target paper to understand the impact and research directions it has influenced."},
        {"role": "user", "content": f"""
I will provide a list of the top {len(papers)} papers citing '{title}'.
Please generate a comprehensive Markdown report that includes:

1.  **Overview**: A brief summary of the types of works citing this paper.
2.  **Key Themes**: Categorize the citing papers into major research themes or directions (e.g., "Methodological Improvements", "Applications in Biology", "Benchmarking", etc.).
3.  **Top Influential Works**: Highlight 5-10 specific papers from the list that seem most significant (based on citation count and relevance) and explain *what they do* and *why they are important* in relation to the target paper.
4.  **Trend Analysis**: Briefly mention if you see any trends over time (e.g., shift from applications to improvements).

Here is the data:

{context_text}
"""}
    ]

    logger.info("Sending request to LLM...")
    llm = get_llm()
    # Check if we have an API key configured, otherwise warn
    if not llm.is_api_llm:
         logger.warning("LLM API is not configured. The report will be a placeholder. Please set GEMINI_API_KEY (or OPENAI_API_KEY).")
    
    report_content = llm.generate(prompt)
    
    # Save report
    safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
    filename = f"report/citation_analysis_{safe_title}.md"
    os.makedirs("report", exist_ok=True)
    
    with open(filename, "w") as f:
        f.write(report_content)
        
    logger.info(f"Report saved to: {filename}")
    print(f"Report generated: {filename}")
    print("-" * 40)
    print(report_content[:500] + "...\n(See file for full report)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze citations for a paper.")
    parser.add_argument("title", help="Title of the paper to analyze")
    parser.add_argument("--limit", type=int, default=100, help="Number of citing papers to analyze")
    args = parser.parse_args()
    
    analyze_citations(args.title, args.limit)

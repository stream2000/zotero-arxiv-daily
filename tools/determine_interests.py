import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
from dotenv import load_dotenv
from main import get_zotero_corpus
from zotero_daily.llm import get_llm, set_global_llm
import json
import re

load_dotenv(override=True)

def determine_interests():
    zotero_id = os.environ.get("ZOTERO_ID")
    zotero_key = os.environ.get("ZOTERO_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not zotero_id or not zotero_key:
        print("Error: ZOTERO_ID or ZOTERO_KEY not found in environment.")
        return
    if not gemini_key:
         print("Error: GEMINI_API_KEY not found. Please set it to use LLM analysis.")
         return

    print("Retrieving Zotero library...")
    # get_zotero_corpus from main.py
    corpus = get_zotero_corpus(zotero_id, zotero_key)
    print(f"Retrieved {len(corpus)} papers.")
    
    if not corpus:
        print("No papers found in Zotero library.")
        return

    # Prepare paper data for LLM
    papers_text = ""
    # Limit to top 100 recent papers if corpus is huge to avoid excessive context, 
    # though Gemini can handle it, let's be efficient.
    # Zotero API response might not be sorted by date by default, but let's take first 50.
    sample_corpus = corpus[:50]
    
    for i, p in enumerate(sample_corpus):
        title = p['data'].get('title', 'No Title')
        abstract = p['data'].get('abstractNote', '')
        papers_text += f"{i+1}. Title: {title}\nAbstract: {abstract[:300]}...\n\n"

    # Read categories
    try:
        with open('biorxiv_categories.txt', 'r') as f:
            categories = f.read()
    except FileNotFoundError:
        print("Error: biorxiv_categories.txt not found.")
        return

    # Setup LLM
    # Using Gemini 2.5 Flash as preferred
    set_global_llm(api_key=gemini_key, provider='gemini', model='models/gemini-2.5-flash')
    llm = get_llm()
    
    prompt = f"""
    Here is a list of scientific papers from my personal library:
    
    {papers_text}
    
    Here is a list of available BioRxiv categories:
    {categories}
    
    Based on the papers in my library, please identify which BioRxiv categories are most relevant to my research interests.
    Select the top 3-5 categories that best cover the topics in my library.
    
    Return ONLY a JSON list of strings, exactly matching the category names provided in the list.
    Example: ["Neuroscience", "Bioinformatics"]
    """
    
    print("Analyzing with Gemini...")
    try:
        response = llm.generate(
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Clean response to ensure it's valid JSON
        cleaned_response = response.replace('```json', '').replace('```', '').strip()
        # Find list bracket in case of extra text
        match = re.search(r'\[.*?\]', cleaned_response, re.DOTALL)
        if match:
            categories_list = json.loads(match.group(0))
            print("\nRecommended Categories:")
            for cat in categories_list:
                print(f"- {cat}")
        else:
            print("Failed to parse list from response:")
            print(response)
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    determine_interests()

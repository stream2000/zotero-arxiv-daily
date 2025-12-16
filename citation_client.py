import requests
import time
import random
import urllib.parse
from typing import Optional, List
from functools import wraps
from loguru import logger

def retry_with_backoff(retries=8, initial_backoff=2):
    """
    A decorator for retrying a function with exponential backoff and jitter.
    """
    def r_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _retries, _backoff = retries, initial_backoff
            while _retries > 1:
                try:
                    return f(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        logger.warning(f"Rate limit exceeded. Retrying in {_backoff} seconds...")
                    else:
                        # For other HTTP errors, you might not want to retry as aggressively
                        logger.error(f"HTTP Error {e.response.status_code}: {e}. Retrying...")
                    
                    time.sleep(_backoff + random.uniform(0, 1)) # Add jitter
                    _retries -= 1
                    _backoff = min(64, _backoff * 2)  # Exponential backoff with a cap
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed with {e}, retrying in {_backoff} seconds...")
                    time.sleep(_backoff + random.uniform(0, 1)) # Add jitter
                    _retries -= 1
                    _backoff = min(64, _backoff * 2) # Exponential backoff with a cap
            return f(*args, **kwargs) # Last attempt
        return wrapper
    return r_decorator

def _get_last_name(name: str) -> str:
    """Extracts the last name from a full name, converting to lowercase."""
    return name.split(' ')[-1].lower()

@retry_with_backoff()
def _fetch_data(url: str):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstructs the abstract from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    
    # The inverted index maps words to lists of positions.
    # We need to create a list of words with the length of the max position + 1.
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

def get_citation_count(title: str, authors: List[str]) -> Optional[int]:
    """
    Fetches the citation count for a paper from Semantic Scholar.
    It fetches multiple results and compares them by title and author names for better accuracy.
    Includes retry logic with exponential backoff.

    Args:
        title: The title of the paper to search for.
        authors: A list of author names for the paper.

    Returns:
        The citation count, or None if the paper could not be found or matched.
    """
    try:
        encoded_query = requests.utils.quote(title)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,citationCount,authors&limit=5"
        data = _fetch_data(url)

        papers = data.get("data")
        if not papers:
            return None

        if len(papers) == 1:
            return papers[0].get("citationCount")

        authors_str = [author.name if hasattr(author, 'name') else str(author) for author in authors]
        input_author_last_names = {_get_last_name(name) for name in authors_str}

        best_match = None
        max_score = -1

        for paper in papers:
            api_title = paper.get('title', '').lower()
            query_title = title.lower()
            title_score = 0.0
            if api_title == query_title:
                title_score = 1.0
            elif query_title in api_title or api_title in query_title:
                title_score = 0.5

            api_authors = paper.get('authors', [])
            author_score = 0.0
            if api_authors:
                api_author_last_names = {_get_last_name(author['name']) for author in api_authors}
                matching_authors = len(input_author_last_names.intersection(api_author_last_names))
                if input_author_last_names:
                    author_score = matching_authors / len(input_author_last_names)
            
            total_score = 0.7 * title_score + 0.3 * author_score
            if total_score > max_score:
                max_score = total_score
                best_match = paper

        if best_match and max_score > 0.5:
            return best_match.get("citationCount")

        if papers:
            return papers[0].get("citationCount")

        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching citation count for '{title}' after retries: {e}")
        return None

def get_citing_papers(title: str, limit: int = 100, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[dict]:
    """
    Fetches the papers that cite the given paper title using OpenAlex API.
    Returns a list of citing papers, ranked by their own citation count.
    
    Args:
        title: The title of the paper to search for.
        limit: The maximum number of citing papers to return.
        start_year: The start year to filter citations (inclusive).
        end_year: The end year to filter citations (inclusive).
        
    Returns:
        A list of dictionaries containing details of citing papers.
    """
    try:
        # 1. Search for the work ID in OpenAlex
        # We use 'title.search' filter for better precision
        encoded_title = urllib.parse.quote(title)
        search_url = f"https://api.openalex.org/works?filter=title.search:{encoded_title}&sort=cited_by_count:desc&per-page=1"
        data = _fetch_data(search_url)
        
        if not data.get("results"):
            logger.warning(f"Paper not found in OpenAlex: {title}")
            return []
            
        work = data["results"][0]
        work_id = work["id"].split('/')[-1] # Extract ID from URL (e.g., https://openalex.org/W123 -> W123)
        work_title = work["display_name"]
        logger.info(f"Found paper (OpenAlex): {work_title} (ID: {work_id})")
        
        # 2. Fetch citing papers
        # Construct filter string
        filters = [f"cites:{work_id}"]
        if start_year and end_year:
            filters.append(f"from_publication_date:{start_year}-01-01")
            filters.append(f"to_publication_date:{end_year}-12-31")
        elif start_year:
            filters.append(f"from_publication_date:{start_year}-01-01")
        elif end_year:
            filters.append(f"to_publication_date:{end_year}-12-31")
            
        filter_str = ",".join(filters)
        
        # We want highly cited papers, so sort by cited_by_count:desc
        citations_url = f"https://api.openalex.org/works?filter={filter_str}&sort=cited_by_count:desc&per-page={limit}"
        
        citations_data = _fetch_data(citations_url)
        
        results = []
        for res in citations_data.get("results", []):
            # Map OpenAlex format to our internal format
            authors = []
            for authorship in res.get("authorships", []):
                author_obj = authorship.get("author", {})
                if author_obj.get("display_name"):
                    authors.append({"name": author_obj["display_name"]})
            
            results.append({
                "title": res.get("display_name"),
                "citationCount": res.get("cited_by_count"),
                "year": res.get("publication_year"),
                "authors": authors,
                "url": res.get("id"),
                "abstract": reconstruct_abstract(res.get("abstract_inverted_index"))
            })
                
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching citing papers for '{title}': {e}")
        return []
import requests
import time
import random
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
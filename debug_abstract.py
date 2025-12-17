from journal_fetcher import fetch_journal_papers

papers = fetch_journal_papers("Nature Methods", limit=5)
for p in papers:
    print(f"Title: {p['title']}")
    print(f"URL: {p['url']}")
    print(f"Abstract length: {len(p['abstract'])}")
    print("-" * 20)
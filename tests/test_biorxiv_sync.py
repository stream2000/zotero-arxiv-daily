import sys
import os
# Add the project root to sys.path to allow importing modules directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import sqlite3
from datetime import datetime, timedelta
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Assuming these are importable from the project root
from storage import Storage
from paper import BioRxivPaper
from main import sync_biorxiv_papers # Import the function directly

# Fixture for a temporary database
@pytest.fixture
def temp_db_path(tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    yield str(db_dir)
    shutil.rmtree(str(db_dir))

@pytest.fixture
def storage_instance(temp_db_path):
    # Ensure storage is initialized with a fresh temp directory
    return Storage(data_dir=temp_db_path)

# Helper to create a dummy BioRxivPaper for testing
def create_dummy_biorxiv_paper(doi, date_str, category):
    return BioRxivPaper({
        "doi": doi,
        "title": f"Test Paper {doi}",
        "abstract": f"Abstract for {doi}",
        "date": date_str,
        "category": category,
        "authors": "Author One; Author Two"
    })


def test_get_candidates_by_date_range_with_dates(storage_instance):
    db = storage_instance
    
    # Add some papers with dates
    paper1 = create_dummy_biorxiv_paper("10.1101/2025.12.01.000001", "2025-12-01", "bioinformatics")
    paper2 = create_dummy_biorxiv_paper("10.1101/2025.12.02.000002", "2025-12-02", "genomics")
    paper3 = create_dummy_biorxiv_paper("10.1101/2025.12.03.000003", "2025-12-03", "neuroscience")
    paper4 = create_dummy_biorxiv_paper("10.1101/2025.12.04.000004", "2025-12-04", "molecular biology")

    # Mock encode_texts to return a dummy numpy array for adding candidates
    with patch('recommender.encode_texts', return_value=np.random.rand(1, 768).astype(np.float32)): 
        db.add_candidates([paper1, paper2, paper3, paper4], None)

    # Test within a range
    papers = db.get_candidates_by_date_range("2025-12-02", "2025-12-03")
    assert len(papers) == 2
    assert papers[0].arxiv_id == "10.1101/2025.12.02.000002"
    assert papers[1].arxiv_id == "10.1101/2025.12.03.000003"

    # Test edge cases
    papers = db.get_candidates_by_date_range("2025-12-01", "2025-12-01")
    assert len(papers) == 1
    assert papers[0].arxiv_id == "10.1101/2025.12.01.000001"

    papers = db.get_candidates_by_date_range("2025-12-05", "2025-12-06")
    assert len(papers) == 0


@patch('main.BioRxivApi')
@patch('recommender.encode_texts', return_value=np.random.rand(1, 768).astype(np.float32))
def test_sync_biorxiv_papers_new_papers(mock_encode_texts, mock_biorxiv_api, storage_instance):
    db = storage_instance
    mock_api_instance = mock_biorxiv_api.return_value
    
    # Simulate BioRxiv API returning papers for a specific day
    mock_api_instance.get_papers.return_value = [
        {"doi": "10.1101/2025.12.01.000001", "title": "Title 1", "abstract": "Abstract 1", "date": "2025-12-01", "category": "bioinformatics", "authors": "A"},
        {"doi": "10.1101/2025.12.01.000002", "title": "Title 2", "abstract": "Abstract 2", "date": "2025-12-01", "category": "genomics", "authors": "B"},
    ]

    # Simulate today is 2025-12-02, so we fetch for 2025-12-01 (days=1)
    with patch('main.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 12, 2)
        mock_datetime.date.today.return_value = datetime(2025, 12, 2).date() # For today_str calculation
        mock_datetime.timedelta = timedelta # Ensure timedelta still works

        sync_biorxiv_papers(db, days=1)

    # Verify papers were added and date marked complete
    assert len(db.get_all_candidates()) == 2
    assert "2025-12-01" in db.get_biorxiv_completed_dates()
    mock_encode_texts.assert_called_once() # Should be called for new papers


@patch('main.BioRxivApi')
@patch('recommender.encode_texts', return_value=np.random.rand(1, 768).astype(np.float32))
def test_sync_biorxiv_papers_existing_papers(mock_encode_texts, mock_biorxiv_api, storage_instance):
    db = storage_instance
    mock_api_instance = mock_biorxiv_api.return_value

    # Add a paper directly to DB first
    existing_paper = create_dummy_biorxiv_paper("10.1101/2025.12.01.000001", "2025-12-01", "bioinformatics")
    with patch('recommender.encode_texts', return_value=np.random.rand(1, 768).astype(np.float32)):
        db.add_candidates([existing_paper], None)
    db.mark_biorxiv_date_completed("2025-12-01") # Mark as completed so sync doesn't fetch

    # Simulate BioRxiv API returning the *same* paper, and a new one
    mock_api_instance.get_papers.return_value = [
        {"doi": "10.1101/2025.12.01.000001", "title": "Title 1", "abstract": "Abstract 1", "date": "2025-12-01", "category": "bioinformatics", "authors": "A"},
        {"doi": "10.1101/2025.12.01.000003", "title": "Title 3", "abstract": "Abstract 3", "date": "2025-12-01", "category": "genomics", "authors": "C"},
    ]

    # Simulate today is 2025-12-02, try to sync for 2025-12-01 (should skip due to completed date)
    with patch('main.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 12, 2)
        mock_datetime.date.today.return_value = datetime(2025, 12, 2).date()
        mock_datetime.timedelta = timedelta

        sync_biorxiv_papers(db, days=1)

    # No new papers should have been added, and encode_texts not called again
    assert len(db.get_all_candidates()) == 1 # Only the initially added paper
    assert "2025-12-01" in db.get_biorxiv_completed_dates() # Still marked completed
    mock_encode_texts.assert_not_called() # Should not be called because date is already completed

@patch('main.BioRxivApi')
@patch('recommender.encode_texts', return_value=np.random.rand(1, 768).astype(np.float32))
def test_sync_biorxiv_papers_fetch_multiple_days(mock_encode_texts, mock_biorxiv_api, storage_instance):
    db = storage_instance
    mock_api_instance = mock_biorxiv_api.return_value

    # Mock get_papers to return different sets for different days
    def mock_get_papers_side_effect(start_date, end_date):
        if start_date == "2025-11-30":
            return [{"doi": "10.1101/2025.11.30.000001", "title": "T1", "abstract": "A1", "date": "2025-11-30", "category": "neuroscience", "authors": "A"}]
        elif start_date == "2025-12-01":
            return [{"doi": "10.1101/2025.12.01.000002", "title": "T2", "abstract": "A2", "date": "2025-12-01", "category": "cell biology", "authors": "B"}]
        return []

    mock_api_instance.get_papers.side_effect = mock_get_papers_side_effect

    # Simulate today is 2025-12-02, fetch for 3 days (11-30, 12-01)
    with patch('main.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 12, 2)
        mock_datetime.date.today.return_value = datetime(2025, 12, 2).date()
        mock_datetime.timedelta = timedelta

        sync_biorxiv_papers(db, days=3) # Should fetch 2025-11-30, 2025-12-01

    # Verify papers for both days were added and marked complete
    all_candidates = db.get_all_candidates()
    assert len(all_candidates) == 2
    assert any(p.arxiv_id == "10.1101/2025.11.30.000001" for p in all_candidates)
    assert any(p.arxiv_id == "10.1101/2025.12.01.000002" for p in all_candidates)
    assert "2025-11-30" in db.get_biorxiv_completed_dates()
    assert "2025-12-01" in db.get_biorxiv_completed_dates()
    assert "2025-11-29" in db.get_biorxiv_completed_dates() # The skipped day should also be marked as completed
    assert mock_encode_texts.call_count == 2 # Called once for each day's new papers


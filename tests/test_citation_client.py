import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock
import requests
from zotero_daily.citation_client import get_citation_count, _get_last_name


class TestCitationClient(unittest.TestCase):

    def test_get_last_name(self):
        self.assertEqual(_get_last_name("John Doe"), "doe")
        self.assertEqual(_get_last_name("Jane"), "jane")
        self.assertEqual(_get_last_name("Jean-Claude Van Damme"), "damme")

    @patch('zotero_daily.citation_client._fetch_data')
    def test_get_citation_count_success(self, mock_fetch):
        # Mock the API response
        mock_fetch.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "citationCount": 123,
                    "authors": [{"name": "John Doe"}]
                }
            ]
        }

        # Call the function
        count = get_citation_count("Test Paper", ["John Doe"])

        # Assert the result
        self.assertEqual(count, 123)
        mock_fetch.assert_called_once()

    @patch('zotero_daily.citation_client._fetch_data')
    def test_get_citation_count_no_paper_found(self, mock_fetch):
        # Mock the API response for no paper found
        mock_fetch.return_value = {"data": None}

        # Call the function
        count = get_citation_count("Unknown Paper", ["Jane Doe"])

        # Assert the result
        self.assertIsNone(count)
        mock_fetch.assert_called_once()

    @patch('zotero_daily.citation_client._fetch_data')
    def test_get_citation_count_api_error(self, mock_fetch):
        # Mock an API error
        mock_fetch.side_effect = requests.exceptions.RequestException("API Error")

        # Call the function (the retry decorator will handle the exception)
        count = get_citation_count("Any Paper", ["Some Author"])

        # Assert the result
        self.assertIsNone(count)

    @patch('zotero_daily.citation_client._fetch_data')
    def test_author_matching_logic(self, mock_fetch):
        # Mock the API response with multiple results
        mock_fetch.return_value = {
            "data": [
                {
                    "title": "A Review of Test Papers", # Partial title match
                    "citationCount": 50,
                    "authors": [{"name": "Jane Smith"}, {"name": "Peter Pan"}]
                },
                {
                    "title": "Test Paper", # Exact title match
                    "citationCount": 150,
                    "authors": [{"name": "John Doe"}, {"name": "Another Guy"}]
                },
                {
                    "title": "Another Paper",
                    "citationCount": 20,
                    "authors": [{"name": "Someone Else"}]
                }
            ]
        }

        # Call the function with authors that should match the second result
        count = get_citation_count("Test Paper", ["J. Doe", "A. Guy"])

        # Assert the result
        self.assertEqual(count, 150)
        mock_fetch.assert_called_once()

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import Mock, patch
from abc import ABC, abstractproperty
from typing import List, Optional
import numpy as np

# Import the actual classes to be tested
from paper import BasePaper, ArxivPaper, BioRxivPaper, SimpleAuthor
from recommender import rerank_paper
from construct_email import render_email

# --- Mocks for testing ---

class MockAuthor:
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        return self._name

class MockPaper(BasePaper):
    def __init__(self, title, summary, authors, arxiv_id, pdf_url, code_url, tldr, affiliations, score=None):
        self._title = title
        self._summary = summary
        self._authors = [MockAuthor(name) for name in authors] # Ensure authors have .name
        self._arxiv_id = arxiv_id
        self._pdf_url = pdf_url
        self._code_url = code_url
        self._tldr = tldr
        self._affiliations = affiliations
        self._score = score

    @property
    def title(self) -> str:
        return self._title

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def authors(self) -> List[MockAuthor]:
        return self._authors
    
    # score property is handled by BasePaper
    
    @property
    def arxiv_id(self) -> str:
        return self._arxiv_id

    @property
    def pdf_url(self) -> str:
        return self._pdf_url

    @property
    def code_url(self) -> Optional[str]:
        return self._code_url
    
    @property
    def tldr(self) -> str:
        return self._tldr

    @property
    def affiliations(self) -> Optional[List[str]]:
        return self._affiliations

# --- Test Cases ---

class TestPaperAbstraction(unittest.TestCase):

    def setUp(self):
        # Create some mock papers
        self.mock_paper1 = MockPaper(
            title="Test Paper 1",
            summary="This is a summary for test paper 1.",
            authors=["Author A", "Author B"],
            arxiv_id="2301.00001",
            pdf_url="http://example.com/pdf1",
            code_url="http://example.com/code1",
            tldr="TLDR for paper 1.",
            affiliations=["Affiliation X"]
        )
        self.mock_paper2 = MockPaper(
            title="Test Paper 2",
            summary="This is a summary for test paper 2, slightly different.",
            authors=["Author C"],
            arxiv_id="2301.00002",
            pdf_url="http://example.com/pdf2",
            code_url=None,
            tldr="TLDR for paper 2.",
            affiliations=["Affiliation Y", "Affiliation Z"]
        )
        self.mock_paper3 = MockPaper(
            title="Test Paper 3",
            summary="Another summary for paper 3.",
            authors=["Author D", "Author E", "Author F"],
            arxiv_id="2301.00003",
            pdf_url="http://example.com/pdf3",
            code_url="http://example.com/code3",
            tldr="TLDR for paper 3.",
            affiliations=None
        )
        self.mock_papers = [self.mock_paper1, self.mock_paper2, self.mock_paper3]

        # Mock Zotero corpus for rerank_paper
        self.mock_corpus = [
            {'data': {'abstractNote': 'Some abstract for a Zotero paper related to test paper 1.', 'dateAdded': '2023-01-01T12:00:00Z'}},
            {'data': {'abstractNote': 'Another Zotero abstract, less related to test paper 1.', 'dateAdded': '2023-01-02T12:00:00Z'}}
        ]

    def test_arxiv_paper_inherits_base_paper(self):
        self.assertTrue(issubclass(ArxivPaper, BasePaper))
        
        mock_arxiv_result = Mock()
        mock_arxiv_result.title = "Mock Arxiv Title"
        mock_arxiv_result.summary = "Mock Arxiv Summary"
        mock_arxiv_result.authors = [MockAuthor("Mock Arxiv Author")]
        mock_arxiv_result.get_short_id.return_value = "2301.12345v1"
        mock_arxiv_result.pdf_url = "http://arxiv.org/pdf/2301.12345.pdf"
        mock_arxiv_result.links = [Mock(href="http://arxiv.org/abs/2301.12345")]
        
        with patch('paper.requests.Session'), \
             patch('paper.TemporaryDirectory'), \
             patch('paper.arxiv.Result.download_source'), \
             patch('paper.tarfile.open'), \
             patch('paper.get_llm'), \
             patch('paper.tiktoken.encoding_for_model'):
            arxiv_paper_instance = ArxivPaper(mock_arxiv_result)
            self.assertIsInstance(arxiv_paper_instance, ArxivPaper)
            self.assertIsInstance(arxiv_paper_instance, BasePaper)
            self.assertEqual(arxiv_paper_instance.title, "Mock Arxiv Title")
            self.assertEqual(arxiv_paper_instance.arxiv_id, "2301.12345")

    def test_biorxiv_paper_inherits_base_paper(self):
        self.assertTrue(issubclass(BioRxivPaper, BasePaper))

        mock_data = {
            "title": "BioRxiv Title",
            "abstract": "BioRxiv Abstract",
            "authors": "Author 1; Author 2",
            "doi": "10.1101/2023.01.01.123456"
        }

        with patch('paper.get_llm'), \
             patch('paper.tiktoken.encoding_for_model'):
            biorxiv_paper = BioRxivPaper(mock_data)
            
            self.assertIsInstance(biorxiv_paper, BioRxivPaper)
            self.assertIsInstance(biorxiv_paper, BasePaper)
            
            self.assertEqual(biorxiv_paper.title, "BioRxiv Title")
            self.assertEqual(biorxiv_paper.summary, "BioRxiv Abstract")
            self.assertEqual(biorxiv_paper.arxiv_id, "10.1101/2023.01.01.123456")
            self.assertEqual(biorxiv_paper.pdf_url, "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf")
            self.assertIsNone(biorxiv_paper.code_url)
            self.assertIsNone(biorxiv_paper.affiliations)
            
            # Check authors parsing
            authors = biorxiv_paper.authors
            self.assertEqual(len(authors), 2)
            self.assertEqual(authors[0].name, "Author 1")
            self.assertEqual(authors[1].name, "Author 2")
            self.assertIsInstance(authors[0], SimpleAuthor)

    @patch('recommender.SentenceTransformer')
    def test_rerank_paper(self, MockSentenceTransformer):
        # Mock SentenceTransformer and its methods
        mock_encoder = Mock()
        MockSentenceTransformer.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x, **kwargs: np.array([[0.1, 0.1]] * len(x), dtype=np.float32) # Return NumPy arrays
        mock_encoder.similarity.return_value = [[0.5] * len(self.mock_corpus)] * len(self.mock_papers)

        reranked_papers = rerank_paper(self.mock_papers, self.mock_corpus)

        self.assertIsInstance(reranked_papers, list)
        self.assertGreater(len(reranked_papers), 0)
        self.assertIsInstance(reranked_papers[0], BasePaper)

    @patch('construct_email.get_block_html')
    @patch('construct_email.get_empty_html')
    @patch('construct_email.tqdm') 
    def test_render_email(self, mock_tqdm, mock_get_empty_html, mock_get_block_html): # Removed mock_sleep
        # Configure mock_tqdm to return the iterable passed to it
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Assign scores
        self.mock_paper1.score = 9.5
        self.mock_paper2.score = 7.0
        self.mock_paper3.score = 5.0

        mock_get_block_html.side_effect = lambda title, authors, rate, arxiv_id, abstract, pdf_url, code, affiliations: f"<div>{title}</div>"
        mock_get_empty_html.return_value = "<div>No Papers</div>"
        
        # Test with papers
        html_output = render_email(self.mock_papers)
        self.assertIn("Test Paper 1", html_output)
        
        # Verify calls to get_block_html
        args, kwargs = mock_get_block_html.call_args_list[0]
        self.assertIn('<span class="full-star">⭐</span>', args[2]) # stars are in the 3rd argument (rate)


if __name__ == '__main__':
    unittest.main()

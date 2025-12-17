import unittest
from unittest.mock import Mock, patch
from abc import ABC, abstractproperty
from typing import List, Optional
import numpy as np

# Import the actual classes to be tested
from zotero_daily.paper import BasePaper, ArxivPaper, BioRxivPaper, SimpleAuthor
from zotero_daily.recommender import rerank_paper
from zotero_daily.report import generate_report

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
    def source(self) -> str:
        # Mock source for testing purposes
        if "arxiv" in self.arxiv_id:
            return "arxiv"
        return "biorxiv"



    
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

# Removed TestPaperAbstraction class
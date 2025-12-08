import unittest
import os
import shutil
import tempfile
import numpy as np
import sqlite3
import faiss
import pickle
from loguru import logger
from unittest.mock import patch, MagicMock

from storage import Storage
from paper import BioRxivPaper, ArxivPaper # Assuming these are used for candidates

class MockPaper:
    def __init__(self, arxiv_id, title="", summary="", category='', score=0.0, source='biorxiv'):
        self.arxiv_id = arxiv_id
        self.title = title
        self.summary = summary
        self._paper = {'category': category} # For BioRxivPaper-like behavior
        self.score = score # for direct access in test
        self.source = source # Added source attribute
        self.tldr_cache = None

    def set_tldr(self, tldr: str):
        self.tldr_cache = tldr
        self.tldr_cache = None

    def set_tldr(self, tldr: str):
        self.tldr_cache = tldr
class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db = Storage(data_dir=self.test_dir)
        self.dim = 768 # Assuming embedding dimension

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "metadata.db")))
        self.assertIsNone(self.db.zotero_index) # Should be None if not created yet
        self.assertIsNone(self.db.candidate_index) # Should be None if not created yet

    def test_update_zotero(self):
        mock_zotero_papers = [
            {'key': 'z1', 'data': {'title': 'Zotero Paper 1', 'abstractNote': 'Abstract 1', 'dateAdded': '2023-01-01T00:00:00Z'}},
            {'key': 'z2', 'data': {'title': 'Zotero Paper 2', 'abstractNote': 'Abstract 2', 'dateAdded': '2023-01-02T00:00:00Z'}}
        ]
        embeddings = np.random.rand(2, self.dim).astype(np.float32)
        
        self.db.update_zotero(mock_zotero_papers, embeddings)
        
        self.assertIsNotNone(self.db.zotero_index)
        self.assertEqual(self.db.zotero_index.ntotal, 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "zotero.index")))
        
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM zotero")
        self.assertEqual(c.fetchone()[0], 2)
        c.execute("SELECT title FROM zotero WHERE id = 'z1'")
        self.assertEqual(c.fetchone()[0], 'Zotero Paper 1')
        conn.close()

    @patch('recommender.encode_texts', side_effect=lambda texts: np.random.rand(len(texts), 768).astype(np.float32))
    def test_add_candidates(self, mock_encode_texts):
        mock_candidates = [
            MockPaper('c1', 'Candidate 1', 'Summary 1', 'bioinformatics'),
            MockPaper('c2', 'Candidate 2', 'Summary 2', 'genomics')
        ]
        self.db.add_candidates(mock_candidates, None) # Embeddings will be mocked internally
        
        self.assertIsNotNone(self.db.candidate_index)
        self.assertEqual(self.db.candidate_index.ntotal, 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "candidates.index")))
        
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM candidates")
        self.assertEqual(c.fetchone()[0], 2)
        c.execute("SELECT title, category FROM candidates WHERE id = 'c1'")
        row = c.fetchone()
        self.assertEqual(row[0], 'Candidate 1')
        self.assertEqual(row[1], 'bioinformatics')
        conn.close()
        
        self.db.add_candidates([], None) # Embeddings will be mocked internally
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM candidates")
        self.assertEqual(c.fetchone()[0], 2) # Still 2
        conn.close()

    @patch('recommender.encode_texts', side_effect=lambda texts: np.random.rand(len(texts), 768).astype(np.float32))
    def test_get_existing_candidate_ids(self, mock_encode_texts):
        mock_candidates = [MockPaper('e1'), MockPaper('e2')]
        self.db.add_candidates(mock_candidates, None)
        
        existing_ids = self.db.get_existing_candidate_ids()
        self.assertEqual(existing_ids, {'e1', 'e2'})

    @patch('recommender.encode_texts', side_effect=lambda texts: np.random.rand(len(texts), 768).astype(np.float32))
    def test_update_scores(self, mock_encode_texts):
        mock_candidates = [MockPaper('s1'), MockPaper('s2')]
        self.db.add_candidates(mock_candidates, None) # Add before updating scores

        scores_dict = {'s1': 0.85, 's2': 0.92}
        self.db.update_scores(scores_dict)
        
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT score FROM candidates WHERE id = 's1'")
        self.assertEqual(c.fetchone()[0], 0.85)
        conn.close()

    @patch('recommender.encode_texts', side_effect=lambda texts: np.random.rand(len(texts), 768).astype(np.float32))
    def test_get_all_candidate_ids(self, mock_encode_texts):
        mock_candidates = [MockPaper('id3'), MockPaper('id1'), MockPaper('id2')]
        self.db.add_candidates(mock_candidates, None)
        
        expected_ids = ['id1', 'id2', 'id3'] # Faiss IDs are assigned based on sorted arxiv_id by _rebuild_candidate_faiss_index (ORDER BY id ASC)
        self.assertEqual(self.db.get_all_candidate_ids(), expected_ids)

    @patch('recommender.encode_texts', side_effect=lambda texts: np.random.rand(len(texts), 768).astype(np.float32))
    def test_get_top_candidates(self, mock_encode_texts):
        # Add candidates with scores and categories
        papers_to_add = [
            MockPaper(arxiv_id='p1', score=0.9, category='genomics'),
            MockPaper(arxiv_id='p2', score=0.7, category='bioinformatics'),
            MockPaper(arxiv_id='p3', score=0.95, category='genomics'),
            MockPaper(arxiv_id='p4', score=0.6, category='systems biology')
        ]
        self.db.add_candidates(papers_to_add, None)
        
        # Update scores in DB (important for sorting)
        scores_dict = {'p1': 0.9, 'p2': 0.7, 'p3': 0.95, 'p4': 0.6}
        self.db.update_scores(scores_dict)

        # Test without filter (limit 2)
        top_papers = self.db.get_top_candidates(limit=2)
        self.assertEqual(len(top_papers), 2)
        self.assertEqual(top_papers[0].arxiv_id, 'p3') # Highest score
        self.assertEqual(top_papers[1].arxiv_id, 'p1')
        
        # Test with limit 3
        top_papers_3 = self.db.get_top_candidates(limit=3)
        self.assertEqual(len(top_papers_3), 3)
        self.assertEqual(top_papers_3[0].arxiv_id, 'p3')
        self.assertEqual(top_papers_3[1].arxiv_id, 'p1')
        self.assertEqual(top_papers_3[2].arxiv_id, 'p2')

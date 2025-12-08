import unittest
import os
import shutil
import tempfile
import numpy as np
import sqlite3
import faiss
import pickle
from loguru import logger

from storage import Storage
from paper import BioRxivPaper, ArxivPaper # Assuming these are used for candidates

class MockPaper:
    def __init__(self, arxiv_id, title="", summary="", category='', score=0.0):
        self.arxiv_id = arxiv_id
        self.title = title
        self.summary = summary
        self._paper = {'category': category} # For BioRxivPaper-like behavior
        self.score = score # for direct access in test

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db = Storage(data_dir=self.test_dir)
        self.dim = 384 # Assuming embedding dimension

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

    def test_add_candidates(self):
        mock_candidates = [
            MockPaper('c1', 'Candidate 1', 'Summary 1', 'bioinformatics'),
            MockPaper('c2', 'Candidate 2', 'Summary 2', 'genomics')
        ]
        embeddings = np.random.rand(2, self.dim).astype(np.float32)
        
        self.db.add_candidates(mock_candidates, embeddings)
        
        self.assertIsNotNone(self.db.candidate_index)
        self.assertEqual(self.db.candidate_index.ntotal, 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "candidates.index")))
        
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM candidates")
        self.assertEqual(c.fetchone()[0], 2)
        c.execute("SELECT title, category, faiss_id FROM candidates WHERE id = 'c1'")
        row = c.fetchone()
        self.assertEqual(row[0], 'Candidate 1')
        self.assertEqual(row[1], 'bioinformatics')
        self.assertEqual(row[2], 0) # First added gets faiss_id 0
        conn.close()
        
        self.db.add_candidates([], np.array([]).astype(np.float32))
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM candidates")
        self.assertEqual(c.fetchone()[0], 2) # Still 2
        conn.close()

    def test_get_existing_candidate_ids(self):
        mock_candidates = [MockPaper('e1'), MockPaper('e2')]
        embeddings = np.random.rand(2, self.dim).astype(np.float32)
        self.db.add_candidates(mock_candidates, embeddings)
        
        existing_ids = self.db.get_existing_candidate_ids()
        self.assertEqual(existing_ids, {'e1', 'e2'})

    def test_update_scores(self):
        mock_candidates = [MockPaper('s1'), MockPaper('s2')]
        embeddings = np.random.rand(2, self.dim).astype(np.float32)
        self.db.add_candidates(mock_candidates, embeddings) # Add before updating scores

        scores_dict = {'s1': 0.85, 's2': 0.92}
        self.db.update_scores(scores_dict)
        
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT score FROM candidates WHERE id = 's1'")
        self.assertEqual(c.fetchone()[0], 0.85)
        conn.close()

    def test_get_all_candidate_ids(self):
        mock_candidates = [MockPaper('id3'), MockPaper('id1'), MockPaper('id2')]
        embeddings = np.random.rand(3, self.dim).astype(np.float32)
        self.db.add_candidates(mock_candidates, embeddings)
        
        expected_ids = ['id3', 'id1', 'id2'] # Faiss IDs are assigned in order of addition (0, 1, 2).
        self.assertEqual(self.db.get_all_candidate_ids(), expected_ids)

    def test_get_top_candidates(self):
        # Add candidates with scores and categories
        papers_to_add = [
            MockPaper(arxiv_id='p1', score=0.9, category='genomics'),
            MockPaper(arxiv_id='p2', score=0.7, category='bioinformatics'),
            MockPaper(arxiv_id='p3', score=0.95, category='genomics'),
            MockPaper(arxiv_id='p4', score=0.6, category='systems biology')
        ]
        embeddings = np.random.rand(len(papers_to_add), self.dim).astype(np.float32)
        self.db.add_candidates(papers_to_add, embeddings)
        
        # Debug: Check actual categories in DB
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT id, category, score FROM candidates ORDER BY id")
        logger.debug(f"DB Candidates before get_top_candidates: {c.fetchall()}")
        conn.close()
        
        # Update scores in DB (important for sorting)
        scores_dict = {'p1': 0.9, 'p2': 0.7, 'p3': 0.95, 'p4': 0.6}
        self.db.update_scores(scores_dict)

        # Test without filter
        top_papers = self.db.get_top_candidates(limit=2)
        self.assertEqual(len(top_papers), 2)
        self.assertEqual(top_papers[0].arxiv_id, 'p3') # Highest score
        self.assertEqual(top_papers[1].arxiv_id, 'p1')
        
        # Test with filter
        filtered_papers = self.db.get_top_candidates(limit=2, filter_categories=['genomics'])
        self.assertEqual(len(filtered_papers), 2)
        self.assertEqual(filtered_papers[0].arxiv_id, 'p3')
        self.assertEqual(filtered_papers[1].arxiv_id, 'p1')
        
        filtered_papers_single = self.db.get_top_candidates(limit=1, filter_categories=['systems biology'])
        self.assertEqual(len(filtered_papers_single), 1)
        self.assertEqual(filtered_papers_single[0].arxiv_id, 'p4')

        # Test case where no category matches
        no_match_papers = self.db.get_top_candidates(limit=5, filter_categories=['nonexistent'])
        self.assertEqual(len(no_match_papers), 0)

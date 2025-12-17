import unittest
import numpy as np
from zotero_daily.recommender import calculate_scores, encode_texts
from unittest.mock import patch, Mock

class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.dim = 384

    def test_calculate_scores_basic(self):
        # Mock embeddings (normalized)
        cand_emb = np.array([
            [0.1, 0.9], # Very similar to zotero_emb[0]
            [0.8, 0.2], # Similar to zotero_emb[1]
            [0.0, 0.0]  # Very dissimilar
        ], dtype=np.float32)

        zotero_emb = np.array([
            [0.1, 0.9], # Newest zotero
            [0.8, 0.2]  # Older zotero
        ], dtype=np.float32)
        
        # Test 1: Simple scores
        scores = calculate_scores(cand_emb, zotero_emb)
        self.assertEqual(scores.shape[0], 3)
        
        # Manually verify scores (approximate with time decay)
        # zotero_weights = [w0, w1]
        # w0 = 1 / (1 + log10(1)) = 1
        # w1 = 1 / (1 + log10(2)) = 1 / 1.3 = 0.76
        # Normalized sum: 1 + 0.76 = 1.76. Weights: [1/1.76, 0.76/1.76] = [0.568, 0.432]
        
        # cand0 vs zotero: dot([0.1,0.9], [0.1,0.9])=0.82; dot([0.1,0.9], [0.8,0.2])=0.26
        # score0 = (0.82 * 0.568 + 0.26 * 0.432) * 10 = (0.466 + 0.112) * 10 = 5.78
        # The exact values depend on log10 base, and rounding. Just check relative order.
        self.assertGreater(scores[0], scores[1]) # cand0 closer to newest zotero
        self.assertGreater(scores[1], scores[2]) # cand1 closer to older zotero than neutral

    def test_calculate_scores_empty_zotero(self):
        cand_emb = np.random.rand(5, self.dim).astype(np.float32)
        zotero_emb_none = None
        zotero_emb_empty = np.array([], dtype=np.float32).reshape(0, self.dim) # Empty 2D array
        
        scores_none = calculate_scores(cand_emb, zotero_emb_none)
        self.assertTrue(np.all(scores_none == 0))
        
        scores_empty = calculate_scores(cand_emb, zotero_emb_empty)
        self.assertTrue(np.all(scores_empty == 0))

    def test_encode_texts(self):
        texts = ["hello world", "this is a test"]
        with patch('zotero_daily.recommender.SentenceTransformer') as MockSentenceTransformer:
            mock_encoder = Mock()
            MockSentenceTransformer.return_value = mock_encoder
            mock_encoder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            
            embeddings = encode_texts(texts)
            
            mock_encoder.encode.assert_called_once_with(texts, normalize_embeddings=True)
            self.assertTrue(np.array_equal(embeddings, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)))

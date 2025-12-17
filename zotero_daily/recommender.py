import numpy as np
from sentence_transformers import SentenceTransformer
from zotero_daily.paper import BasePaper
from datetime import datetime
import concurrent.futures

def get_encoder(model_name='avsolatorio/GIST-small-Embedding-v0'):
    return SentenceTransformer(model_name)

def encode_texts(texts, model_name='avsolatorio/GIST-small-Embedding-v0'):
    encoder = get_encoder(model_name)
    # GIST and most retrieval models work best with normalized vectors (Cosine Similarity)
    return encoder.encode(texts, normalize_embeddings=True)

def calculate_scores(candidate_embeddings, zotero_embeddings):
    """
    Calculate scores based on similarity and time decay.
    Assumes zotero_embeddings are sorted by date (newest first).
    """
    if zotero_embeddings is None or len(zotero_embeddings) == 0:
        return np.zeros(candidate_embeddings.shape[0])
        
    n_zotero = zotero_embeddings.shape[0]
    # Time decay: newer papers (lower index) have higher weight
    time_decay_weight = 1 / (1 + np.log10(np.arange(n_zotero) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    
    # Dot product for cosine similarity (vectors normalized)
    # candidate [Nc, D] . zotero.T [D, Nz] -> [Nc, Nz]
    sim = np.dot(candidate_embeddings, zotero_embeddings.T)
    
    # Weighted sum
    scores = (sim * time_decay_weight).sum(axis=1) * 10
    return scores

def rerank_paper(candidate:list[BasePaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[BasePaper]:
    # Legacy wrapper
    encoder = get_encoder(model)
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    
    corpus_texts = [paper['data']['abstractNote'] for paper in corpus]
    candidate_texts = [paper.summary for paper in candidate]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_corpus = executor.submit(encoder.encode, corpus_texts, normalize_embeddings=True)
        f_candidate = executor.submit(encoder.encode, candidate_texts, normalize_embeddings=True)
        corpus_feature = f_corpus.result()
        candidate_feature = f_candidate.result()
        
    scores = calculate_scores(candidate_feature, corpus_feature)
    
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate

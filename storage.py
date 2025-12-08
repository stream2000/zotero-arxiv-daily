import os
import sqlite3
import pickle
import faiss
import numpy as np
from loguru import logger

class Storage:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "metadata.db")
        self.zotero_index_path = os.path.join(data_dir, "zotero.index")
        self.candidate_index_path = os.path.join(data_dir, "candidates.index")
        
        self._init_db()
        self.zotero_index = self._load_or_create_index(self.zotero_index_path)
        self.candidate_index = self._load_or_create_index(self.candidate_index_path)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS zotero (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        abstract TEXT,
                        date_added TEXT,
                        raw_data BLOB
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS candidates (
                        id TEXT PRIMARY KEY,
                        source TEXT,
                        title TEXT,
                        abstract TEXT,
                        category TEXT,
                        score REAL,
                        faiss_id INTEGER,
                        raw_data BLOB
                    )''')
        conn.commit()
        conn.close()

    def _load_or_create_index(self, path):
        if os.path.exists(path):
            try:
                return faiss.read_index(path)
            except Exception as e:
                logger.error(f"Failed to load index {path}: {e}")
                return None
        return None

    def _create_index(self, dim):
        # Inner Product for Cosine Similarity (normalized vectors)
        return faiss.IndexFlatIP(dim)

    def update_zotero(self, papers, embeddings):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM zotero") 
        
        data_to_insert = []
        for i, p in enumerate(papers):
            pid = p['key']
            title = p['data'].get('title', '')
            abstract = p['data'].get('abstractNote', '')
            date = p['data'].get('dateAdded', '')
            raw = pickle.dumps(p)
            data_to_insert.append((pid, title, abstract, date, raw))
            
        c.executemany("INSERT INTO zotero VALUES (?,?,?,?,?)", data_to_insert)
        conn.commit()
        conn.close()
        
        dim = embeddings.shape[1]
        self.zotero_index = self._create_index(dim)
        self.zotero_index.add(embeddings.astype(np.float32))
        faiss.write_index(self.zotero_index, self.zotero_index_path)
        logger.info(f"Updated Zotero storage with {len(papers)} papers.")

    def get_existing_candidate_ids(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM candidates")
        ids = {row[0] for row in c.fetchall()}
        conn.close()
        return ids

    def add_candidates(self, papers, embeddings):
        if not papers:
            return
            
        # Ensure index exists
        if self.candidate_index is None:
            dim = embeddings.shape[1]
            self.candidate_index = self._create_index(dim)
            
        start_id = self.candidate_index.ntotal
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        data_to_insert = []
        for i, p in enumerate(papers):
            pid = p.arxiv_id 
            raw = pickle.dumps(p) 
            cat = p._paper.get('category') if hasattr(p, '_paper') and isinstance(p._paper, dict) else ''
            
            faiss_id = start_id + i
            data_to_insert.append((pid, 'biorxiv', p.title, p.summary, cat, 0.0, faiss_id, raw))
            
        c.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?)", data_to_insert)
        conn.commit()
        conn.close()
        
        self.candidate_index.add(embeddings.astype(np.float32))
        faiss.write_index(self.candidate_index, self.candidate_index_path)
        logger.info(f"Added {len(papers)} candidates to storage.")

    def update_scores(self, scores_dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        data = [(s, pid) for pid, s in scores_dict.items()]
        c.executemany("UPDATE candidates SET score = ? WHERE id = ?", data)
        conn.commit()
        conn.close()
        
    def get_zotero_embeddings(self):
        if self.zotero_index and self.zotero_index.ntotal > 0:
             return self.zotero_index.reconstruct_n(0, self.zotero_index.ntotal)
        return None

    def get_candidate_embeddings(self):
         if self.candidate_index and self.candidate_index.ntotal > 0:
             return self.candidate_index.reconstruct_n(0, self.candidate_index.ntotal)
         return None

    def get_all_candidate_ids(self):
        # Return list of IDs corresponding to faiss IDs 0..N
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Order by faiss_id ASC to match embedding matrix
        c.execute("SELECT id FROM candidates ORDER BY faiss_id ASC")
        ids = [row[0] for row in c.fetchall()]
        conn.close()
        return ids

    def get_top_candidates(self, limit=20, filter_categories=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        query = "SELECT raw_data FROM candidates"
        params = []
        
        if filter_categories:
             clauses = [f"category LIKE ?" for _ in filter_categories]
             where_str = " OR ".join(clauses)
             # Handle case where category is empty?
             query += f" WHERE ({where_str})"
             params.extend([f"%{cat}%" for cat in filter_categories])
        
        query += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        return [pickle.loads(r[0]) for r in rows]

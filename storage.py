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
        logger.debug(f"Storage initialized. DB: {self.db_path}, Zotero Index: {self.zotero_index_path}, Candidate Index: {self.candidate_index_path}")
        
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
                        raw_data BLOB
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS biorxiv_history (
                        date TEXT PRIMARY KEY
                    )''')
        
        # Robust migration using PRAGMA
        c.execute("PRAGMA table_info(candidates)")
        columns = [row[1] for row in c.fetchall()]
        
        migrations = {
            "tldr": "TEXT",
            "date": "TEXT",
            "citation_count": "INTEGER",
            "citation_last_updated": "TEXT"
        }
        
        for col, col_type in migrations.items():
            if col not in columns:
                try:
                    c.execute(f"ALTER TABLE candidates ADD COLUMN {col} {col_type}")
                    logger.info(f"Successfully migrated database: Added column '{col}'.")
                except sqlite3.OperationalError as e:
                    logger.error(f"Failed to add column '{col}': {e}")

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

    def _get_paper_date(self, paper) -> str:
        """Extract date from paper object as YYYY-MM-DD string."""
        if paper.source == 'biorxiv':
            return paper._paper.get('date', '')
        elif paper.source == 'arxiv':
            # arxiv.Result.published is a datetime object
            if hasattr(paper._paper, 'published') and paper._paper.published:
                return paper._paper.published.strftime("%Y-%m-%d")
            # Fallback or updated
            if hasattr(paper._paper, 'updated') and paper._paper.updated:
                return paper._paper.updated.strftime("%Y-%m-%d")
        return ''

    def add_candidates(self, papers, embeddings, rebuild_index=True):
        if not papers:
            return
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # First, insert new candidates into SQLite.
        
        # Prepare data for insertion (only new papers). Use INSERT OR IGNORE.
        data_to_insert_sqlite = []
        for p in papers:
            pid = p.arxiv_id
            # Check if this paper already exists in DB to prevent adding to Faiss if already there
            c.execute("SELECT id FROM candidates WHERE id = ?", (pid,))
            if c.fetchone() is None: # Only add if it doesn't exist
                raw = pickle.dumps(p)
                cat = p._paper.get('category') if hasattr(p, '_paper') and isinstance(p._paper, dict) else ''
                date_str = self._get_paper_date(p)
                data_to_insert_sqlite.append((pid, 'biorxiv' if p.source == 'biorxiv' else 'arxiv', p.title, p.summary, cat, 0.0, raw, date_str))
        
        if data_to_insert_sqlite: # Only execute if there's new data
            c.executemany("INSERT OR IGNORE INTO candidates (id, source, title, abstract, category, score, raw_data, date) VALUES (?,?,?,?,?,?,?,?)", data_to_insert_sqlite)
            conn.commit()
            logger.info(f"Inserted {len(data_to_insert_sqlite)} new candidates into SQLite.")
        else:
            logger.info("No new candidates to insert into SQLite.")
        conn.close()

        if rebuild_index:
            # Rebuild Faiss index from all candidates currently in SQLite
            self.rebuild_candidate_index()
        else:
            logger.info("Skipping Faiss index rebuild as requested.")

    def rebuild_candidate_index(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Fetch all candidate metadata and their actual raw_data (papers) to re-encode
        # This is to ensure Faiss index perfectly matches what's in SQLite
        c.execute("SELECT id, raw_data FROM candidates ORDER BY id ASC") # Consistent order
        candidate_db_data = c.fetchall()
        conn.close()

        if not candidate_db_data:
            self.candidate_index = None # No candidates, so no index
            if os.path.exists(self.candidate_index_path):
                os.remove(self.candidate_index_path)
            logger.info("No candidates in DB, Faiss index cleared.")
            return

        papers_from_db = [pickle.loads(row[1]) for row in candidate_db_data]
        
        # Re-encode all abstracts from DB to get fresh embeddings
        # This might be slow if many, but ensures consistency.
        # This also assumes embeddings are always the same for same abstract.
        from recommender import encode_texts # Import here to avoid circular dependency
        candidate_texts = [p.summary for p in papers_from_db]
        embeddings = encode_texts(candidate_texts)
        
        dim = embeddings.shape[1]
        self.candidate_index = self._create_index(dim)
        self.candidate_index.add(embeddings.astype(np.float32))
        faiss.write_index(self.candidate_index, self.candidate_index_path)
        logger.info(f"Rebuilt candidate Faiss index with {len(papers_from_db)} embeddings.")

    def update_scores(self, scores_dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        data = [(s, pid) for pid, s in scores_dict.items()]
        c.executemany("UPDATE candidates SET score = ? WHERE id = ?", data)
        conn.commit()
        conn.close()
        logger.info(f"Updated scores for {len(scores_dict)} candidates.")

    def update_tldr(self, paper_id, tldr_text):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE candidates SET tldr = ? WHERE id = ?", (tldr_text, paper_id))
        conn.commit()
        conn.close()

    def update_citation_count(self, paper_id, count):
        from datetime import datetime
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("UPDATE candidates SET citation_count = ?, citation_last_updated = ? WHERE id = ?", (count, now_str, paper_id))
        conn.commit()
        conn.close()
        logger.debug(f"Updated citation count for {paper_id} to {count}.")
        
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
        # Order by id ASC to match embedding matrix (which is built using ORDER BY id ASC)
        c.execute("SELECT id FROM candidates ORDER BY id ASC")
        ids = [row[0] for row in c.fetchall()]
        conn.close()
        return ids

    def get_all_candidates(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Fetch score, tldr, and citation info
        c.execute("SELECT raw_data, score, tldr, citation_count, citation_last_updated FROM candidates")
        rows = c.fetchall()
        conn.close()
        
        papers = []
        for row in rows:
            p = pickle.loads(row[0])
            p.score = row[1]
            if row[2]:
                p.set_tldr(row[2])
            # Set citation info from DB
            p._citation_count_cache = row[3]
            p._citation_last_updated = row[4]
            papers.append(p)
        return papers

    def get_candidates_by_date_range(self, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Select candidates within the date range
        c.execute("SELECT raw_data, score, tldr, citation_count, citation_last_updated FROM candidates WHERE date >= ? AND date <= ?", (start_date, end_date))
        rows = c.fetchall()
        conn.close()
        
        papers = []
        for row in rows:
            p = pickle.loads(row[0])
            p.score = row[1]
            if row[2]:
                p.set_tldr(row[2])
            # Set citation info from DB
            p._citation_count_cache = row[3]
            p._citation_last_updated = row[4]
            papers.append(p)
        return papers

    def get_top_candidates(self, limit=20):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        query_base = "SELECT id, score, raw_data, tldr FROM candidates"
        params = []
        
        query = query_base

        query += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        
        # Inject scores and tldr into the unpickled BasePaper objects
        papers_with_scores = []
        for row in rows:
            pid, score, raw_data, tldr = row
            paper = pickle.loads(raw_data)
            paper.score = score # Set the score from DB
            paper.set_tldr(tldr) # Set the tldr from DB
            papers_with_scores.append(paper)
            
        return papers_with_scores

    def get_zotero_ids(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM zotero")
        ids = {row[0] for row in c.fetchall()}
        conn.close()
        return ids

    def get_recent_zotero_items(self, limit=25):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT title, abstract FROM zotero ORDER BY date_added DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            items.append({"title": row[0], "abstract": row[1]})
        return items

    def get_zotero_sample(self, recent_count=10, random_count=15):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get recent items
        c.execute("SELECT id, title, abstract FROM zotero ORDER BY date_added DESC LIMIT ?", (recent_count,))
        recent_rows = c.fetchall()
        
        recent_ids = [r[0] for r in recent_rows]
        items = [{"title": r[1], "abstract": r[2]} for r in recent_rows]
        
        # Get random items excluding recent ones
        if random_count > 0:
            placeholders = ','.join(['?'] * len(recent_ids))
            query = f"SELECT title, abstract FROM zotero WHERE id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT ?"
            c.execute(query, recent_ids + [random_count])
            random_rows = c.fetchall()
            items.extend([{"title": r[0], "abstract": r[1]} for r in random_rows])
            
        conn.close()
        return items

    def get_biorxiv_completed_dates(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT date FROM biorxiv_history")
        dates = {row[0] for row in c.fetchall()}
        conn.close()
        return dates

    def mark_biorxiv_date_completed(self, date_str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO biorxiv_history (date) VALUES (?)", (date_str,))
        conn.commit()
        conn.close()
        logger.info(f"Marked BioRxiv date {date_str} as completed.")
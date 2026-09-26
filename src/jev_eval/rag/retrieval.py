"""Shared chunk store, lexical search, optional dense search, and frozen pools."""

import re
import sqlite3
import threading
import uuid
from pathlib import Path

import numpy as np

from jev_eval.io import digest, read_json, write_json

from .contracts import CandidatePool, CandidateRecord, ChunkRecord, QueryRecord


class ChunkIndex:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
          rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,
          document_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
          corpus_scope TEXT, text_hash TEXT NOT NULL, text TEXT NOT NULL,
          title TEXT);
        CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(corpus_scope);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(text, content='chunks', content_rowid='rowid');
        """)

    @property
    def db(self):
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30)
            self._local.connection = connection
        return connection

    @db.setter
    def db(self, connection):
        self._local.connection = connection

    def add(self, chunks):
        count = 0
        for chunk in chunks:
            if not isinstance(chunk, ChunkRecord):
                chunk = ChunkRecord.model_validate(chunk)
            cursor = self.db.execute(
                "INSERT INTO chunks(chunk_id,document_id,dataset_id,corpus_scope,text_hash,text,title) VALUES(?,?,?,?,?,?,?)",
                (chunk.chunk_id, chunk.document_id, chunk.dataset_id, chunk.corpus_scope, chunk.text_hash, chunk.text, chunk.title),
            )
            self.db.execute("INSERT INTO chunk_fts(rowid,text) VALUES(?,?)", (cursor.lastrowid, chunk.text))
            count += 1
            if count % 1000 == 0:
                self.db.commit()
        self.db.commit()
        return count

    def get(self, chunk_id: str) -> ChunkRecord:
        row = self.db.execute(
            "SELECT chunk_id,document_id,dataset_id,corpus_scope,text_hash,text,title FROM chunks WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        cid, did, dataset, scope, text_hash, text, title = row
        return ChunkRecord(dataset_id=dataset, document_id=did, chunk_id=cid, text=text, text_hash=text_hash, title=title, corpus_scope=scope)

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def search_bm25(self, query: QueryRecord, depth: int) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(re.findall(r"\w+", query.text.casefold())))
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        sql = "SELECT c.chunk_id, -bm25(chunk_fts) AS score FROM chunk_fts JOIN chunks c ON c.rowid=chunk_fts.rowid WHERE chunk_fts MATCH ?"
        params = [expression]
        if query.corpus_scope is not None:
            sql += " AND c.corpus_scope=?"
            params.append(query.corpus_scope)
        sql += " ORDER BY score DESC,c.chunk_id LIMIT ?"
        params.append(depth)
        return [(cid, float(score)) for cid, score in self.db.execute(sql, params)]

    def all_ids(self, scope=None):
        if scope is None:
            return [row[0] for row in self.db.execute("SELECT chunk_id FROM chunks ORDER BY rowid")]
        return [row[0] for row in self.db.execute("SELECT chunk_id FROM chunks WHERE corpus_scope=? ORDER BY rowid", (scope,))]


def rrf(rankings: list[list[str]], constant: int = 60) -> list[tuple[str, float]]:
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Duplicate source rank IDs")
        for position, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (constant + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def freeze_pool(query: QueryRecord, retriever_id: str, results: list[tuple[str, float]], index: ChunkIndex, depth: int):
    candidates = []
    for rank, (cid, score) in enumerate(results[:depth], 1):
        chunk = index.get(cid)
        if chunk.dataset_id != query.dataset_id or (query.corpus_scope is not None and chunk.corpus_scope != query.corpus_scope):
            raise ValueError("Retriever returned a chunk outside query corpus scope")
        candidates.append(CandidateRecord(query_id=query.query_id, chunk_id=cid, rank=rank, score=score, text_hash=chunk.text_hash, source_document_id=chunk.document_id))
    return CandidatePool.build(query.query_id, retriever_id, candidates)


class DenseIndex:
    """Exact fixture search; FAISS index used when installed for large corpora."""

    def __init__(self, index: ChunkIndex, vector_path: str | Path, encoder):
        self.index, self.vector_path, self.encoder = index, Path(vector_path), encoder
        self.ids = index.all_ids()
        self.id_to_row = {cid: row for row, cid in enumerate(self.ids)}
        meta = read_json(self.vector_path.with_suffix(".meta.json"))
        if meta["ids_hash"] != digest(self.ids):
            raise ValueError("Dense index row mapping differs from chunk index")
        if meta.get("encoder") != {"name": getattr(encoder, "name", None), "revision": getattr(encoder, "revision", None)}:
            raise ValueError("Dense index was built with a different embedding model or revision")
        self.vectors = np.load(self.vector_path, mmap_mode="r")
        if len(self.ids) != len(self.vectors) or self.vectors.shape[1] != meta["dimension"]:
            raise ValueError("Dense vector and chunk row counts differ")
        self.faiss_index = None
        try:
            import faiss

            faiss_path = self.vector_path.with_suffix(".faiss")
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
        except ImportError:
            pass

    @staticmethod
    def build(index: ChunkIndex, vector_path: str | Path, encoder, batch_size=32):
        ids = index.all_ids()
        if not ids:
            raise ValueError("Cannot embed an empty corpus")
        vector_path = Path(vector_path)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        if vector_path.exists() or vector_path.with_suffix(".meta.json").exists():
            raise FileExistsError("Dense index already exists; use a new run ID for changed inputs")
        pending = vector_path.with_name(f".{vector_path.stem}.pending-{uuid.uuid4().hex}.npy")
        first = np.asarray(encoder.encode([index.get(ids[0]).text]), dtype=np.float32)
        if first.ndim != 2 or first.shape[0] != 1 or not np.isfinite(first).all():
            raise ValueError("Invalid encoder output")
        matrix = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32, shape=(len(ids), first.shape[1]))
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(first.shape[1])
        except ImportError:
            faiss_index = None
            if len(ids) > 100_000:
                raise RuntimeError("FAISS is required for dense indexes above 100,000 chunks") from None
        for start in range(0, len(ids), batch_size):
            texts = [index.get(cid).text for cid in ids[start : start + batch_size]]
            batch = np.asarray(encoder.encode(texts), dtype=np.float32)
            if batch.shape != (len(texts), first.shape[1]) or not np.isfinite(batch).all():
                raise ValueError("Invalid encoder batch shape or values")
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Zero dense vector")
            batch /= norms
            matrix[start : start + len(batch)] = batch
            if faiss_index is not None:
                faiss_index.add(batch)
        matrix.flush()
        if faiss_index is not None:
            faiss.write_index(faiss_index, str(pending.with_suffix(".faiss")))
        meta = {"count": len(ids), "dimension": matrix.shape[1], "ids_hash": digest(ids),
                "encoder": {"name": getattr(encoder, "name", None), "revision": getattr(encoder, "revision", None)}}
        pending.replace(vector_path)
        pending_faiss = pending.with_suffix(".faiss")
        if pending_faiss.exists():
            pending_faiss.replace(vector_path.with_suffix(".faiss"))
        write_json(vector_path.with_suffix(".meta.json"), meta)
        return meta

    def search(self, query: QueryRecord, depth: int):
        vector = np.asarray(self.encoder.encode([query.text], query=True)[0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid query embedding")
        vector /= norm
        if query.corpus_scope is not None:
            rows = np.array([self.id_to_row[cid] for cid in self.index.all_ids(query.corpus_scope)], dtype=np.int64)
            scores = np.asarray(self.vectors[rows]) @ vector
            hits = [(self.ids[rows[i]], float(scores[i])) for i in range(len(rows))]
        elif self.faiss_index is not None:
            scores, rows = self.faiss_index.search(vector.reshape(1, -1), depth)
            hits = [(self.ids[int(i)], float(score)) for i, score in zip(rows[0], scores[0]) if i >= 0]
        else:
            if len(self.ids) > 100_000:
                raise RuntimeError("FAISS index missing for large dense search")
            scores = np.asarray(self.vectors) @ vector
            hits = list(zip(self.ids, map(float, scores)))
        return sorted(hits, key=lambda item: (-item[1], item[0]))[:depth]

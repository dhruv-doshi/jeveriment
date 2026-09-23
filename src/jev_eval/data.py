"""Streaming BEIR ingestion, preserving IDs and explicit zero judgments."""

import csv
import json
import sqlite3
import tempfile
from pathlib import Path

from .contracts import Document, Qrel, Query
from .io import digest, file_digest, write_json


def ingest_beir(source, destination, split="train", sample_queries=None, seed=1729):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "corpus.sqlite").exists():
        raise ValueError("Prepared corpus already exists; use a new run directory")
    # A malformed or interrupted source must not leave a half-prepared corpus.
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=destination) as staging:
        stage = Path(staging)
        connection = sqlite3.connect(stage / "corpus.sqlite")
        try:
            manifest = _ingest_rows(
                source, stage, connection, split, sample_queries, seed
            )
        finally:
            connection.close()
        for name in (
            "corpus.sqlite",
            "queries.json",
            "qrels.json",
            "dataset_manifest.json",
        ):
            (stage / name).replace(destination / name)
    return manifest


def _ingest_rows(source, destination, connection, split, sample_queries, seed):
    connection.execute(
        "CREATE TABLE documents(doc_id TEXT PRIMARY KEY, title TEXT, text TEXT, content_sha256 TEXT)"
    )
    count = 0
    with (source / "corpus.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            values = {"title": row.get("title", ""), "text": row["text"]}
            document = Document(
                doc_id=row["_id"], **values, content_sha256=digest(values)
            )
            connection.execute(
                "INSERT INTO documents VALUES(?,?,?,?)",
                tuple(document.model_dump().values()),
            )
            count += 1
    connection.commit()
    qrels = {}
    with (source / "qrels" / f"{split}.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            judgment = Qrel(
                query_id=row["query-id"],
                doc_id=row["corpus-id"],
                grade=float(row["score"]),
                judgment_status="judged",
            )
            if judgment.doc_id in qrels.setdefault(judgment.query_id, {}):
                raise ValueError("Duplicate judgment pair")
            if (
                connection.execute(
                    "SELECT 1 FROM documents WHERE doc_id=?", (judgment.doc_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("Qrel references missing corpus document")
            qrels[judgment.query_id][judgment.doc_id] = judgment.grade
    queries = {}
    with (source / "queries.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["_id"] in qrels:
                query = Query(query_id=row["_id"], text=row["text"], split=split)
                if query.query_id in queries:
                    raise ValueError("Duplicate query ID")
                queries[query.query_id] = query.model_dump()
    if set(queries) != set(qrels):
        raise ValueError("Qrels reference missing queries")
    if not queries or not count:
        raise ValueError("Dataset must contain documents and split queries")
    selected = sorted(queries, key=lambda q: (digest({"seed": seed, "query_id": q}), q))
    if sample_queries:
        selected = selected[:sample_queries]
    write_json(destination / "queries.json", {q: queries[q] for q in selected})
    # Labels live in their own artifact and are never passed into scorers.
    write_json(destination / "qrels.json", {q: qrels[q] for q in selected})
    manifest = {
        "corpus_count": count,
        "split": split,
        "eligible_query_count": len(queries),
        "selected_query_ids": selected,
        "seed": seed,
        "files": {
            str(p.relative_to(source)): file_digest(p)
            for p in [
                source / "corpus.jsonl",
                source / "queries.jsonl",
                source / "qrels" / f"{split}.tsv",
            ]
        },
        "license": "upstream terms; not cleared for redistribution",
    }
    manifest["revision"] = digest(manifest["files"])
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest


def iter_documents(database):
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        for row in connection.execute(
            "SELECT doc_id,title,text,content_sha256 FROM documents ORDER BY doc_id"
        ):
            yield Document(
                doc_id=row[0], title=row[1], text=row[2], content_sha256=row[3]
            )
    finally:
        connection.close()

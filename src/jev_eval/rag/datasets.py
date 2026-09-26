"""Official-format readers. Downloaded source files stay outside Git."""

import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from jev_eval.io import digest

from .contracts import ChunkRecord, GoldRecord, QueryRecord


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _json_records(path: Path) -> Iterator[dict]:
    with _open(path) as stream:
        if path.name.endswith((".json", ".json.gz")) and path.name != "kilt_knowledgesource.json":
            value = json.load(stream)
            if isinstance(value, list):
                yield from value
            elif isinstance(value, dict):
                yield from value.values()
            else:
                raise ValueError(f"Unsupported JSON root: {path}")
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _chunk(dataset: str, document: str, unit: str, text: str, *, title=None, scope=None):
    if not text.strip():
        return None
    return ChunkRecord(
        dataset_id=dataset,
        document_id=str(document),
        chunk_id=f"{document}:{unit}",
        text=text.strip(),
        text_hash=digest(text.strip()),
        title=title,
        corpus_scope=scope,
    )


def _split(query_id: str, seed: int = 1729) -> str:
    return "development" if int(digest(f"{seed}:{query_id}")[:8], 16) % 5 == 0 else "reporting"


def _rows(value):
    """Hugging Face sequence features can arrive as rows or columns."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        count = len(next(iter(value.values()), []))
        return [{key: values[i] for key, values in value.items()} for i in range(count)]
    raise ValueError("Expected nested rows or columns")


class DatasetAdapter:
    id: str
    required_files: tuple[str, ...]

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def audit(self):
        missing = [name for name in self.required_files if not (self.root / name).is_file()]
        return {"dataset": self.id, "root": str(self.root), "missing": missing, "ready": not missing}

    def require_files(self):
        audit = self.audit()
        if audit["missing"]:
            raise FileNotFoundError(f"{self.id} missing official source files: {audit['missing']}")

    def iter_chunks(self) -> Iterator[ChunkRecord]:
        raise NotImplementedError

    def iter_queries(self) -> Iterator[tuple[QueryRecord, GoldRecord]]:
        raise NotImplementedError


class Mirage(DatasetAdapter):
    id = "mirage"
    required_files = ("benchmark.json",)

    def _corpus_files(self):
        combined = self.root / "medrag_corpus.jsonl"
        if combined.is_file():
            return [combined]
        return sorted((self.root / "medrag").glob("**/chunk/*.jsonl"))

    def audit(self):
        report = super().audit()
        if not self._corpus_files():
            report["missing"].append("medrag_corpus.jsonl or medrag/**/chunk/*.jsonl")
        report["ready"] = not report["missing"]
        return report

    def iter_chunks(self):
        self.require_files()
        for path in self._corpus_files():
            for row in _json_records(path):
                doc_id = row.get("id") or row.get("_id")
                text = row.get("content") or row.get("text")
                if not doc_id or not isinstance(text, str):
                    raise ValueError("MedRAG corpus row needs id and content/text")
                chunk = _chunk(self.id, str(doc_id), "0", text, title=row.get("title"))
                if chunk:
                    yield chunk

    def iter_queries(self):
        self.require_files()
        benchmark = json.loads((self.root / "benchmark.json").read_text())
        if not isinstance(benchmark, dict):
            raise ValueError("MIRAGE benchmark root must map subtask to questions")
        for subtask, rows in benchmark.items():
            if isinstance(rows, dict):
                rows = list(rows.values())
            for i, row in enumerate(rows):
                qid = f"{subtask}:{row.get('id', i)}"
                question = row["question"]
                options = row.get("options") or {}
                answer = row.get("answer")
                if answer is None:
                    raise ValueError("MIRAGE question has no public answer")
                yield (
                    QueryRecord(dataset_id=self.id, query_id=qid, text=question, split=_split(qid), task="medical_multiple_choice", answer_options=options),
                    GoldRecord(dataset_id=self.id, query_id=qid, answers=(str(answer),), metric_availability={"retrieval_qrels": "not_provided_by_benchmark"}),
                )


class Clerc(DatasetAdapter):
    id = "clerc"
    required_files = (
        "collection/collection.passage.tsv.gz",
        "queries/test.single-removed.direct.tsv",
        "qrels/qrels-passage.test.direct.tsv",
    )

    def iter_chunks(self):
        self.require_files()
        with _open(self.root / self.required_files[0]) as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < 2:
                    raise ValueError("CLERC passage row needs ID and text")
                chunk = _chunk(self.id, row[0], "0", row[1])
                if chunk:
                    yield chunk

    def iter_queries(self):
        self.require_files()
        qrels = defaultdict(dict)
        with _open(self.root / self.required_files[2]) as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) == 4:
                    qid, _, pid, grade = row
                elif len(row) == 3:
                    qid, pid, grade = row
                else:
                    raise ValueError("Unexpected CLERC qrel row")
                qrels[qid][f"{pid}:0"] = int(grade)
        with _open(self.root / self.required_files[1]) as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < 2:
                    raise ValueError("CLERC query row needs ID and text")
                qid, question = row[:2]
                if qid not in qrels:
                    raise ValueError(f"CLERC query {qid} has no qrels")
                yield (
                    QueryRecord(dataset_id=self.id, query_id=qid, text=question, split=_split(qid), task="legal_retrieval_generation"),
                    GoldRecord(dataset_id=self.id, query_id=qid, qrels=qrels[qid], metric_availability={"answer": "retrieval_split_has_no_reference_answer"}),
                )


class HotpotQA(DatasetAdapter):
    id = "hotpotqa"
    required_files = ("hotpot_dev_fullwiki_v1.json", "wiki.jsonl")

    def iter_chunks(self):
        self.require_files()
        for article in _json_records(self.root / "wiki.jsonl"):
            title = article["title"]
            sentences = article.get("sentences") or article.get("text")
            if isinstance(sentences, str):
                raise ValueError("Hotpot fullwiki corpus needs sentence arrays to map supporting facts")
            for i, sentence in enumerate(sentences):
                chunk = _chunk(self.id, title, str(i), sentence, title=title)
                if chunk:
                    yield chunk

    def iter_queries(self):
        self.require_files()
        for row in _json_records(self.root / self.required_files[0]):
            qid = row["_id"]
            facts = tuple(f"{title}:{sentence}" for title, sentence in row.get("supporting_facts", []))
            if not facts or "answer" not in row:
                raise ValueError("Hotpot fullwiki development labels missing")
            yield (
                QueryRecord(dataset_id=self.id, query_id=qid, text=row["question"], split="reporting", task="multi_hop_question_answering"),
                GoldRecord(dataset_id=self.id, query_id=qid, answers=(row["answer"],), qrels={fact: 1 for fact in facts}, evidence_sets=(facts,)),
            )


class Qasper(DatasetAdapter):
    id = "qasper"
    required_files = ()

    def _parquet_files(self):
        return sorted(self.root.glob("qasper/*.parquet"))

    def audit(self):
        report = super().audit()
        if not (self.root / "qasper.jsonl").is_file() and not self._parquet_files():
            report["missing"].append("qasper.jsonl or qasper/*.parquet")
        report["ready"] = not report["missing"]
        return report

    def _papers(self):
        self.require_files()
        jsonl = self.root / "qasper.jsonl"
        if jsonl.is_file():
            yield from _json_records(jsonl)
            return
        import pyarrow.parquet as pq

        for path in self._parquet_files():
            split = path.name.split("-")[0]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    row["split"] = split
                    yield row

    def iter_chunks(self):
        for paper in self._papers():
            paper_id = paper["id"]
            for section_index, section in enumerate(_rows(paper["full_text"])):
                for paragraph_index, paragraph in enumerate(section["paragraphs"]):
                    chunk = _chunk(self.id, paper_id, f"{section_index}:{paragraph_index}", paragraph, title=section.get("section_name"), scope=paper_id)
                    if chunk:
                        yield chunk

    def iter_queries(self):
        for paper in self._papers():
            paper_id = paper["id"]
            source_split = paper.get("split")
            split = {"train": "development", "validation": "reporting", "test": "heldout"}.get(
                source_split, _split(paper_id)
            )
            paragraphs = list(self._paper_paragraphs(paper))
            for qa in _rows(paper["qas"]):
                qid = qa["question_id"]
                answers = []
                qrels = {}
                for annotation in _rows(qa["answers"]):
                    answer = annotation["answer"]
                    if answer.get("unanswerable"):
                        answers.append("unanswerable")
                    elif answer.get("extractive_spans") or answer.get("free_form_answer"):
                        answers.extend(answer.get("extractive_spans") or [])
                        if answer.get("free_form_answer"):
                            answers.append(answer["free_form_answer"])
                    else:
                        answers.append("yes" if answer.get("yes_no") else "no")
                    for evidence in answer.get("evidence") or []:
                        matches = [cid for cid, text in paragraphs if evidence.strip() and evidence.strip() in text]
                        if not matches:
                            raise ValueError(f"QASPER evidence does not map to paper {paper_id}")
                        qrels.update({cid: 1 for cid in matches})
                yield (
                    QueryRecord(dataset_id=self.id, query_id=qid, text=qa["question"], split=split, task="scientific_paper_question_answering", corpus_scope=paper_id),
                    GoldRecord(dataset_id=self.id, query_id=qid, answers=tuple(dict.fromkeys(answers)), qrels=qrels),
                )

    @staticmethod
    def _paper_paragraphs(paper):
        for section_index, section in enumerate(_rows(paper["full_text"])):
            for paragraph_index, paragraph in enumerate(section["paragraphs"]):
                yield f"{paper['id']}:{section_index}:{paragraph_index}", paragraph


class KiltFever(DatasetAdapter):
    id = "kilt_fever"
    required_files = ("kilt_knowledgesource.json", "fever-dev-kilt.jsonl")

    def iter_chunks(self):
        self.require_files()
        for page in _json_records(self.root / self.required_files[0]):
            page_id = str(page["wikipedia_id"])
            for i, paragraph in enumerate(page["text"]):
                chunk = _chunk(self.id, page_id, str(i), paragraph, title=page.get("wikipedia_title"))
                if chunk:
                    yield chunk

    def iter_queries(self):
        self.require_files()
        for row in _json_records(self.root / self.required_files[1]):
            qid = str(row["id"])
            answers, qrels, evidence_sets = [], {}, []
            for output in row.get("output", []):
                if output.get("answer"):
                    answers.append(output["answer"])
                evidence = tuple(
                    f"{item['wikipedia_id']}:{paragraph}"
                    for item in output.get("provenance", [])
                    for paragraph in range(item["start_paragraph_id"], item.get("end_paragraph_id", item["start_paragraph_id"]) + 1)
                )
                if evidence:
                    evidence_sets.append(evidence)
                    qrels.update({cid: 1 for cid in evidence})
            if not answers:
                raise ValueError("KILT-FEVER labeled development answer missing")
            yield (
                QueryRecord(dataset_id=self.id, query_id=qid, text=row["input"], split="reporting", task="fact_verification"),
                GoldRecord(dataset_id=self.id, query_id=qid, answers=tuple(dict.fromkeys(answers)), qrels=qrels, evidence_sets=tuple(evidence_sets)),
            )


ADAPTERS = {adapter.id: adapter for adapter in (Mirage, Clerc, HotpotQA, Qasper, KiltFever)}


def adapter_for(dataset_id: str, source_root: str | Path) -> DatasetAdapter:
    return ADAPTERS[dataset_id](Path(source_root) / dataset_id)

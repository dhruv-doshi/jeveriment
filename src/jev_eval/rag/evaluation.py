"""Native answer/evidence metrics and explicit applicability."""

import re
from collections import Counter

from jev_eval.metrics import evaluate_query

from .contracts import ContextSelection, GenerationResult, GoldRecord, RerankResult


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def token_f1(predicted: str, reference: str) -> float:
    p, r = normalize(predicted).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def evaluate_ranking(result: RerankResult, gold: GoldRecord):
    if not gold.qrels:
        return {"applicable": False, "reason": gold.metric_availability.get("retrieval_qrels", "no_qrels")}
    return {"applicable": True, **evaluate_query(list(result.ordered_ids), gold.qrels)}


def evaluate_answer(result: GenerationResult, gold: GoldRecord, context: ContextSelection):
    if result.query_id != gold.query_id or context.query_id != gold.query_id:
        raise ValueError("Evaluation query IDs differ")
    output = {"citation_validity": len(set(result.citations) & set(context.selected_ids)) / len(result.citations) if result.citations else None,
              "used_evidence_tokens": context.estimated_tokens}
    if gold.qrels:
        positives = {cid for cid, grade in gold.qrels.items() if grade > 0}
        output["context_evidence_recall"] = len(positives & set(context.selected_ids)) / len(positives) if positives else None
        output["citation_evidence_precision"] = len(positives & set(result.citations)) / len(result.citations) if result.citations else None
    else:
        output["context_evidence_recall"] = None
        output["citation_evidence_precision"] = None
    if not gold.answers:
        output.update({"answer_applicable": False, "reason": gold.metric_availability.get("answer", "no_reference_answer")})
        return output
    output["answer_applicable"] = True
    output["answer_exact_match"] = float(any(normalize(result.answer) == normalize(ref) for ref in gold.answers))
    output["answer_token_f1"] = max(token_f1(result.answer, ref) for ref in gold.answers)
    return output

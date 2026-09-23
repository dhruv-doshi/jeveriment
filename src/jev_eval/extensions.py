"""Optional evidence-set and RAG audit helpers; never manufacture human labels."""

import numpy as np


def mmr(ids, relevance, embeddings, k, diversity_weight=0.7):
    if (
        len(set(ids)) != len(ids)
        or len(ids) != len(embeddings)
        or set(ids) != set(relevance)
    ):
        raise ValueError("MMR candidate mapping mismatch")
    if not 0 <= diversity_weight <= 1 or k < 1:
        raise ValueError("Invalid MMR parameters")
    vectors = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embeddings")
    vectors /= norms
    rows = {d: i for i, d in enumerate(ids)}
    selected = []
    while len(selected) < min(k, len(ids)):

        def score(d):
            redundancy = max(
                (float(vectors[rows[d]] @ vectors[rows[s]]) for s in selected),
                default=0.0,
            )
            return diversity_weight * relevance[d] - (1 - diversity_weight) * redundancy

        selected.append(
            min((d for d in ids if d not in selected), key=lambda d: (-score(d), d))
        )
    return selected


def evidence_coverage(selected_ids, alternative_support_sets):
    if not alternative_support_sets or any(not s for s in alternative_support_sets):
        raise ValueError("Evidence coverage requires nonempty annotated support sets")
    selected = set(selected_ids)
    return {
        "best_support_recall": max(
            len(selected & set(s)) / len(set(s)) for s in alternative_support_sets
        ),
        "complete_evidence_set": any(
            set(s) <= selected for s in alternative_support_sets
        ),
    }


def rag_audit(outputs):
    """Aggregate independent adjudicated claim/citation records, including abstention."""
    if not outputs:
        raise ValueError("No audited generations")
    answered, grounded, cited, total_citations, supported_claims, total_claims = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    for output in outputs:
        if output.get("audit_source") != "independent_human":
            raise ValueError("RAG evaluation needs independent human review")
        if output.get("abstained"):
            continue
        answered += 1
        claims = output.get("claims", [])
        if not claims:
            raise ValueError("Answered output is missing claim audit")
        grounded += all(c["supported"] for c in claims)
        for claim in claims:
            total_claims += 1
            supported_claims += bool(
                claim["supported"] and claim["has_supporting_citation"]
            )
            for citation in claim.get("citations", []):
                if citation["doc_id"] not in output["evidence_ids"]:
                    raise ValueError(
                        "Citation refers to evidence not supplied to generator"
                    )
                total_citations += 1
                cited += bool(citation["supports_claim"])
    return {
        "answer_coverage": answered / len(outputs),
        "grounded_answer_rate": grounded / answered if answered else None,
        "citation_correctness": cited / total_citations if total_citations else None,
        "citation_completeness": supported_claims / total_claims
        if total_claims
        else None,
    }

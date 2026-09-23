"""Import a reproducible remote fixed-pool stage without claiming local indexing."""

from pathlib import Path

from .config import load_config
from .contracts import Pool
from .io import digest, read_json, write_json


def import_fixed_pool(config_path, bundle_path):
    config = load_config(config_path)
    bundle = read_json(bundle_path)
    manifest = bundle["dataset_manifest"]
    required = {
        "revision",
        "corpus_count",
        "split",
        "license",
        "source_stage",
        "query_count",
    }
    if (
        not required <= set(manifest)
        or manifest["source_stage"] != "remote_full_corpus_retrieval"
    ):
        raise ValueError("Remote pools require complete first-stage provenance")
    if manifest["split"] != config.split or (
        config.corpus_revision and config.corpus_revision != manifest["revision"]
    ):
        raise ValueError("Remote split/revision differs from requested experiment")
    language = manifest.get("language", "en")
    if language != "en" and not manifest.get("human_language_review"):
        raise ValueError("Multilingual extension requires a human-reviewed task rubric")
    queries, views, pools = bundle["queries"], bundle["text_views"], bundle["pools"]
    if set(queries) != set(pools) or len(queries) != manifest["query_count"]:
        raise ValueError("Remote query/pool counts mismatch")
    for q, raw in pools.items():
        pool = Pool.model_validate(raw)
        if pool.query_id != q:
            raise ValueError("Remote pool query ID mismatch")
        for candidate in pool.candidates:
            view = views[candidate.doc_id]
            if (
                digest(view["text"]) != view["hash"]
                or view["hash"] != candidate.text_view_hash
            ):
                raise ValueError("Remote evidence hash mismatch")
        ids = {c.doc_id for c in pool.candidates}
        for name in ("bm25", "dense", "rrf"):
            if name == "rrf" and not ids <= set(bundle["source_runs"][name][q]):
                raise ValueError("Remote RRF omits pool members")
    if set(bundle["qrels"]) != set(queries):
        raise ValueError("Remote qrels/query mapping mismatch")
    models = bundle["model_manifest"]
    for name in (config.dense_model, config.reranker_model):
        if not models[name].get("revision"):
            raise ValueError("Remote/local model revisions must be frozen")
    dest = Path("runs") / config.experiment_id
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Remote import requires a fresh run directory")
    for name in (
        "queries",
        "text_views",
        "pools",
        "source_runs",
        "qrels",
        "model_manifest",
        "dataset_manifest",
    ):
        write_json(dest / f"{name}.json", bundle[name])
    resolved = {
        **config.model_dump(),
        "corpus_revision": manifest["revision"],
        "dense_revision": models[config.dense_model]["revision"],
        "reranker_revision": models[config.reranker_model]["revision"],
    }
    write_json(dest / "input_config.json", config.model_dump())
    write_json(dest / "resolved_config.json", resolved)
    write_json(
        dest / "remote_provenance.json",
        {
            "bundle_hash": digest(bundle),
            "execution": "remote_first_stage_local_fixed_pool_evaluation",
            "leaderboard_snapshot": bundle.get("leaderboard_snapshot"),
            "training_exposure": bundle.get("training_exposure", "unknown"),
        },
    )
    return dest


def import_external_scores(run_path, name, artifact_path):
    from .metrics import rank

    if not name.replace("_", "").isalnum():
        raise ValueError("Invalid external system name")
    dest = Path(run_path)
    artifact = read_json(artifact_path)
    metadata = artifact["metadata"]
    if not all(
        metadata.get(k)
        for k in ("model", "revision", "provider", "input_view_policy", "resources")
    ):
        raise ValueError(
            "External scorer requires identity, text policy, and resource provenance"
        )
    pools = read_json(dest / "pools.json")
    if set(artifact["scores"]) != set(pools):
        raise ValueError("External scorer must account for every query")
    for q, p in pools.items():
        if artifact["ordered_pool_hashes"][q] != p["ordered_input_hash"]:
            raise ValueError("External scorer used different pools or text")
        rank(artifact["scores"][q], [c["doc_id"] for c in p["candidates"]])
    write_json(
        dest / f"scores_external_{name}.json",
        {
            "status": "complete",
            "scores": artifact["scores"],
            "metadata": metadata,
            "provenance": digest(artifact),
        },
    )

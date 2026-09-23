import numpy as np

from .io import digest


def checked_pairs(pairs, allowed_splits=None):
    if not pairs:
        raise ValueError("Calibration requires explicit human judgments")
    seen = set()
    for row in pairs:
        key = row["dataset"], row["query_id"], row["doc_id"]
        if key in seen:
            raise ValueError("Duplicate calibration pair")
        seen.add(key)
        if row.get("judgment_status") != "judged" or row.get("label") not in (0, 1):
            raise ValueError(
                "Unjudged/insufficient pairs cannot become calibration negatives"
            )
        if row.get("label_source") != "human_audit":
            raise ValueError("Primary calibration requires an independent human audit")
        if allowed_splits and row.get("split") not in allowed_splits:
            raise ValueError("Invalid calibration split")
        if not np.isfinite(row["probability"]) or not 0 <= row["probability"] <= 1:
            raise ValueError("Invalid probability")
        if not 0 < row.get("inclusion_probability", 1) <= 1:
            raise ValueError("Invalid sampling probability")


def calibration_metrics(pairs, bins=10):
    checked_pairs(pairs)
    p = np.array([r["probability"] for r in pairs], dtype=float)
    y = np.array([r["label"] for r in pairs], dtype=float)
    w = np.array([1 / r.get("inclusion_probability", 1) for r in pairs], dtype=float)
    w /= w.sum()
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    rows, ece = [], 0.0
    indices = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        mask = indices == b
        mass = float(w[mask].sum())
        mean_p = float(np.average(p[mask], weights=w[mask])) if mass else None
        mean_y = float(np.average(y[mask], weights=w[mask])) if mass else None
        if mass:
            ece += mass * abs(mean_p - mean_y)
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "weight": mass,
                "predicted": mean_p,
                "observed": mean_y,
            }
        )
    confidence = np.maximum(p, 1 - p)
    order = np.argsort(-confidence, kind="stable")
    curves = []
    for count in sorted(set(np.linspace(1, len(p), min(len(p), 100), dtype=int))):
        ids = order[:count]
        wrong = (p[ids] >= 0.5) != y[ids]
        returned = ids[p[ids] >= 0.5]
        curves.append(
            {
                "coverage": float(w[ids].sum()),
                "risk": float(np.average(wrong, weights=w[ids])),
                "precision": float(np.average(y[returned], weights=w[returned]))
                if len(returned)
                else None,
            }
        )
    return {
        "brier": float(np.sum(w * (p - y) ** 2)),
        "log_loss": float(
            -np.sum(w * (y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
        ),
        "ece": ece,
        "bins": rows,
        "risk_coverage": curves,
        "clip_epsilon": 1e-7,
        "predicted_prevalence": float(w @ p),
        "observed_prevalence": float(w @ y),
        "false_negative_rate": float(
            np.sum(w[(y == 1) & (p < 0.5)]) / np.sum(w[y == 1])
        )
        if np.any(y == 1)
        else None,
    }


def fit_logistic(pairs, evaluation_pairs):
    from sklearn.linear_model import LogisticRegression

    checked_pairs(pairs, {"train", "dev"})
    checked_pairs(evaluation_pairs, {"test"})
    train_groups = {(r["dataset"], r["query_id"]) for r in pairs}
    test_groups = {(r["dataset"], r["query_id"]) for r in evaluation_pairs}
    if train_groups & test_groups:
        raise ValueError("Calibration query groups overlap")
    if len({r["label"] for r in pairs}) < 2:
        raise ValueError("Logistic calibration requires both classes")

    def features(rows):
        p = np.clip([r["probability"] for r in rows], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    model = LogisticRegression(C=1.0, random_state=1729).fit(
        features(pairs),
        [r["label"] for r in pairs],
        sample_weight=[1 / r.get("inclusion_probability", 1) for r in pairs],
    )
    predictions = model.predict_proba(features(evaluation_pairs))[:, 1]
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "fit_groups": sorted(train_groups),
        "fit_hash": digest(pairs),
        "predictions": [
            {**row, "probability": float(p)}
            for row, p in zip(evaluation_pairs, predictions)
        ],
    }


def cluster_brier_interval(pairs, seed=1729, replicates=10000):
    checked_pairs(pairs)
    groups = {}
    for row in pairs:
        groups.setdefault((row["dataset"], row["query_id"]), []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        sample = [
            row
            for i in rng.integers(0, len(keys), len(keys))
            for row in groups[keys[i]]
        ]
        losses = [(r["probability"] - r["label"]) ** 2 for r in sample]
        means.append(
            np.average(
                losses, weights=[1 / r.get("inclusion_probability", 1) for r in sample]
            )
        )
    return np.quantile(means, [0.025, 0.975]).tolist()

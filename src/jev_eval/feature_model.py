"""Judged-only regularized feature comparator with explicit query holdout."""

import numpy as np


def fit_feature_model(training, validation, feature_names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not training or not validation or not feature_names:
        raise ValueError("Training and validation rows are required")

    def groups(rows):
        return {(r["dataset"], r["query_id"]) for r in rows}

    if groups(training) & groups(validation):
        raise ValueError("Feature-model validation must hold out complete queries")
    for row in training + validation:
        if (
            row.get("split") not in {"train", "dev"}
            or row.get("judgment_status") != "judged"
            or row.get("label") not in (0, 1)
        ):
            raise ValueError(
                "Only explicitly judged train/dev pairs can fit feature models"
            )

    def x(rows):
        values = np.array(
            [[r["features"][name] for name in feature_names] for r in rows], dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Missing/nonfinite feature values")
        return values

    if len({r["label"] for r in training}) < 2:
        raise ValueError("Feature model needs both classes")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, random_state=1729, max_iter=1000)
    )
    model.fit(x(training), [r["label"] for r in training])
    predictions = model.predict_proba(x(validation))[:, 1]
    return model, {
        "features": feature_names,
        "regularization_C": 1.0,
        "fit_queries": sorted(groups(training)),
        "validation_queries": sorted(groups(validation)),
        "validation_brier": float(
            np.mean((predictions - [r["label"] for r in validation]) ** 2)
        ),
    }

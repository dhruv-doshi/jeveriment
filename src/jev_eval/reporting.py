"""Render plots exclusively from saved measurements; missing inputs are skipped."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import read_json, write_json


def plots(run_path):
    path = Path(run_path)
    output = path / "plots"
    output.mkdir(parents=True, exist_ok=True)
    made = []
    if (path / "summary.json").exists():
        summary = read_json(path / "summary.json")
        for metric in ("ndcg@10", "recall@10", "mrr@10", "judged@10"):
            valid = {
                name: r[metric]
                for name, r in summary.items()
                if r.get(metric) is not None
            }
            if not valid:
                continue
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            ax.barh(list(valid), list(valid.values()))
            ax.set(xlabel=metric, xlim=(0, 1), title="Measured fixed-pool results")
            target = output / (metric.replace("@", "_at_") + ".png")
            fig.savefig(target, dpi=150)
            plt.close(fig)
            made.append(str(target))
    if (path / "calibration.json").exists():
        metrics = read_json(path / "calibration.json")["raw"]
        bins = [b for b in metrics["bins"] if b["count"]]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].plot(
            [b["predicted"] for b in bins], [b["observed"] for b in bins], "o-"
        )
        axes[0].set(
            xlabel="Predicted relevance",
            ylabel="Observed relevance",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1].bar([b["bin"] for b in bins], [b["count"] for b in bins])
        axes[1].set(xlabel="Equal-width probability bin", ylabel="Audited pairs")
        target = output / "reliability.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        made.append(str(target))
    write_json(
        output / "manifest.json",
        {
            "source_run": str(path),
            "files": made,
            "note": "Missing measured inputs are never replaced with illustrative results",
        },
    )
    return made

import argparse
import json

from .config import Settings, load_config


def main():
    parser = argparse.ArgumentParser(description="Jev fixed-pool retrieval evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env")
    check = sub.add_parser("validate-config")
    check.add_argument("config")
    sub.add_parser("preflight")
    for name in ("prepare", "retrieve", "rerank", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("config")
        if name == "prepare":
            command.add_argument(
                "--source",
                help="Existing BEIR directory; otherwise download core release",
            )
        if name == "rerank":
            command.add_argument(
                "--system", choices=["qwen", "bge", "jev"], default="qwen"
            )
            command.add_argument(
                "--resume",
                action="store_true",
                help="Validated checkpoints are always resumed",
            )
    paired = sub.add_parser("paired-test")
    paired.add_argument("input", help="JSON dataset -> query -> [baseline, challenger]")
    paired.add_argument("output")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("development_pairs")
    calibration.add_argument("test_pairs")
    calibration.add_argument("output")
    audit = sub.add_parser("export-audit")
    audit.add_argument("config")
    audit.add_argument("--queries", type=int, required=True)
    audit.add_argument("--pairs", type=int, default=20)
    plot = sub.add_parser("plot")
    plot.add_argument("run_directory")
    remote = sub.add_parser("import-fixed-pool")
    remote.add_argument("config")
    remote.add_argument("bundle")
    external = sub.add_parser("import-scores")
    external.add_argument("run_directory")
    external.add_argument("name")
    external.add_argument("artifact")
    rag = sub.add_parser("generate-rag")
    rag.add_argument("run_directory")
    rag.add_argument("generator_config")
    rag.add_argument("--systems", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "check-env":
        settings = Settings.load()
        print(
            json.dumps(
                {
                    "api_key": "set",
                    "max_cost_usd": str(settings.max_cost_usd),
                    "model": settings.model,
                    "base_url": settings.base_url,
                }
            )
        )
    elif args.command == "validate-config":
        print(load_config(args.config).model_dump_json(indent=2))
    elif args.command == "preflight":
        from .preflight import run_preflight

        report = run_preflight()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checks": len(report["checks"]),
                    "budget": report["budget"],
                }
            )
        )
    elif args.command in {"prepare", "retrieve", "rerank", "evaluate"}:
        from . import pipeline
        from .resources import ResourceMonitor

        config = load_config(args.config)
        phase = (
            f"{args.command}_{args.system}"
            if args.command == "rerank"
            else args.command
        )
        with ResourceMonitor(pipeline.run_dir(config) / f"resources_{phase}.json"):
            kwargs = (
                {"source": args.source}
                if args.command == "prepare"
                else ({"system": args.system} if args.command == "rerank" else {})
            )
            print(
                json.dumps(
                    getattr(pipeline, args.command)(args.config, **kwargs), default=str
                )
            )
    elif args.command == "paired-test":
        from .io import read_json, write_json
        from .stats import paired_comparison

        write_json(args.output, paired_comparison(read_json(args.input)))
    elif args.command == "calibrate":
        from .calibration import (
            calibration_metrics,
            cluster_brier_interval,
            fit_logistic,
        )
        from .io import read_json, write_json

        development, test = (
            read_json(args.development_pairs),
            read_json(args.test_pairs),
        )
        fit = fit_logistic(development, test)
        write_json(
            args.output,
            {
                "raw": calibration_metrics(test),
                "raw_brier_ci95": cluster_brier_interval(test),
                "recalibrated": calibration_metrics(fit["predictions"]),
                "fit": fit,
            },
        )
    elif args.command == "export-audit":
        from .audit import sample_audit
        from .io import read_json, write_json
        from .pipeline import load_run

        config, dest = load_run(args.config)
        blinded, manifest = sample_audit(
            config.dataset,
            read_json(dest / "pools.json"),
            read_json(dest / "queries.json"),
            read_json(dest / "text_views.json"),
            config.split,
            args.queries,
            args.pairs,
            config.seed,
        )
        write_json(dest / "audit_blinded.json", blinded)
        write_json(dest / "audit_private_mapping.json", manifest)
    elif args.command == "plot":
        from .reporting import plots

        print(json.dumps(plots(args.run_directory)))
    elif args.command == "import-fixed-pool":
        from .expansion import import_fixed_pool

        print(import_fixed_pool(args.config, args.bundle))
    elif args.command == "import-scores":
        from .expansion import import_external_scores

        import_external_scores(args.run_directory, args.name, args.artifact)
    elif args.command == "generate-rag":
        from .generation import generate_rag
        from .io import read_json

        print(
            json.dumps(
                generate_rag(
                    args.run_directory, read_json(args.generator_config), args.systems
                )
            )
        )

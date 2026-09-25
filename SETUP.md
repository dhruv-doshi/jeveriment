# Direct TypeSafe setup

This worktree calls TypeSafe directly at `https://api.typesafe.ai/v1/systemone`.
The existing Vercel run in the main checkout keeps its own credentials, ledger,
and scan checkpoint.

Fill in this worktree's `.env`:

- `TYPESAFE_API_KEY`: a direct API key from the TypeSafe console. Keep it in `.env`.
- `JEV_MODEL`: use the pinned `jev-1.13.0` unless you intentionally start a new
  model segment. TypeSafe's `jev-latest` alias can move.
- `JEV_MAX_COST_USD`: local direct API allowance. Enter the amount you authorize
  before live calls. This does not configure an account limit.
- `JEV_INPUT_USD_PER_MILLION_TOKENS`: the published direct input price, currently
  `0.042`. Recheck the TypeSafe model page before a large scan.

Then run:

```bash
./scripts/01_setup.sh
.venv/bin/python -m jev_eval preflight
./scripts/06_jev_scan.sh 1 5183 8000 30000000
```

The direct preflight writes `manifests/capabilities_typesafe.json`. The scan uses
the already prepared SciFact data under `runs/scifact_train_pilot_v1/`, but writes
its own `jev_scan_typesafe_scores.json`, raw responses under `typesafe_scan/`, and
`runs/budget_typesafe.sqlite`. The Vercel scores cannot be resumed as direct
scores because the provider identity and resolved model version were not proven
equivalent. The scan defaults to 12 workers and 10 request starts per second,
adapting downward on 429 or 529 responses.

TypeSafe currently publishes a ceiling of 1,200 requests per minute and 250,000
tokens per second for Jev 1.13. It says those limits can change. The default
scan rate is below the published request ceiling, but your account may receive
different limits, so inspect 429 responses and the scan's rate adjustments.

For fixed-pool Jev reranking or the decision benchmark through TypeSafe, use a
new experiment ID so those outputs cannot be confused with the completed Vercel
pilot. The direct scan's own output filenames already isolate its results.

References: [TypeSafe API](https://docs.typesafe.ai/api),
[TypeSafe models and limits](https://docs.typesafe.ai/models).

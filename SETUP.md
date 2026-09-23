# Environment setup — first step

Status: core implementation and testing are documented in `README.md`. Synthetic
live capability checks have run. Benchmark experiments remain unrun at the user's
explicit request. Follow `Jev_RAG_Retrieval_Final_Guide.md`, section 11, in phase order.

Fill in `.env` locally:

- `AI_GATEWAY_API_KEY`: create an AI Gateway key in your Vercel team's dashboard.
- `JEV_MAX_COST_USD`: your total USD allowance for the initial capability preflight
  and SciFact training pilot, including retries. Set `0` for verified free-only
  execution. Blank must block live execution. This project setting will be enforced
  by the harness before live requests; creating this file does not configure a
  Vercel account budget. A matching Vercel key budget provides provider-side control.
- `HF_TOKEN`: optional read token for Hugging Face; public pilot models do not
  require it. Gated extensions may require both a token and access approval later.

The endpoint and model are already configured. No direct TypeSafe key, Vercel
deployment/project ID, database credentials, or generator API key is needed for
the initial retrieval study. Models and indexes run locally. Optional API
comparators and the phase 8 generator will be configured when those models are
selected.

Vercel integration uses `POST /typesafe/v1/systemone` with bearer authentication
and model `typesafe-ai/jev`, preserving the guide's Noul/Score/Choice contracts.
The base URL excludes `/v1/systemone`; the adapter will append that path.
Vercel recommends its generic evaluation API for new applications; the compatible
API is chosen here to preserve the guide's documented schemas. Actual capability,
usage, pricing, rate limits, and model revision behavior remain preflight gates.

As checked on 2026-09-22, Vercel lists promotional free pricing ending September 25,
2026. Recheck pricing before live execution; do not infer permanent free access.
The pilot contains 1,500 pair evaluations plus separately recorded preflight,
retries, and repeats. Token/request limits will be frozen after preflight estimates.

Next: phase 0 repository/contracts and capabilities, then phase 1 evaluation,
phase 2 retrieval, phase 3 Jev/local rerankers and pilot, phase 4 selection,
phase 5 fusion, phase 6 calibration/statistics, and phases 7–8 extensions.
Human calibration judgments and multilingual review must come from actual
assessors; implementation alone cannot satisfy these experimental gates.

References:

- [Gateway API keys](https://vercel.com/docs/ai-gateway/authentication-and-byok/api-keys)
- [TypeSafe-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe)
- [Jev model and pricing](https://vercel.com/ai-gateway/models/jev)

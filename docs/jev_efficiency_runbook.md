# Finish the Jev article experiments

Keep the droplet until the efficiency comparison finishes and the new results are downloaded and checked. The existing SciFact and NFCorpus data is already available locally. Offline depth replay and hosted Jev calls work on the Mac; the droplet preserves the deployment used for the original Qwen/BGE runs. The runner records its hardware and measures freshly warmed local models. Deleting it now would lose that environment and require replacement infrastructure for this comparison.

## Completed locally

- `output/jev_article_analysis_2026-10-02/report.md`: frozen independent scores replayed at candidate depths 10, 25, 50, and 100; quality tables, paired uncertainty, archived API usage, and plots.
- `output/jev_efficiency_v1/manifest.json`: 30 queries randomly selected per dataset using seed 1729; identical frozen candidate IDs and text; depths 25 and 100; randomized condition order.
- `output/jev_efficiency_smoke_2026-10-02/`: one query per dataset at depth 10. All three Jev arms completed. This verifies the runner, rather than providing article-scale latency evidence. Estimated API cost including warmups: $0.001977612.
- Validation: 45 tests passed; Ruff checks passed for the new analysis, runner, wrappers, and tests.

FiQA is excluded. These are exploratory comparisons, with one measured pass per condition. Thirty query observations give limited p95 precision and cannot establish reliable p99 behavior.

## Upload the prepared bundle from your Mac

The bundle contains Python source and the frozen benchmark manifest. It contains no `.env`, API credentials, original result files, or model weights. The original repository files on the VM are not overwritten.

```bash
cd ~/coding/jeveriment
scp output/jev_efficiency_runner_v1.tar.gz root@139.59.45.57:/root/jeveriment/
```

Enter your SSH password when prompted. Direct automated SSH access was unavailable because the server requires authentication not provided to the agent.

## Run on the VM inside tmux

```bash
cd ~/jeveriment
mkdir -p benchmarks
tar -xzf jev_efficiency_runner_v1.tar.gz -C benchmarks
tmux new -s efficiency
```

Inside that tmux session:

```bash
cd ~/jeveriment
set -o pipefail
PYTHONPATH="$PWD/benchmarks/efficiency_runner" \
  .venv/bin/python benchmarks/efficiency_runner/benchmark_efficiency.py run \
  --input-root . \
  --output benchmarks/jev_efficiency_v1 \
  --max-cost-usd 5 \
  2>&1 | tee -a benchmarks/jev_efficiency_v1/run.log
```

The VM's existing `.env` supplies the API key. The benchmark has its own persistent ledger and caps estimated API cost plus uncertain reservations at the smaller of $5 and `JEV_MAX_COST_USD`. This cap excludes VM rental. It also limits attempts to 25,000 and accounted tokens to 100 million. Initial scoring workload is 16,500 Jev requests before retries and warmups.

The runner checks hashes of the archived inputs before sending requests. Each dataset/depth condition measures Qwen and BGE with the original pinned revisions and batch size, Jev sequentially, Jev with four workers sharing a five-request-per-second limiter, and Jev with five independently targeted questions per packed request. Packed text is inside each question's instructions, so its layout differs from the original independent request and its quality is measured again.

Query timers include document scoring, worker scheduling, pacing, and retries. They exclude model loading, connection/model warmup, retrieval, and generation. Queries run one at a time; throughput therefore describes this workload rather than peak multi-query serving capacity. Local model scores are brought back to CPU before the timer stops. Local precision and actual device, including fallback to CPU, are recorded.

Detach with **Ctrl-b, then d**. Reattach using `tmux attach -t efficiency`. Based on the original local inference runtimes, allow several hours. If interrupted, repeat the same run command: completed query results are skipped; incomplete queries receive fresh requests. Retained raw attempts and the budget ledger cover interrupted work too.

Results appear under `benchmarks/jev_efficiency_v1/`: hardware, pinned model metadata, per-query scores/timers/usage, raw API responses, ledger, summary JSON/CSV, and paired quality comparisons. Successful-only latency/quality is explicitly labelled; failed and missing queries are counted. Query costs exclude warmups and earlier abandoned trials; the full ledger retains those costs. Cost estimates use the configured input-token price, not an invoice. Qwen/BGE dollar cost needs the droplet rental rate and an explicit allocation of VM time.

## Download and verify before deleting the droplet

After the run exits, run these on your **Mac**:

```bash
cd ~/coding/jeveriment
rsync -avz --progress \
  root@139.59.45.57:/root/jeveriment/benchmarks/jev_efficiency_v1/ \
  output/jev_efficiency_vm_v1/

rsync -ainc \
  root@139.59.45.57:/root/jeveriment/benchmarks/jev_efficiency_v1/ \
  output/jev_efficiency_vm_v1/
```

The second command is a checksum dry run; no listed changes means the copied files match the server at that time. Check completion locally:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('output/jev_efficiency_vm_v1')
rows = json.loads((root / 'summary.json').read_text())
assert len(rows) == 20, 'Missing conditions'
assert all(r['completed'] == 30 and r['failed'] == 0 and r['missing'] == 0 for r in rows), 'Incomplete benchmark'
assert (root / 'hardware.json').exists()
assert (root / 'budget_typesafe.sqlite').exists()
print('All 20 conditions complete and downloaded.')
PY
```

If the check fails, inspect `run.log` and failed query JSON before rerunning. Keep `output/server_export_2026-10-02/` alongside this new download. The original export excluded `.env`, virtual environments, caches, and Git metadata; preserve any VM-only credentials or other work you need separately. Once the benchmark and needed files are verified locally, the droplet is no longer needed for this article analysis.

## Reproduce the offline analysis

```bash
.venv/bin/python scripts/analyze_tradeoffs.py \
  --input-root output/server_export_2026-10-02 \
  --output output/jev_article_analysis_2026-10-02
```

The replay makes no API calls. It keeps corpus-wide relevance denominators, gives every system the same RRF prefix at each depth, and rejects missing scores or changed frozen document text. The report distinguishes API attempt latency from full-query latency and saved final-invocation timing from the entire resumed SciFact run.

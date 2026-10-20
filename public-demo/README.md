# Public demo

Source for <https://lltm-memory.pages.dev>. Deploy **only `public-demo/site`**.
The seven files in that folder are the complete public surface; documentation,
Python tools, credentials, datasets and stores stay outside it.

The live LLM playground was enabled on 2026-10-20 UTC. Its
[deployment record](deployments/20261020-live-memory.json) includes the published
file hashes and the scope of the real-provider and resource checks.

## Live Memory Playground

The page sends raw natural language to `POST /demo/messages`, which uses the product's
`MemoryService` / `TwoStageExtractor`. Questions go to `POST /demo/answer`, using the
product `t4` profile: structured memory, selective raw turns, evidence hydration and
conditional recovery. Presets contain utterances only, not precomputed facts or answers.

The Inspector shows extracted facts and their validity intervals, source spans for
selected memories, and the actual answer-request prompts (including a fallback pass).
These spans are source references, not independently verified citations. The benchmark
section describes separate recorded experiments, not a score for the live visitor data.

Enable on the **backend** only:

```bash
export LLTM_DEMO_LLM_ENABLED=1
export GEMINI_API_KEY=your-server-side-key
.venv/bin/python -m uvicorn app:app --app-dir demo-api --host 127.0.0.1 --port 8100
```

Install the project extras locally, or use the separately locked
`demo-api/requirements.txt` on the deployment. The live mode uses the existing ONNX
encoder when present; verify it with `demo-api/fetch_encoder.py` before serving it.
ONNX is checked against the reference embeddings and uses a separate demo store;
it is not mixed into a benchmark index.

With the flag absent, the UI explicitly reports that LLM service is not enabled.
Provider errors, exhausted capacity and timeouts never substitute rule-generated
answers. Legacy `/demo/facts` and retrieval routes remain for engine smoke tests.
No model calls occur during page load, health checks or session creation.

### Identity, data and capacity

The server issues a signed anonymous namespace token. New conversations retain that
namespace but do not submit the old browser transcript. Presets explicitly delete the
current namespace and create a new one. Each namespace expires after 60 minutes;
expiry sweeps and explicit deletion remove memories, turns, replay responses and vectors.
Counts without user text remain for admission control. Prompts captured for the
Inspector are cleared from backend memory after each operation.

| Backend setting | Default | Meaning |
|---|---:|---|
| `LLTM_DEMO_SESSION_CALLS` | 32 | Reserved attempt capacity per namespace per UTC day |
| `LLTM_DEMO_SESSION_TOKENS` | 300,000 | Reserved token capacity per namespace per UTC day |
| `LLTM_DEMO_DAILY_CALLS` | 100 | Reserved attempt capacity across all namespaces per UTC day |
| `LLTM_DEMO_DAILY_TOKENS` | 1,000,000 | Reserved token capacity across all namespaces per UTC day |

Reservations happen before each model step: two provider attempts total, shared by
one backoff retry for transient API errors and any compatibility retry. Failed and
unused reservations are not refunded. SDK and transport
retries are disabled, calls time out at 30 seconds, and no new step starts after the
operation's 60-second admission window. The client exposes recorded usage separately;
reservations are conservative capacity bounds, not a monetary billing meter.

The capacity ledger is SQLite-backed, survives process restarts while the store is
retained, and is not cleared by visitor deletion. Render's free ephemeral disk can
lose it on redeploy: use retained storage and a separate model project for a public
LLM deployment whose budget must survive store loss. The ONNX + SDK runtime still
needs memory verification on the chosen host; no production deployment is implied.

### Verification

```bash
.venv/bin/pytest tests/test_live_playground.py
.venv/bin/python demo-api/check_imports.py
.venv/bin/python demo-api/smoke.py
node --check public-demo/site/app.js
.venv/bin/python public-demo/check_site.py
```

The route tests use scripted model transport with real extraction, deduplication,
temporal resolution and answering code. The isolated dependency probe constructs
both live paths without contacting a model; neither is a real-provider quality test.

## Results and evidence

`release.json` contains two separately labelled comparisons:

- `regression`: the latest audited, exposed original-test100 regression, three
  repeats per arm, including recorded answer and grading token usage.
- `result`: the unchanged historical frozen v2 one-shot comparison.

`release.schema.json` defines both. The optional regression section extends schema
2 without changing the historical `result` or the engineering-count definitions.
The page renders values from this manifest in English and Chinese.

Refresh regression figures from the gate and completion audit, then verify them:

```bash
.venv/bin/python public-demo/update_regression_figures.py
.venv/bin/python public-demo/update_regression_figures.py --check
.venv/bin/python public-demo/check_site.py
```

The generator verifies the completion audit and its input hashes before writing.
It does not run a model or change experiment results. To refresh engineering
counts and coverage, run `public-demo/check_release_figures.py --update`.

## Local preview

From the repository root, run these in separate shells:

```bash
.venv/bin/python -m uvicorn app:app --app-dir demo-api --host 127.0.0.1 --port 8100
```

```bash
.venv/bin/python -m http.server 8779 --bind 127.0.0.1 --directory public-demo/site
```

Open <http://localhost:8779>. Localhost selects port 8100 for the API; opening the
HTML through `file://` does not support its module fetch. The demo API must expose
the `conversations` capability for the new-conversation button to appear.

## Deployment

```bash
npx wrangler@4 pages deploy public-demo/site --project-name lltm-memory --branch main --commit-dirty=true
```

Authenticate separately with `npx wrangler@4 login` when needed. Verify that links
to GitHub evidence already exist on `origin/main` before deploying. The upload
must contain exactly the seven files checked by `check_site.py`.

The repository's Inspector is a separate authenticated tool, not this public demo.

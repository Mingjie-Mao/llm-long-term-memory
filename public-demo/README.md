# Public demo

Source for <https://lltm-memory.pages.dev>. Deploy **only `public-demo/site`**.
The seven files in that folder are the complete public surface; documentation,
Python tools, credentials, datasets and stores stay outside it.

## What it shows

The worked example and the interactive box call `demo-api` in separate user
namespaces. Facts are created by sentence patterns; answers come from memory
retrieval. The page does not run LLM extraction or generation. A new conversation
keeps the same user namespace and creates a new source session.

Demo access expires after 60 minutes. Explicit deletion removes the namespace;
expiry is persisted and swept by the backend. This is not a production retention
policy. The page loads no third-party fonts or analytics.

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

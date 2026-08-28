# public-demo

The public demonstration page, deployed at **<https://lltm-memory.pages.dev>**.

```
public-demo/
├── README.md       this file — notes, never deployed
├── check_site.py   stdlib-only CI check, never deployed
└── site/           the complete deploy root
    ├── index.html     semantic page structure
    ├── styles.css    responsive visual system
    ├── app.js        browser and live-engine interaction
    ├── content.js    bilingual fictional scenarios
    ├── release.json measured figures shown on the page
    └── playground.html legacy redirect
```

The split is not tidiness. The first version of this directory kept the README beside
the page, so `wrangler` uploaded both and
`https://lltm-memory.pages.dev/README.md` served these notes — including the sentence
about the repository root holding `.env` and the `stores/` databases — to anyone who
guessed the path. Deploying a directory publishes *everything* in it, so the deploy
root now contains nothing but the site.

The site has no framework, package install or build step. Splitting structure, style,
interaction and content keeps a deploy diff reviewable without adding a JavaScript
toolchain. It makes two deliberately different promises:

- the guided tour runs entirely in the browser over curated fictional scenarios and
  transmits nothing;
- the live engine sends only the visitor's chosen scenario or developer-mode input to
  `demo-api`, under a server-issued namespace that is hard-deleted after 60 minutes.

The page loads no third-party font or analytics. Its only non-static network traffic is
to the project's own public demo API. Measured figures come from `release.json`, the one
site-facing release manifest; explanatory links point to the complete evidence rather
than duplicating another table in the page.

It is bilingual: one toggle swaps static copy and every guided/live scenario view via
`data-zh` / `data-en` attributes and parallel `zh` / `en` data in `content.js`. The
choice is remembered in `localStorage` and defaults from `navigator.language`.

Run the zero-dependency contract check before deploying:

```bash
python3 public-demo/check_site.py
```

## Run it locally

`app.js` treats `localhost` and `127.0.0.1` as development and calls the demo API on
port 8100 instead of the deployed service, so both halves have to be up. From the
repository root, in two shells:

```bash
.venv/bin/python -m uvicorn app:app --app-dir demo-api --host 127.0.0.1 --port 8100
```

```bash
.venv/bin/python -m http.server 8779 --bind 127.0.0.1 --directory public-demo/site
```

Then open <http://localhost:8779>. Port 8779 is not arbitrary: it is the local origin
`render.yaml` allows in `LLTM_DEMO_ORIGINS`, so the same page can also be pointed at the
deployed API without a CORS failure. Opening `index.html` from the filesystem does not
work: the page is an ES module, so `file://` fails the module fetch.

## Why this exists separately from the inspector

The inspector (`src/llm_long_term_memory/api/static/index.html`) is a *tool*: it
shows the namespace you ask it for and nothing before that. It used to double as the
demo, with four `?demo=` links hard-coding LongMemEval namespaces, which meant
opening it put a synthetic persona's private-looking history on screen unasked. See
the 2026-08-27 note in [`docs/ROADMAP.md`](../docs/ROADMAP.md).

So the demonstration lives here instead, on openly fictional data, and the inspector
went back to being a tool.

## Redeploy

From the repository root. `npx` fetches wrangler on demand, so this Python project
carries no Node dependency of its own:

```bash
npx wrangler@4 pages deploy public-demo/site --project-name lltm-memory --branch main --commit-dirty=true
```

Authentication is a one-off and interactive — `npx wrangler@4 login` opens a browser
and stores an OAuth token in `~/Library/Preferences/.wrangler`. Nothing about it
lives in this repository.

**Deploy `public-demo/site`, never `public-demo` and never the repository root.** The
root holds `.env`, the `stores/` databases and a 277 MB dataset, none of which belongs
on a public CDN.

**Every `blob/main/` link on the page must already exist on `main`.** The page links to
`docs/CURRENT_STATUS.md`, `docs/REPORT.md`, `results/data-protocol.md` and
`results/heldout-variance.md`. `check_site.py` verifies local assets but cannot verify
GitHub URLs, so deploying the page ahead of the branch that adds a linked document
publishes a 404. Confirm with:

```bash
for f in docs/CURRENT_STATUS.md docs/REPORT.md results/data-protocol.md results/heldout-variance.md; do git cat-file -e origin/main:$f || echo "MISSING ON MAIN: $f"; done
```

**Check the `Uploaded N files` line every time.** A correct deploy contains exactly
the six files asserted by `check_site.py`. Any other file means the public surface
changed and must be reviewed before deployment.

## Not covered by the experiment freeze

`v2-candidate-preingest` hashes `src/**/*.py`, `src/**/*.sql`, `scripts/*.py`,
`pyproject.toml` and `uv.lock`. Nothing in this directory is in that set, so editing
the page cannot invalidate the `dev100` ingest lineage. Adding a `.py` file to
`scripts/` would — put tooling for this page here instead.

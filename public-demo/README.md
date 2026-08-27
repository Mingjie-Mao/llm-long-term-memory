# public-demo

The public demonstration page, deployed at **<https://lltm-memory.pages.dev>**.

```
public-demo/
├── README.md      this file — notes, never deployed
└── site/          the deploy root: only what should be public
    └── index.html
```

The split is not tidiness. The first version of this directory kept the README beside
the page, so `wrangler` uploaded both and
`https://lltm-memory.pages.dev/README.md` served these notes — including the sentence
about the repository root holding `.env` and the `stores/` databases — to anyone who
guessed the path. Deploying a directory publishes *everything* in it, so the deploy
root now contains nothing but the site.

`site/index.html` is one self-contained file: no build step, no framework, no network
requests except Google Fonts. Every memory, conversation and retrieval result in it
is **fictional data written for the demo** — no evaluation fixture and no real store
content appears here, and the page reads nothing from the visitor. The only measured
figures on it are the benchmark numbers in the "Measured" table, which come from
`docs/REPORT.md`.

It is bilingual: one toggle swaps every string, including the demo data, via
`data-zh` / `data-en` attributes and a parallel `CASES` table in the script. The
choice is remembered in `localStorage` and defaults from `navigator.language`.

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

**Check the `Uploaded N files` line every time.** A correct deploy is exactly one
file. Any other number means something joined the deploy root that should not have.

## Not covered by the experiment freeze

`v2-candidate-preingest` hashes `src/**/*.py`, `src/**/*.sql`, `scripts/*.py`,
`pyproject.toml` and `uv.lock`. Nothing in this directory is in that set, so editing
the page cannot invalidate the `dev100` ingest lineage. Adding a `.py` file to
`scripts/` would — put tooling for this page here instead.

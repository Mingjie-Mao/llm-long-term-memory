# Deploying

Two things get deployed and they are not the same thing.

| | what it is | where | holds data? |
|---|---|---|---|
| **the page** | a static demo, no backend of its own | Cloudflare Pages | no |
| **the playground API** | the LLM-free half: retrieval and temporal updates, no extraction | Render, free plan | no — `/tmp`, wiped on restart by design |
| **the memory service** | the whole system: extraction, writes, budgets, erasure | not yet deployed | **yes** — needs a disk |

The first two are live. The third is configured and has never been deployed, because it
holds real data and that changes what it needs.

---

## B2 · Redeploy the page

The live page was built from `5a456a78` on 2026-09-13 and the repository has moved since.

```bash
# 1. Check the deploy root is complete and self-consistent. Stdlib only, no install.
python3 public-demo/check_site.py

# 2. Publish. `wrangler` uploads the directory you name, so name the deploy root and
#    nothing above it — an earlier deploy published the notes file by pointing one
#    level too high.
npx wrangler pages deploy public-demo/site --project-name lltm-memory --branch main
```

`wrangler` will open a browser to authenticate the first time. The account id and
project name land in `.wrangler/`, which is gitignored: an account id is not a
credential but is not something to publish either.

Then record what went out, so a page in the wild can always be traced to a commit:

```bash
python3 -c "
import hashlib, json, subprocess, datetime, pathlib
root = pathlib.Path('public-demo/site')
print(json.dumps({
  'schema_version': 1,
  'deployed_at_utc': datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
  'provider': 'cloudflare-pages', 'project': 'lltm-memory', 'branch': 'main',
  'git_head': subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,encoding='utf-8').stdout.strip(),
  'dirty_worktree': bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,encoding='utf-8').stdout.strip()),
  'deploy_root': 'public-demo/site',
  'files': sorted(({'path': str(p.relative_to(root)),
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in root.rglob('*') if p.is_file()), key=lambda f: f['path']),
}, indent=1))" > public-demo/deployments/$(date -u +%Y%m%dT%H%M%SZ).json
```

Fill in `deployment_id` and `deployment_url` from what `wrangler` prints, then commit
the record.

**Check afterwards.** The page reads `release.json` for its figures, so confirm the
live one matches the repository — a stale page showing an old test count is the
"old result impersonating a current one" failure this project avoids elsewhere:

```bash
curl -s https://lltm-memory.pages.dev/release.json | python3 -m json.tool | head -20
```

---

## B4 · Deploy the memory service

`render.yaml` now carries a second service, `lltm-memory`. It is **not** a matter of
pushing: three things have to be true first, and two of them cost money.

### Before you start

**1. It cannot run on the free plan.** The encoder is `sentence-transformers`, so the
image carries CPU torch. The Dockerfile already strips the CUDA stack that `uv`
resolves by default — nvidia/ at 2.9GB and triton/ at 649MB, measured inside the image
— and what remains still does not fit 512MB. The blueprint asks for `standard`; check
Render's current plans and pick one with at least 2GB.

*The alternative is a code change, not a configuration one.* The playground fits the
free plan because it runs ONNX (162MB against 565MB of wheels), but that encoder lives
in `demo-api/onnx_encoder.py` and the engine package has no path to it. Giving
`llm_long_term_memory.embed.Encoder` an ONNX backend would let the service run free —
and it is a real candidate, not a workaround, since nothing in the read path needs torch.

**2. It needs a disk.** A memory service on ephemeral storage forgets everything on
every restart. The blueprint mounts 1GB at `/data`, which is where the image already
points `LLTM_STORE_DIR`. Disks are billed separately from the instance.

**3. It must not come up in open mode.** `LLTM_REQUIRE_AUTH=1` is set in the blueprint,
which means the process **refuses to start** unless `LLTM_API_TOKENS` configures at
least one token. That is the intent: forgetting the tokens otherwise produces a service
that works perfectly and authorises everyone.

### Environment variables

Set in the Render dashboard, never in the repository.

| key | value | why it is not in the repo |
|---|---|---|
| `LLTM_API_TOKENS` | `<token>:<tenant>[,<token>:<tenant>…]` | a credential. Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `GEMINI_API_KEY` | your provider key | a credential. Without it the read path still serves and `POST /v1/messages` answers 503 |
| `LLTM_REQUIRE_AUTH` | `1` | already in the blueprint; listed so nobody removes it to "fix" a startup failure |
| `LLTM_STORE_DIR` | `/data/stores` | already in the blueprint, and must agree with the disk mount |
| `LLTM_RESULTS_DIR` | `/data/results` | same |

A malformed `LLTM_API_TOKENS` is refused rather than downgraded to open access, and an
unparseable `LLTM_REQUIRE_AUTH` is an error rather than False — `LLTM_REQUIRE_AUTH=ture`
silently disabling the guard is the accident the guard exists to prevent.

### Deploying

1. In Render, **New → Blueprint**, point it at this repository. It will offer both
   services; `lltm-playground` already exists, so apply only `lltm-memory`.
2. Set the two secrets above before the first build. The build will succeed without
   them and the service will then refuse to start, which is the designed behaviour and
   looks like a failure if you are not expecting it.
3. First build is slow: it resolves the dependency tree, reinstalls CPU torch from
   PyTorch's own index and deletes the CUDA orphans.

### Checking it came up correctly

```bash
BASE=https://lltm-memory.onrender.com          # whatever Render assigns
TOKEN=<one of the tokens you configured>

# Ready, and saying that it is secured rather than open.
curl -s $BASE/healthz | python3 -m json.tool
#   expect: "authenticated_access": true, "authentication_required": true

# No credential is refused.
curl -s -o /dev/null -w '%{http_code}\n' -X POST $BASE/v1/memories/search \
  -H 'content-type: application/json' -d '{"query":"anything"}'
#   expect: 401

# A credential works, and names its own tenant rather than one the caller asked for.
curl -s -X POST $BASE/v1/memories/search -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"anything"}' | python3 -m json.tool

# Asking for someone else's namespace is refused rather than quietly redirected.
curl -s -o /dev/null -w '%{http_code}\n' -X POST $BASE/v1/memories/search \
  -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"query":"anything","user_id":"someone-else"}'
#   expect: 403
```

If `/healthz` returns 503 with `open mode` in the detail, `LLTM_API_TOKENS` is unset or
malformed. That is the fail-closed switch working, not a bug.

### Once it is up

Schedule the operations run. Render offers cron jobs; anything that can call a command
on a timer will do.

```bash
python3 tools/operate.py \
  --store /data/stores/live.db \
  --backups /data/backups \
  --journal-copy <somewhere on different storage> \
  --keep 7 --alert-at 0.8 --log /data/ops.jsonl
```

`--journal-copy` pointed inside `/data` is better than nothing and is **not** what it is
for: the erasure journal exists so a restore can replay deletions served after the
backup was taken, and a copy on the same disk is lost with the disk. Point it at object
storage or another volume.

Rehearse a restore before you need one:

```bash
python3 tools/backup_restore.py verify --backup <a backup folder> --into /tmp/drill \
  --erasures /data/stores/live.erasures.jsonl
```

### What is still missing before real users

- No scheduled restore drill; `verify` exists and nothing runs it on a timer.
- The store is SQLite with one writer. Concurrent writes are serialised inside one
  process and there is no story for two.
- No rate limiting in front of the service; budgets cap spend per account per day, which
  is not the same thing.

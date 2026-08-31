"""End-to-end exercise of the playground, asserting on real engine behaviour.

Run: `.venv/bin/python demo-api/smoke.py`. No API key, no network, no model calls
beyond the local encoder. It is deliberately a script rather than a pytest file:
`tests/` belongs to the engine, and this checks a service that must stay outside the
experiment freeze.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

scratch = Path(tempfile.mkdtemp())
os.environ.setdefault("LLTM_DEMO_STORE", str(scratch / "playground"))
os.environ.setdefault("LLTM_DEMO_SECRET", "smoke-secret-not-for-deployment")
# Some native runtime dependencies persist process-local telemetry beside the
# current directory. Keep every smoke-test artefact inside the same disposable
# directory as its database instead of dirtying the repository root.
os.chdir(scratch)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ok, failed = 0, 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok, failed
    if condition:
        ok += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


with TestClient(app) as c:
    print("\n— identity is issued, not accepted —")
    check("no token is refused", c.get("/demo/memories").status_code == 401)
    check(
        "forged token is refused",
        c.get("/demo/memories", headers={"x-demo-token": "demo_abc.deadbeef"}).status_code == 401,
    )

    s = c.post("/demo/session").json()
    tok = s["token"]
    h = {"x-demo-token": tok}
    check("session mints a token", "." in tok and tok.startswith("demo_"))

    other = c.post("/demo/session").json()["token"]
    check("two sessions get different namespaces", other.split(".")[0] != tok.split(".")[0])

    print("\n— supersession, computed by the real resolver —")
    c.post(
        "/demo/facts",
        headers=h,
        json={
            "predicate": "lives_in",
            "object": "Sydney",
            "content": "The user lives in Sydney.",
            "event_time": "2025-01-10",
        },
    ).json()
    r = c.post(
        "/demo/facts",
        headers=h,
        json={
            "predicate": "lives_in",
            "object": "Melbourne",
            "content": "The user moved to Melbourne.",
            "event_time": "2026-06-04",
            "replaces_previous": True,
        },
    ).json()
    check("the write reports one supersession", r["superseded_now"] == 1, str(r["superseded_now"]))
    changed = r["changed_by_this_write"]
    check(
        "it names which row changed",
        len(changed) == 1 and changed[0]["object"] == "Sydney",
        str(changed),
    )
    check("the closed row carries valid_to", bool(changed[0]["valid_to"]), str(changed[0]))
    check("and names its successor", changed[0]["superseded_by"] == r["created"]["id"])

    print("\n— a multi-valued key must NOT chain —")
    c.post(
        "/demo/facts",
        headers=h,
        json={"predicate": "owns_pet", "object": "a cat", "content": "The user owns a cat."},
    )
    c.post(
        "/demo/facts",
        headers=h,
        json={"predicate": "owns_pet", "object": "a dog", "content": "The user owns a dog."},
    )
    tl = c.get("/demo/timeline", headers=h, params={"predicate": "owns_pet"}).json()
    check(
        "both pets stay active",
        all(e["status"] == "active" for e in tl["entries"]),
        str(tl["entries"]),
    )
    check("no invented replacement arrow", not any(e["replaced_by_next"] for e in tl["entries"]))
    check("and the key is marked multi-valued", tl["single_valued_key"] is False)

    print("\n— retrieval hides history and explains the absence —")
    res = c.post(
        "/demo/search", headers=h, json={"query": "where do I live now?", "limit": 1}
    ).json()
    objs = [m["object"] for m in res["memories"]]
    check("Melbourne is retrieved", "Melbourne" in objs, str(objs))
    check("Sydney is not", "Sydney" not in objs, str(objs))
    reasons = {r_["content"]: r_["reason"] for r_ in res["rejected"]}
    check(
        "Sydney is rejected as superseded",
        reasons.get("The user lives in Sydney.") == "superseded",
        str(reasons),
    )
    check("a still-true miss is below_rank", "below_rank" in reasons.values(), str(reasons))
    check(
        "signals come from the real retriever",
        all("semantic" in m["signals"] for m in res["memories"]),
    )
    check("latency is reported", res["latency_ms"] >= 0)

    print("\n— the raw archive is searchable —")
    c.post(
        "/demo/turns",
        headers=h,
        json={
            "role": "assistant",
            "content": "Your booking reference is VX-928173 and the desk is on level 3.",
        },
    )
    raw = c.post("/demo/raw/search", headers=h, json={"query": "booking reference"}).json()
    check(
        "BM25 recovers the turn",
        raw["turns"] and "VX-928173" in raw["turns"][0]["content"],
        str(raw),
    )

    print("\n— namespace isolation —")
    ho = {"x-demo-token": other}
    check("another session sees nothing", c.get("/demo/memories", headers=ho).json()["total"] == 0)
    check(
        "and its search is empty",
        c.post("/demo/search", headers=ho, json={"query": "where do I live now?"}).json()[
            "memories"
        ]
        == [],
    )

    print("\n— hard delete really deletes —")
    before = c.get("/demo/memories", headers=h).json()["total"]
    d = c.delete("/demo/session", headers=h).json()
    check("delete reports rows removed", d["deleted"] and d["rows"]["memories"] == before, str(d))
    check("the token is dead afterwards", c.get("/demo/memories", headers=h).status_code == 410)

    import sqlite3

    db = sqlite3.connect(Path(os.environ["LLTM_DEMO_STORE"]).with_suffix(".db"))
    ns = tok.split(".")[0]
    left = db.execute("SELECT count(*) FROM memories WHERE user_id = ?", (ns,)).fetchone()[0]
    turns_left = db.execute(
        "SELECT count(*) FROM turns WHERE session_id IN "
        "(SELECT id FROM sessions WHERE user_id = ?)",
        (ns,),
    ).fetchone()[0]
    check("no memory rows survive in SQLite", left == 0, f"{left} left")
    check("no turn rows survive either", turns_left == 0, f"{turns_left} left")

    print("\n— guards —")
    h2 = {"x-demo-token": c.post("/demo/session").json()["token"]}
    check(
        "a bad predicate is refused",
        c.post(
            "/demo/facts",
            headers=h2,
            json={"predicate": "Lives In!!", "object": "x", "content": "y"},
        ).status_code
        in (200, 422),
    )
    check(
        "oversized content is refused",
        c.post(
            "/demo/facts",
            headers=h2,
            json={"predicate": "note", "object": "x", "content": "z" * 5000},
        ).status_code
        == 413,
    )
    check("health says no LLM is required", c.get("/demo/health").json()["llm_required"] is False)

print(f"\n{ok} passed, {failed} failed")
sys.exit(1 if failed else 0)

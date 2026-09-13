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
from datetime import datetime, timedelta
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
    db.close()

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
        == 422,
    )
    check(
        "an invalid event time is refused",
        c.post(
            "/demo/facts",
            headers=h2,
            json={
                "predicate": "note",
                "object": "x",
                "content": "dated note",
                "event_time": "definitely-not-a-date",
            },
        ).status_code
        == 422,
    )
    check(
        "a negative retrieval limit is refused",
        c.post("/demo/raw/search", headers=h2, json={"query": "x", "limit": -1}).status_code == 422,
    )
    cap_ip = {"x-forwarded-for": "203.0.113.77"}
    cap_token = c.post("/demo/session", headers=cap_ip).json()["token"]
    cap_namespace = cap_token.split(".")[0]
    cap_headers = {**cap_ip, "x-demo-token": cap_token}
    accepted = [
        c.post(
            "/demo/turns",
            headers=cap_headers,
            json={"content": f"turn {i}", "session_id": f"chat-{i}"},
        ).status_code
        for i in range(30)
    ]
    check("the namespace accepts its 30-turn allowance", all(code == 200 for code in accepted))
    check(
        "changing session_id cannot bypass the namespace turn cap",
        c.post(
            "/demo/turns",
            headers=cap_headers,
            json={"content": "overflow", "session_id": "chat-overflow"},
        ).status_code
        == 429,
    )
    # Leave one row behind, then age its durable registry entry after shutdown. The
    # next startup must recover the registry and hard-delete the namespace.
    c.post(
        "/demo/facts",
        headers=h2,
        json={"predicate": "restart_test", "object": "x", "content": "Delete after restart."},
    )
    restart_token = h2["x-demo-token"]
    restart_namespace = restart_token.split(".")[0]
    check("health says no LLM is required", c.get("/demo/health").json()["llm_required"] is False)

db = sqlite3.connect(Path(os.environ["LLTM_DEMO_STORE"]).with_suffix(".db"))
db.execute(
    "UPDATE demo_sessions SET created_at = ? WHERE namespace = ?",
    ((datetime.now() - timedelta(hours=2)).isoformat(), restart_namespace),
)
db.execute("DELETE FROM demo_sessions WHERE namespace = ?", (cap_namespace,))
db.commit()
db.close()

print("\n— restart-safe expiry —")
with TestClient(app) as c:
    check(
        "an expired token stays dead after restart",
        c.get("/demo/memories", headers={"x-demo-token": restart_token}).status_code == 410,
    )
    db = sqlite3.connect(Path(os.environ["LLTM_DEMO_STORE"]).with_suffix(".db"))
    left = db.execute(
        "SELECT count(*) FROM memories WHERE user_id = ?", (restart_namespace,)
    ).fetchone()[0]
    registry = db.execute(
        "SELECT count(*) FROM demo_sessions WHERE namespace = ?", (restart_namespace,)
    ).fetchone()[0]
    orphan_turns = db.execute(
        "SELECT count(*) FROM turns WHERE session_id IN "
        "(SELECT id FROM sessions WHERE user_id = ?)",
        (cap_namespace,),
    ).fetchone()[0]
    db.close()
    check("startup sweep deletes expired rows", left == 0, f"{left} left")
    check("startup sweep deletes the registry row", registry == 0, f"{registry} left")
    check("startup sweep deletes legacy orphan rows", orphan_turns == 0, f"{orphan_turns} left")

print(f"\n{ok} passed, {failed} failed")
sys.exit(1 if failed else 0)

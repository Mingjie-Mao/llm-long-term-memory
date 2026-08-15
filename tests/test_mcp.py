"""MCP tools, driven directly rather than through a client.

The tools are a protocol surface over `MemoryService`, so what needs testing is that
they expose the *right* surface and enforce the same boundaries the REST API does —
not that the SDK can serialise a dict.

Namespace isolation is the case that matters. An agent serving several users calls
these tools with whichever `user_id` it believes it is acting for; if the store did
not enforce the split, one user's history would answer another's question and the
failure would look like unusually good recall.
"""

from __future__ import annotations

import json
from datetime import datetime

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from llm_long_term_memory.api.service import MemoryService
from llm_long_term_memory.mcp_server import build_server
from llm_long_term_memory.store import Memory, Session, Turn

NOW = datetime(2026, 8, 15)
EARLIER = datetime(2026, 2, 3)


class StubEncoder:
    dim = 2

    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode(self, texts, **kw):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def _unwrap(result):
    """The SDK returns (content, structured) or content; take the JSON payload."""
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, dict):
        return payload
    return json.loads(payload.content[0].text)


def call(server, tool: str, args: dict):
    """Invoke a tool synchronously.

    Wrapping with `anyio.run` — which ships with the MCP SDK — rather than adding an
    async pytest plugin: this is the only async surface in the suite, and a plugin is
    a dependency CI would carry for five calls.
    """
    return _unwrap(anyio.run(lambda: server.call_tool(tool, args)))


def tools_of(server):
    return anyio.run(server.list_tools)


def memory(mid: str, user: str, content: str, **kw) -> Memory:
    return Memory(
        id=mid,
        user_id=user,
        type="semantic",
        content=content,
        token_count=6,
        ingested_at=kw.pop("ingested_at", NOW),
        valid_from=kw.pop("valid_from", NOW),
        **kw,
    )


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    svc = MemoryService(store_name="mcp-test", encoder=StubEncoder())
    svc.store.add_session(
        Session(
            id="s1",
            user_id="alice",
            started_at=NOW,
            turns=[
                Turn(
                    id="s1:0",
                    session_id="s1",
                    turn_index=0,
                    role="assistant",
                    content="Try Mod Podge to seal the newspaper vase.",
                    ts=NOW,
                )
            ],
        )
    )
    svc.store.add_memories(
        [
            memory(
                "m_old",
                "alice",
                "The user lives in Canberra.",
                subject="user",
                predicate="lives_in",
                object="Canberra",
                scope="profile",
                valid_from=EARLIER,
                ingested_at=EARLIER,
            ),
            memory(
                "m_new",
                "alice",
                "The user moved to Sydney.",
                subject="user",
                predicate="lives_in",
                object="Sydney",
                scope="profile",
            ),
            memory("m_bob", "bob", "The user lives in Perth.", subject="user"),
        ]
    )
    svc.store.mark_superseded("m_old", "m_new", NOW)
    svc.index.add(
        ["m_old", "m_new", "m_bob"],
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    yield build_server(svc)
    svc.close()


def test_the_tool_surface_is_small_and_named_for_what_an_agent_does(server):
    """Five tools covering write, recall, cite, watch-a-fact-change, forget. A larger
    surface is a larger thing for a model to get wrong."""
    assert {t.name for t in tools_of(server)} == {
        "search_memory",
        "remember",
        "search_conversations",
        "get_timeline",
        "forget",
    }


def test_every_tool_requires_an_explicit_user_id(server):
    """There is no ambient session identity: an agent acting for several people must
    say which one, and the store enforces it rather than trusting the caller."""
    for tool in tools_of(server):
        assert "user_id" in tool.input_schema["properties"], tool.name
        assert "user_id" in tool.input_schema.get("required", []), tool.name


def test_search_returns_current_facts_with_provenance(server):
    body = call(server, "search_memory", {"user_id": "alice", "query": "where do they live?"})

    ids = [m["id"] for m in body["memories"]]
    assert "m_new" in ids
    assert "m_old" not in ids, "a superseded fact is not current"
    assert body["memories"][0]["said_by"] == "user"


def test_explain_reports_what_was_not_returned_and_why(server):
    """The differentiator: an absence with no explanation is indistinguishable from a
    retrieval bug, and a model cannot tell the difference either."""
    body = call(
        server,
        "search_memory",
        {"user_id": "alice", "query": "where do they live?", "explain": True},
    )

    rejected = {r["id"]: r for r in body["not_returned"]}
    assert rejected["m_old"]["reason"] == "superseded"
    assert rejected["m_old"]["superseded_by"] == "m_new"


def test_tools_never_cross_a_namespace(server):
    body = call(server, "search_memory", {"user_id": "alice", "query": "where do they live?"})
    assert "m_bob" not in [m["id"] for m in body["memories"]]

    # And an id from another namespace cannot be forgotten by guessing it.
    result = call(server, "forget", {"user_id": "alice", "memory_id": "m_bob"})
    assert result["ok"] is False


def test_the_timeline_shows_a_fact_changing(server):
    body = call(
        server,
        "get_timeline",
        {"user_id": "alice", "subject": "user", "predicate": "lives_in"},
    )

    assert [e["id"] for e in body["entries"]] == ["m_old", "m_new"]
    assert body["entries"][0]["valid_to"] is not None
    assert body["entries"][1]["valid_to"] is None


def test_raw_conversation_search_reaches_what_extraction_dropped(server):
    """The recovery path, exposed as its own tool: structured memory holds nothing
    about Mod Podge, and the original turn does."""
    memories = call(server, "search_memory", {"user_id": "alice", "query": "Mod Podge"})
    assert not any("Mod Podge" in m["content"] for m in memories["memories"])

    turns = call(server, "search_conversations", {"user_id": "alice", "query": "Mod Podge"})
    assert any("Mod Podge" in t["text"] for t in turns["turns"])
    assert turns["turns"][0]["speaker"] == "assistant"


def test_forget_evicts_rather_than_deleting(server):
    result = call(server, "forget", {"user_id": "alice", "memory_id": "m_new"})
    assert result == {"ok": True, "memory_id": "m_new", "status": "evicted"}

    body = call(server, "search_memory", {"user_id": "alice", "query": "where do they live?"})
    assert "m_new" not in [m["id"] for m in body["memories"]]

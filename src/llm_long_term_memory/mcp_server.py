"""MCP server — the same memory service, spoken as a protocol an agent understands.

REST is how a program integrates; MCP is how an agent does. Both are thin wrappers
over `MemoryService`, which is the point: a second implementation here would let the
tools drift from the measured behaviour, and the whole value of this project is that
what you can inspect is what was measured.

Five tools, chosen so an agent can do the loop it actually needs — write a turn,
recall context, cite a source, watch a fact change, forget something — and nothing
else. A larger surface is a larger thing for a model to get wrong.

Run it:

    lltm mcp                      # stdio, for Claude Desktop / Cursor / any client
    lltm mcp --transport http     # streamable HTTP

Every tool takes an explicit `user_id`. There is no ambient session identity: an
agent serving several people must say which one it is acting for, and the store
enforces the boundary rather than trusting the caller to keep them apart.
"""

from __future__ import annotations

from typing import Any

from llm_long_term_memory.api.service import MemoryNotFound, MemoryService, NamespaceRequired

INSTRUCTIONS = """\
Long-term memory for a user, across conversations.

Use `search_memory` before answering anything that might depend on what the user has
said before — preferences, plans, past decisions, things you previously recommended.
Memories come back with the reason each was selected and, when relevant, why others
were not.

Use `remember` to store something worth recalling later. Store what a future
conversation would need, not the whole exchange.

When a memory is on topic but lacks a specific detail — an exact URL, figure or name
that extraction did not preserve — use `search_conversations` to recover the original
wording from the raw archive.

`get_timeline` shows how a fact changed over time. `forget` marks a memory evicted;
nothing is ever hard-deleted, so provenance survives.
"""


def _memory_dict(memory: Any) -> dict[str, Any]:
    """The wire shape for a model to read.

    Deliberately smaller than the REST response: a model does not need internal
    strength bookkeeping, and every extra field is context spent on plumbing rather
    than on the user's question.
    """
    return {
        "id": memory.id,
        "content": memory.content,
        "scope": memory.scope,
        "said_by": memory.source_role,
        "about": memory.subject,
        "status": memory.status,
        "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
        "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
        "source": {
            "session_id": memory.source_session_id,
            "turn_index": memory.source_turn_index,
        },
    }


def build_server(service: MemoryService | None = None, name: str = "llm-long-term-memory"):
    """Construct the MCP server over a `MemoryService`.

    The service is injected so tests can drive the tools against a temporary store
    without a credential or a network.
    """
    from mcp.server.mcpserver import MCPServer

    svc = service if service is not None else MemoryService()
    server = MCPServer(
        name=name,
        title="Long-term memory",
        instructions=INSTRUCTIONS,
        version="1.0.0",
    )

    @server.tool()
    def search_memory(
        user_id: str,
        query: str,
        limit: int = 10,
        include_superseded: bool = False,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Recall what is known about a user, relevant to a query.

        Returns current facts by default. `explain=True` additionally reports what was
        retrieved and *not* returned, with the reason — `superseded` (no longer true)
        or `below_rank` (still true, scored outside the cut).
        """
        result = svc.search(
            user_id, query, limit=limit, include_superseded=include_superseded, explain=explain
        )
        payload: dict[str, Any] = {
            "memories": [
                {**_memory_dict(hit.memory), "relevance": round(hit.score, 3)}
                for hit in result.memories
            ],
            "candidates_considered": result.candidates_considered,
        }
        if explain:
            payload["not_returned"] = [
                {
                    "id": r.memory_id,
                    "reason": r.reason,
                    "superseded_by": r.superseded_by,
                    "content": r.content,
                }
                for r in result.rejected
            ]
        return payload

    @server.tool()
    def remember(user_id: str, content: str, role: str = "user") -> dict[str, Any]:
        """Store a conversation turn and extract what is worth remembering from it.

        Returns the memories created, so the agent can confirm what was understood
        rather than assuming the whole turn was retained verbatim.
        """
        result = svc.add_message(user_id, role, content)
        return {
            "session_id": result["session_id"],
            "memories_created": [_memory_dict(m) for m in result["memories"]],
        }

    @server.tool()
    def search_conversations(user_id: str, query: str, limit: int = 3) -> dict[str, Any]:
        """Search the original conversation text, not the extracted memories.

        For when a memory is on topic but the exact detail — a URL, a figure, a
        product name — did not survive extraction. Structured memory is a lossy
        compression; this is the source it was compressed from.
        """
        turns = svc.raw_search(user_id, query, limit=limit)
        return {
            "turns": [
                {
                    "session_id": t.session_id,
                    "turn_index": t.turn_index,
                    "speaker": t.role,
                    "text": t.content,
                }
                for t in turns
            ]
        }

    @server.tool()
    def get_timeline(user_id: str, subject: str, predicate: str) -> dict[str, Any]:
        """How one fact changed over time, oldest first.

        `subject` is who the fact is about ("user", or a person's name); `predicate`
        is the attribute, as it appears on a memory ("lives_in", "job_title").
        Entries with `valid_to` set were replaced by the entry naming them.
        """
        entries = svc.timeline(user_id, subject, predicate)
        return {
            "subject": subject,
            "predicate": predicate,
            "entries": [_memory_dict(m) for m in entries],
        }

    @server.tool()
    def forget(user_id: str, memory_id: str) -> dict[str, Any]:
        """Mark a memory as evicted so it stops being retrieved.

        Not a deletion: the row and its provenance remain, so it stays possible to
        explain why something was once believed.
        """
        try:
            svc.forget(user_id, memory_id)
        except MemoryNotFound:
            return {"ok": False, "error": f"no memory {memory_id!r} in this namespace"}
        return {"ok": True, "memory_id": memory_id, "status": "evicted"}

    return server


def run(transport: str = "stdio", service: MemoryService | None = None) -> None:
    server = build_server(service)
    if transport == "stdio":
        import anyio

        anyio.run(server.run_stdio_async)
    elif transport == "http":
        import anyio

        anyio.run(server.run_streamable_http_async)
    else:  # pragma: no cover - guarded by the CLI's choices
        raise ValueError(f"unknown transport {transport!r}")


__all__ = ["INSTRUCTIONS", "NamespaceRequired", "build_server", "run"]

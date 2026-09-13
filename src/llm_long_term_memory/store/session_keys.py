"""Reversible database keys for session ids that are only unique per user."""

from __future__ import annotations

_PREFIX = "scoped-session-v1:"


def split_scoped_session_id(value: str | None) -> tuple[str, str] | None:
    """Return ``(user_id, external_id)`` or ``None`` for a legacy/raw id."""
    if value is None or not value.startswith(_PREFIX):
        return None
    length_text, separator, payload = value[len(_PREFIX) :].partition(":")
    if not separator or not length_text.isdigit():
        return None
    user_length = int(length_text)
    if user_length > len(payload):
        return None
    return payload[:user_length], payload[user_length:]


def scoped_session_id(user_id: str, external_id: str) -> str:
    """Return a stable, globally unique storage id for a user's session."""
    parsed = split_scoped_session_id(external_id)
    if parsed is not None:
        if parsed[0] != user_id:
            raise ValueError("a scoped session id cannot be moved to another user")
        return external_id
    return f"{_PREFIX}{len(user_id)}:{user_id}{external_id}"


def external_session_id(value: str | None) -> str | None:
    """Return the client/dataset id, hiding the internal namespace prefix."""
    parsed = split_scoped_session_id(value)
    return parsed[1] if parsed is not None else value

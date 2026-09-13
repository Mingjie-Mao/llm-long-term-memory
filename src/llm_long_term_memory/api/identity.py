"""Who the caller is, decided by a credential rather than by the request.

Until now `user_id` arrived in the query string or the body, so any caller could read,
write and delete any namespace by typing a different value. Every "tenant isolation" the
service performs was isolation between namespaces the caller chose for itself, which is
addressing, not authorization.

A principal is now derived from a bearer token the caller had to already hold. Tokens map
to tenants in the environment; the mapping is never in a committed config, because an
experiment config is checked in so results reproduce and a credential must never end up
in one.

**Open mode is deliberate, and loud.** With no tokens configured the service keeps the old
behaviour, because the research CLI, the inspector and the offline test suite all drive it
without credentials and breaking them buys nothing. What changes is that open mode is now
a stated condition rather than an unexamined default: `/healthz` reports it, startup logs
it, and `Principal.trusted` is False so nothing downstream can mistake a namespace the
caller named for one it proved.

**Why bearer tokens rather than OIDC.** A shared secret verified with a constant-time
compare needs no dependency, no network call and no key rotation infrastructure, so it can
land and be tested now. It is deliberately the weaker scheme: tokens do not expire, carry
no claims, and cannot be revoked without an environment change. `principal_from_token` is
the seam an OIDC/JWT verifier replaces, and it returns the same `Principal` either way.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

# `LLTM_API_TOKENS=token1:tenant-a,token2:tenant-b`
#
# Read from the environment rather than from `Settings` so that a token never reaches a
# model dump, a `/v1/config` response or a log line that renders settings. Nothing here
# returns the token itself.
TOKENS_ENV = "LLTM_API_TOKENS"


@dataclass(frozen=True, slots=True)
class Principal:
    """The namespace a caller has proved it may act on.

    `trusted` is what separates a proved namespace from a named one. It is carried
    explicitly rather than inferred from whether auth happens to be configured, so that a
    handler cannot accidentally treat open mode as authenticated by forgetting to check.
    """

    user_id: str
    trusted: bool

    @property
    def is_anonymous(self) -> bool:
        return not self.trusted


class AuthenticationRequired(RuntimeError):
    """No usable credential was presented, and the deployment requires one."""


class NamespaceForbidden(RuntimeError):
    """A caller named a namespace that is not the one it authenticated as."""


class AuthenticationConfigurationError(RuntimeError):
    """Authentication was requested but its configuration is unusable."""


def configured_tokens(environ: dict[str, str] | None = None) -> dict[str, str]:
    """token -> tenant. Empty means open mode."""
    raw = (environ if environ is not None else os.environ).get(TOKENS_ENV, "")
    if not raw.strip():
        return {}
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        token, _, tenant = pair.strip().partition(":")
        token, tenant = token.strip(), tenant.strip()
        # Never turn malformed authentication configuration into open access. Keep
        # the error independent of the configured value so credentials cannot leak.
        if not token or not tenant or token in mapping:
            raise AuthenticationConfigurationError("invalid LLTM_API_TOKENS configuration")
        mapping[token] = tenant
    return mapping


def auth_enabled(environ: dict[str, str] | None = None) -> bool:
    return bool(configured_tokens(environ))


def _bearer(header: str | None) -> str:
    value = (header or "").strip()
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def principal_from_token(
    authorization: str | None, environ: dict[str, str] | None = None
) -> Principal:
    """Resolve a credential to a principal, or refuse.

    In open mode this raises nothing and returns no principal: the caller falls back to
    the namespace it named, which `resolve_namespace` records as untrusted.
    """
    tokens = configured_tokens(environ)
    if not tokens:
        return Principal(user_id="", trusted=False)

    presented = _bearer(authorization)
    if not presented:
        raise AuthenticationRequired("this deployment requires a bearer token")

    # Constant-time compare against every configured token. A dict lookup would leak
    # token length and prefix through timing, and the set is small enough that comparing
    # all of them costs nothing.
    matched = ""
    for token, tenant in tokens.items():
        if hmac.compare_digest(token.encode("utf-8"), presented.encode("utf-8")):
            matched = tenant
    if not matched:
        raise AuthenticationRequired("the presented token is not recognised")
    return Principal(user_id=matched, trusted=True)


def resolve_namespace(principal: Principal, requested: str | None) -> str:
    """The namespace this call may act on.

    When the principal is trusted the credential wins, and a request naming a different
    namespace is refused rather than quietly redirected. Silently rewriting it would make
    a cross-tenant attempt look like a successful call against an empty namespace, which
    is indistinguishable from a legitimate one and leaves no trace.
    """
    asked = (requested or "").strip()
    if principal.trusted:
        if asked and asked != principal.user_id:
            raise NamespaceForbidden("user_id does not match the authenticated principal")
        return principal.user_id
    return asked

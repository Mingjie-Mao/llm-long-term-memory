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

**`LLTM_REQUIRE_AUTH` makes open mode impossible to reach by accident.** A deployment that
holds real user data cannot rely on someone remembering to set `LLTM_API_TOKENS`: forgetting
it produces a service that works perfectly and authorises everyone, which is the failure
that looks most like success. With the switch set, a missing or unusable token map is a
refusal at three independent points — the process will not start, `/healthz` goes 503, and
every authenticated route answers 401 — because one of them will eventually be bypassed
(an injected app in a test, an env var edited on a running host) and fail-closed means the
remaining two still hold.

An unparseable value for the switch is an error rather than False. `LLTM_REQUIRE_AUTH=ture`
silently disabling the guard is precisely the accident the guard exists to prevent.

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

REQUIRE_ENV = "LLTM_REQUIRE_AUTH"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


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


def auth_required(environ: dict[str, str] | None = None) -> bool:
    """Whether this deployment refuses to run without credentials."""
    raw = (environ if environ is not None else os.environ).get(REQUIRE_ENV, "")
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise AuthenticationConfigurationError(
        f"{REQUIRE_ENV} must be one of {sorted(_TRUE | _FALSE - {''})}, not {raw.strip()!r}"
    )


def check_authentication_configuration(environ: dict[str, str] | None = None) -> bool:
    """Validate the pair, and return whether authentication is enforced.

    Raises when the switch is set and no usable token map backs it. Called at startup, by
    `/healthz`, and on every credential resolution: three places, because the point of a
    fail-closed switch is that no single bypass reopens the door.
    """
    required = auth_required(environ)
    tokens = configured_tokens(environ)
    if required and not tokens:
        raise AuthenticationConfigurationError(
            f"{REQUIRE_ENV} is set but {TOKENS_ENV} configures no tokens; "
            "this deployment refuses to serve in open mode"
        )
    return bool(tokens)


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
    # Re-checked per request, not read from startup state: an environment edited on a
    # running host must not leave the door open behind a process that validated once.
    check_authentication_configuration(environ)
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

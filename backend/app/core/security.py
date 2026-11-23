"""API gateway middleware: authentication, rate limiting, audit log, metrics, headers.

* **Authentication** (optional): set ``MD_API_KEYS='["alice:<key>", "ci:<key>"]'``.
  Requests must then send ``X-API-Key: <key>`` (or ``Authorization: Bearer <key>``).
  The key's name becomes the *actor* recorded in the audit log and registry.
  Health stays public; ``/metrics`` is public unless ``MD_METRICS_REQUIRE_AUTH``.
* **Rate limiting**: sliding 60-second window per actor (or client IP),
  ``MD_RATE_LIMIT_PER_MINUTE`` (0 disables). Responds 429 with Retry-After.
* **Audit log**: every mutating request (POST/PUT/PATCH/DELETE) is stored with
  actor, method, path, status and client IP.
* **Metrics**: request counts and latency per route template.
* **Security headers** on every response.
"""

import hmac
import logging
import threading
import time
from collections import defaultdict, deque

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.monitoring.metrics import REGISTRY

log = logging.getLogger("modeldoctor.gateway")

PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect",
                "/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/status"}
# Read-only users may still ask the assistant (a POST that changes nothing).
VIEWER_ALLOWED_POSTS = ("/api/v1/assistant/",)
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def parse_api_keys(entries: list[str]) -> dict[str, str]:
    """``["name:key", ...]`` -> {key: name}. A bare key gets the name 'api-key-N'."""
    keys = {}
    for i, entry in enumerate(entries, 1):
        name, sep, key = entry.partition(":")
        if not sep:
            name, key = f"api-key-{i}", entry
        if key:
            keys[key.strip()] = name.strip() or f"api-key-{i}"
    return keys


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> float | None:
        """None if allowed, otherwise seconds until a slot frees up."""
        if self.per_minute <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            q = self._hits[identity]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(0.0, 60 - (now - q[0]))
            q.append(now)
            return None


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class GatewayMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings, session_factory):
        super().__init__(app)
        self.settings = settings
        self.session_factory = session_factory
        self.keys = parse_api_keys(settings.api_keys)
        self.limiter = RateLimiter(settings.rate_limit_per_minute)

    def _authenticate(self, request) -> tuple[str, str, str | None] | None:
        """(actor, role, user_id) from an API key or a user's Bearer token, else None."""
        supplied = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else None
        for candidate in (supplied, bearer):
            if not candidate:
                continue
            for key, name in self.keys.items():
                if hmac.compare_digest(candidate.encode(), key.encode()):
                    return name, "admin", None  # API keys are service accounts with full access
        if bearer and bearer.count(".") == 2:
            return self._user_from_token(bearer)
        return None

    def _user_from_token(self, token: str) -> tuple[str, str, str | None] | None:
        from app.core.auth import decode_token
        from app.db.models import User

        claims = decode_token(self.settings, token)
        if not claims:
            return None
        # Re-read the user so deactivation and role changes take effect immediately.
        with self.session_factory() as db:
            user = db.get(User, claims["sub"])
            if user is None or not user.is_active:
                return None
            return user.email, user.role, user.id

    async def dispatch(self, request, call_next):
        started = time.perf_counter()
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        actor = None

        enforce = bool(self.keys) or self.settings.auth_required
        needs_auth = enforce and request.method != "OPTIONS" and path not in PUBLIC_PATHS and not (
            path == "/metrics" and not self.settings.metrics_require_auth
        )
        identity = self._authenticate(request) if request.method != "OPTIONS" else None
        role = user_id = None
        if identity:
            actor, role, user_id = identity
        if needs_auth and identity is None:
            response = _error(401, "unauthorized", "Sign in, or send an API key (X-API-Key).",
                              {"WWW-Authenticate": "Bearer"})
            return self._finish(request, response, started, actor, client_ip)
        if (role == "viewer" and request.method in MUTATING and path.startswith("/api/")
                and path not in PUBLIC_PATHS and not path.startswith(VIEWER_ALLOWED_POSTS)):
            response = _error(403, "forbidden", "Your account is read-only (viewer).")
            return self._finish(request, response, started, actor, client_ip)
        request.state.actor = actor
        request.state.role = role
        request.state.user_id = user_id

        if request.method != "OPTIONS" and path not in PUBLIC_PATHS and path != "/metrics":
            wait = self.limiter.check(actor or client_ip)
            if wait is not None:
                response = _error(429, "rate_limited", "Too many requests; slow down.",
                                  {"Retry-After": str(int(wait) + 1)})
                return self._finish(request, response, started, actor, client_ip)

        response = await call_next(request)
        return self._finish(request, response, started, actor, client_ip)

    def _finish(self, request, response, started, actor, client_ip):
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"
        if route.startswith("/") and request.url.path.startswith("/api/v1") and not route.startswith("/api/v1"):
            route = "/api/v1" + route  # sub-router routes report their path without the prefix
        status = response.status_code
        REGISTRY.inc("modeldoctor_http_requests_total",
                     {"method": request.method, "route": route, "status": str(status)},
                     help_="HTTP requests by route template and status.")
        REGISTRY.observe("modeldoctor_http_request_duration_seconds",
                         {"method": request.method, "route": route}, time.perf_counter() - started,
                         help_="HTTP request latency in seconds.")
        if request.method in MUTATING and request.url.path.startswith("/api/"):
            self._audit(actor, request.method, request.url.path, status, client_ip)
        return response

    def _audit(self, actor, method, path, status, ip):
        from app.db.models import AuditLog

        try:
            with self.session_factory() as db:
                db.add(AuditLog(actor=actor or "anonymous", method=method, path=path[:500],
                                status_code=status, client_ip=ip[:64]))
                db.commit()
        except Exception:  # noqa: BLE001 - auditing must never break a request
            log.exception("failed to write audit log")

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

PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}
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

    def _authenticate(self, request) -> str | None:
        supplied = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not supplied and auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if not supplied:
            return None
        for key, name in self.keys.items():
            if hmac.compare_digest(supplied.encode(), key.encode()):
                return name
        return None

    async def dispatch(self, request, call_next):
        started = time.perf_counter()
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        actor = None

        needs_auth = bool(self.keys) and request.method != "OPTIONS" and path not in PUBLIC_PATHS and not (
            path == "/metrics" and not self.settings.metrics_require_auth
        )
        if self.keys:
            actor = self._authenticate(request)
            if needs_auth and actor is None:
                response = _error(401, "unauthorized", "Missing or invalid API key (send X-API-Key).",
                                  {"WWW-Authenticate": "Bearer"})
                return self._finish(request, response, started, actor, client_ip)
        request.state.actor = actor

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

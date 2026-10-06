"""Opt-in Headroom proxy extension: local read-only session metrics and inspection."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .telemetry import IncompatibleLogger, belongs, inspect, record, snapshot, summarize

PREFIX = "/headroom-mod/v1"


def require_exact_origin(request: Request) -> None:
    """Captured text is readable only by this origin, not other local sites."""
    origin = request.headers.get("origin")
    if origin is None:
        return  # Native clients do not send a browser Origin header.
    try:
        parsed = urlsplit(origin)
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        target_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        allowed = (
            parsed.scheme in {"http", "https"}
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and (parsed.scheme, parsed.hostname, port)
            == (request.url.scheme, request.url.hostname, target_port)
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def install(app: Any, config: Any) -> None:
    # Reuse Headroom's actual peer + Host/DNS-rebinding + same-origin guards.
    from headroom.proxy.loopback_guard import require_loopback, require_same_origin

    proxy = app.state.proxy
    # Fail loudly during extension install rather than serve made-up zeros.
    snapshot(proxy.logger)
    epoch = str(uuid.uuid4())
    cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
    router = APIRouter(
        prefix=PREFIX,
        dependencies=[
            Depends(require_loopback),
            Depends(require_same_origin),
            Depends(require_exact_origin),
        ],
    )

    def response(payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse(
            payload, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        )

    def entries() -> tuple[Any, ...]:
        try:
            return snapshot(proxy.logger)
        except IncompatibleLogger as exc:
            raise HTTPException(503, "Headroom request-log adapter is incompatible") from exc

    @router.get("/health")
    async def health() -> JSONResponse:
        entries()
        try:
            headroom_version = version("headroom-ai")
        except PackageNotFoundError:
            headroom_version = "source/unknown"
        return response(
            {
                "schema_version": 1,
                "service": "headroom-claude-mod",
                "version": __version__,
                "headroom_version": headroom_version,
                "epoch": epoch,
                "log_full_messages": bool(config.log_full_messages),
                "read_only": True,
            }
        )

    @router.get("/sessions/{session_id}")
    async def session(session_id: uuid.UUID, limit: int = Query(100, ge=1, le=100)) -> JSONResponse:
        sid = str(session_id)
        now = time.monotonic()
        cached = cache.get(sid)
        if cached is not None and now - cached[0] < 2:
            payload = cached[1]
            cache.move_to_end(sid)
        else:
            logs = entries()
            unique: dict[str, dict[str, Any]] = {}
            for index, entry in enumerate(logs):
                if not belongs(entry, sid):
                    continue
                key = getattr(entry, "request_id", None) or f"unidentified-{index}"
                # A completion update/repeated log line must not double-count a request.
                unique.pop(key, None)
                unique[key] = record(entry)
            rows = list(unique.values())
            # The window is proxy-global: other conversations can evict this one's history.
            capacity = proxy.logger._logs.maxlen
            payload = {
                "schema_version": 1,
                "session_id": sid,
                "epoch": epoch,
                "scope": "conversation_and_inherited_children",
                "basis": "retained_request_totals_not_unique_context_or_lifetime",
                "retention": {
                    "capacity": capacity,
                    "proxy_records": len(logs),
                    "window_full": len(logs) >= capacity,
                    "message_window": getattr(proxy.logger, "MESSAGE_WINDOW", None),
                },
                "log_full_messages": bool(config.log_full_messages),
                "totals": summarize(rows),
                "latest": rows[-1] if rows else None,
                "requests": list(reversed(rows[-100:])),
            }
            cache[sid] = (now, payload)
            cache.move_to_end(sid)
            while len(cache) > 32:
                cache.popitem(last=False)
        return response({**payload, "requests": payload["requests"][:limit]})

    @router.get("/sessions/{session_id}/requests/{request_id}")
    async def request_detail(
        session_id: uuid.UUID,
        request_id: str,
        side: Literal["original", "compressed", "diff"] = "compressed",
        message: int = Query(0, ge=0, le=100_000),
        page: int = Query(0, ge=0, le=100_000),
    ) -> JSONResponse:
        if len(request_id) > 128:
            raise HTTPException(404, "Request not retained in this conversation")
        # Filter before touching a body; don't call the global feed and post-filter.
        entry = next(
            (
                e
                for e in reversed(entries())
                if belongs(e, str(session_id)) and getattr(e, "request_id", None) == request_id
            ),
            None,
        )
        if entry is None:
            raise HTTPException(404, "Request not retained in this conversation")
        result = (
            inspect(entry, side=side, message=message, page=page)
            if config.log_full_messages
            else {
                "available": False,
                "reason": "Start Headroom with --log-messages to enable inspection. Capture is never enabled by the mod.",
            }
        )
        return response(
            {
                "schema_version": 1,
                "session_id": str(session_id),
                "epoch": epoch,
                "request_id": request_id,
                **result,
            }
        )

    app.include_router(router)

"""FastAPI integration for gateway authentication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from headroom.proxy.gateway.auth import (
    GatewayAuthenticator,
    GatewayAuthorizer,
    validate_gateway_browser_request,
)
from headroom.proxy.gateway.errors import GatewayPublicError

if TYPE_CHECKING:
    from headroom.proxy.gateway.config import GatewayConfigSnapshot


_UNPRIVILEGED_LOCAL_PATHS = frozenset({"/livez", "/readyz"})


def install_gateway_auth_middleware(
    app: FastAPI,
    snapshot: GatewayConfigSnapshot,
    environ: Mapping[str, str],
) -> None:
    """Install mandatory gateway auth while leaving readiness locally observable."""

    authenticator = GatewayAuthenticator(snapshot, environ)
    authorizer = GatewayAuthorizer(snapshot)
    app.state.gateway_authenticator = authenticator
    app.state.gateway_authorizer = authorizer

    @app.middleware("http")
    async def gateway_authentication(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _UNPRIVILEGED_LOCAL_PATHS:
            return await call_next(request)
        try:
            validate_gateway_browser_request(request.headers)
            request.state.gateway_principal = authenticator.authenticate(
                request.headers,
                query_string=request.scope.get("query_string", b""),
            )
        except GatewayPublicError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "type": "gateway_error",
                        "code": exc.code,
                        "message": exc.message,
                    }
                },
            )
        return await call_next(request)

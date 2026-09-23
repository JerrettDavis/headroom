"""FastAPI integration for gateway authentication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from headroom.proxy.gateway.auth import validate_gateway_browser_request
from headroom.proxy.gateway.egress import EgressPolicy
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.runtime import GatewayRuntime
from headroom.proxy.gateway.transport import tls_context

if TYPE_CHECKING:
    from headroom.proxy.gateway.config import GatewayConfigSnapshot


_UNPRIVILEGED_LOCAL_PATHS = frozenset({"/livez", "/readyz"})
_DISABLED_CONTROL_PATHS = frozenset(
    {
        "/admin/runtime-env",
        "/settings",
        "/settings/apply",
        "/settings/schema",
        "/dashboard/settings",
    }
)


def install_gateway_auth_middleware(
    app: FastAPI,
    snapshot: GatewayConfigSnapshot,
    environ: Mapping[str, str],
) -> None:
    """Install mandatory gateway auth while leaving readiness locally observable."""

    runtime = GatewayRuntime(snapshot, environ=environ)
    app.state.gateway_egress_policy = EgressPolicy()
    app.state.gateway_tls_context = tls_context(snapshot)
    app.state.gateway_runtime = runtime
    app.state.gateway_authenticator = runtime.authenticator
    app.state.gateway_authorizer = runtime.authorizer
    app.state.gateway_model_registry = runtime.models
    app.state.gateway_resource_registry = runtime.resources
    app.state.gateway_account_router = runtime.router
    app.state.gateway_admission = runtime.admission
    app.state.gateway_credential_broker = runtime.broker

    @app.middleware("http")
    async def gateway_authentication(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _DISABLED_CONTROL_PATHS:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        if request.url.path in _UNPRIVILEGED_LOCAL_PATHS:
            return await call_next(request)
        try:
            validate_gateway_browser_request(request.headers)
            request.state.gateway_principal = runtime.authenticator.authenticate(
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

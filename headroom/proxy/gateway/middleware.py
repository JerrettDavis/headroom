"""FastAPI integration for gateway authentication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from headroom.proxy.gateway.admission import AdmissionController
from headroom.proxy.gateway.auth import (
    GatewayAuthenticator,
    GatewayAuthorizer,
    validate_gateway_browser_request,
)
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.models import ModelRegistry
from headroom.proxy.gateway.resources import ResourceRegistry
from headroom.proxy.gateway.routing import AccountRouter

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
    app.state.gateway_model_registry = ModelRegistry(snapshot)
    app.state.gateway_resource_registry = ResourceRegistry()
    app.state.gateway_account_router = AccountRouter(
        available_accounts={
            credential.id for credential in snapshot.credentials if credential.enabled
        }
    )
    app.state.gateway_admission = AdmissionController(
        budget_limit=None,
        max_concurrency=128,
        queue_limit=128,
        unknown_cost_policy="allow",
    )

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

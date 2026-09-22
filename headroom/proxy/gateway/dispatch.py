"""Pure native gateway dispatch using existing proxy HTTP transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from headroom.proxy.gateway.config import Protocol
from headroom.proxy.gateway.context import GatewayRequestContext
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError, GatewayPublicError
from headroom.proxy.gateway.models import Capability

_REQUEST_HEADER_DENYLIST = frozenset(
    {"host", "content-length", "connection", "transfer-encoding", "upgrade"}
)
_RESPONSE_HEADER_DENYLIST = frozenset(
    {"content-length", "connection", "transfer-encoding", "content-encoding"}
)
_QUERY_CREDENTIAL_NAMES = frozenset(
    {"api_key", "key", "access_token", "token", "x-api-key", "x-goog-api-key"}
)

DispatchContract = Literal["strict-native", "routed-native", "translated"]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    contract: DispatchContract
    capabilities: frozenset[Capability]
    body: bytes
    mutation_reasons: tuple[str, ...]


class GatewayDispatcher:
    """Resolve explicit native entity mutations before provider I/O."""

    @staticmethod
    def resolve_body(
        body: bytes,
        *,
        public_model: str,
        upstream_model: str,
        declared_contract: DispatchContract,
    ) -> DispatchPlan:
        if declared_contract == "strict-native":
            if public_model != upstream_model:
                raise GatewayAuthorizationError(
                    status_code=500,
                    code="gateway_route_invalid",
                    message="Strict-native route cannot rewrite its model",
                )
            return DispatchPlan(
                contract="strict-native",
                capabilities=frozenset({Capability.GENERATE}),
                body=body,
                mutation_reasons=(),
            )
        if declared_contract == "routed-native":
            rewritten = rewrite_routed_native_model(
                body,
                public_model=public_model,
                upstream_model=upstream_model,
            )
            reasons = ("model_alias",) if rewritten is not body else ()
            return DispatchPlan(
                contract="routed-native",
                capabilities=frozenset({Capability.GENERATE}),
                body=rewritten,
                mutation_reasons=reasons,
            )
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_translation_required",
            message="Translated dispatch requires a qualified protocol adapter",
        )


async def _iter_upstream_bytes(upstream: Any) -> AsyncIterator[bytes]:
    """Yield preloaded test responses or live transport chunks exactly once."""

    if upstream.is_stream_consumed:
        yield upstream.content
        return
    async for chunk in upstream.aiter_raw():
        yield chunk


def rewrite_routed_native_model(
    body: bytes,
    *,
    public_model: str,
    upstream_model: str,
) -> bytes:
    """Return original bytes for identity routes; patch only model otherwise."""

    if public_model == upstream_model:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_request_invalid",
            message="Gateway request body is not valid JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("model") != public_model:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_model_mismatch",
            message="Request model does not match the authorized route model",
        )
    payload["model"] = upstream_model
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


async def dispatch_native_http(
    request: Request,
    proxy: Any,
    protocol: Protocol,
    *,
    public_model: str | None = None,
) -> Response:
    """Authorize, lease, and forward one native HTTP entity without optimization."""

    try:
        body = await request.body()
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_request_invalid",
                message="Gateway request body must be a JSON object",
            )
        body_model = payload.get("model")
        if public_model is None and not isinstance(body_model, str):
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_model_required",
                message="A string model is required",
            )
        requested_model = public_model if public_model is not None else body_model
        assert isinstance(requested_model, str)
        principal = request.state.gateway_principal
        route = request.app.state.gateway_authorizer.authorize(
            principal,
            scope="inference",
            protocol=protocol,
            public_model=requested_model,
        )
        lease = await request.app.state.gateway_credential_broker.acquire(route)
        upstream_path = request.url.path
        if public_model is not None and route.public_model != route.upstream_model:
            upstream_path = upstream_path.replace(
                route.public_model,
                route.upstream_model,
                1,
            )
        target = route.upstream_origin.rstrip("/") + upstream_path
        safe_query = [
            (name, value)
            for name, value in parse_qsl(request.url.query, keep_blank_values=True)
            if name.casefold() not in _QUERY_CREDENTIAL_NAMES
        ]
        if safe_query:
            target += "?" + urlencode(safe_query)
        outbound_body = body
        if public_model is None:
            plan = GatewayDispatcher.resolve_body(
                body,
                public_model=route.public_model,
                upstream_model=route.upstream_model,
                declared_contract=route.body_contract,
            )
            outbound_body = plan.body
        client_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _REQUEST_HEADER_DENYLIST
        }
        outbound_headers = build_managed_upstream_headers(
            client_headers,
            lease,
            target,
            method=request.method,
            body=outbound_body,
        )
        request.state.gateway = GatewayRequestContext(
            principal=principal,
            route=route,
            ingress_protocol=protocol,
            request_id=request.headers.get("x-request-id", ""),
        )
        if proxy.http_client is None:
            raise RuntimeError("gateway HTTP transport is not initialized")
        upstream_request = proxy.http_client.build_request(
            request.method,
            target,
            headers=outbound_headers,
            content=outbound_body,
        )
        upstream = await proxy.http_client.send(
            upstream_request,
            stream=True,
            follow_redirects=False,
        )
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _RESPONSE_HEADER_DENYLIST
        }
        return StreamingResponse(
            _iter_upstream_bytes(upstream),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=None,
            background=BackgroundTask(upstream.aclose),
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


def gateway_model_catalog(request: Request) -> JSONResponse:
    principal = request.state.gateway_principal
    routes = request.app.state.gateway_model_registry.visible_routes(principal)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": route.id,
                    "object": "model",
                    "owned_by": "headroom-gateway",
                    "headroom": {
                        "route": route.route_id,
                        "protocols": route.protocols,
                        "body_contract": route.body_contract,
                    },
                }
                for route in routes
            ],
        }
    )

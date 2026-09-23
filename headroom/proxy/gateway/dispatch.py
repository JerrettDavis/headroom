"""Pure native gateway dispatch using existing proxy HTTP transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from headroom.proxy.gateway.admission import AdmissionRequest, AdmissionReservation
from headroom.proxy.gateway.capabilities import requested_features
from headroom.proxy.gateway.config import Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayRequestContext
from headroom.proxy.gateway.credentials import CredentialLease
from headroom.proxy.gateway.destinations import path_within, validate_path
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError, GatewayPublicError
from headroom.proxy.gateway.models import Capability
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.transport import private_transport

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


def route_target(route: RouteConfig, path: str) -> str:
    validate_path(path)
    if route.provider == "compatible" and not path_within(path, route.upstream_path_prefix):
        if not path.startswith("/v1/"):
            raise ValueError("unsupported compatible route path")
        path = route.upstream_path_prefix.rstrip("/") + "/" + path.removeprefix("/v1/")
    return route.upstream_origin.rstrip("/") + path


def _upstream_error() -> GatewayPublicError:
    return GatewayPublicError(
        status_code=502, code="gateway_upstream_error", message="Upstream request failed"
    )


def _safe_response_headers(upstream: Any) -> dict[str, str]:
    content_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return (
        {"content-type": content_type}
        if content_type in {"application/json", "text/event-stream", "application/octet-stream"}
        else {}
    )


async def _send_managed(
    request: Request,
    proxy: Any,
    route: RouteConfig,
    lease: CredentialLease,
    target: str,
    client_headers: dict[str, str],
    body: bytes,
) -> Any:
    destination = await asyncio.to_thread(
        request.state.gateway_generation.egress_policy.authorize, lease, target, route=route
    )
    headers = build_managed_upstream_headers(
        client_headers,
        lease,
        target,
        resolved_addresses=destination.addresses,
        route=route,
        method=request.method,
        body=body,
    )
    if request.state.gateway_generation.http_client is None:
        raise _upstream_error()
    upstream_request = request.state.gateway_generation.http_client.build_request(
        request.method, target, headers=headers, content=body
    )
    upstream_request.extensions["gateway_destination"] = destination
    try:
        with private_transport():
            upstream = await request.state.gateway_generation.http_client.send(
                upstream_request, stream=True, follow_redirects=False
            )
    except Exception:
        raise _upstream_error() from None
    if not 200 <= upstream.status_code < 300:
        await upstream.aclose()
        raise _upstream_error()
    return upstream


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

    async def chunks() -> AsyncIterator[bytes]:
        if upstream.is_stream_consumed:
            yield upstream.content
        else:
            async for chunk in upstream.aiter_raw():
                yield chunk

    try:
        is_sse = (
            getattr(upstream, "headers", {})
            .get("content-type", "")
            .split(";", 1)[0]
            .strip()
            .lower()
            == "text/event-stream"
        )
        pending = b""
        async for chunk in chunks():
            if not is_sse:
                yield chunk
                continue
            pending += chunk
            if len(pending) > 1_048_576:
                raise _upstream_error()
            while b"\n\n" in pending or b"\r\n\r\n" in pending:
                separators = [
                    separator for separator in (b"\n\n", b"\r\n\r\n") if separator in pending
                ]
                separator = min(separators, key=pending.index)
                event, pending = pending.split(separator, 1)
                fields = []
                for line in event.splitlines():
                    name, _, value = line.partition(b":")
                    fields.append((name, value.removeprefix(b" ")))
                if any(name == b"event" and value == b"error" for name, value in fields):
                    raise _upstream_error()
                data = b"\n".join(value for name, value in fields if name == b"data")
                if data and data != b"[DONE]":
                    parsed = json.loads(data)
                    if isinstance(parsed, dict) and (
                        parsed.get("error")
                        or parsed.get("type") in {"error", "response.failed", "response.error"}
                    ):
                        raise _upstream_error()
                yield event + separator
        if pending:
            raise _upstream_error()
    except Exception:
        raise _upstream_error() from None


async def _bind_response_stream(
    chunks: AsyncIterator[bytes],
    *,
    registry: Any,
    principal_id: str,
    route_id: str,
    account_ref: str,
    generation: Any,
) -> AsyncIterator[bytes]:
    """Observe bounded complete SSE events while forwarding each chunk unchanged."""

    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
        if len(buffer) > 1_048_576:
            raise GatewayAuthorizationError(
                status_code=502,
                code="gateway_stream_event_too_large",
                message="Upstream stream event exceeded the gateway bound",
            )
        while b"\n\n" in buffer:
            raw_event, remainder = bytes(buffer).split(b"\n\n", 1)
            buffer = bytearray(remainder)
            data = b"\n".join(
                line.removeprefix(b"data: ")
                for line in raw_event.splitlines()
                if line.startswith(b"data:")
            )
            if data and data != b"[DONE]":
                with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError):
                    event = json.loads(data)
                    response = event.get("response") if isinstance(event, dict) else None
                    if (
                        isinstance(response, dict)
                        and event.get("type") == "response.created"
                        and isinstance(response.get("id"), str)
                    ):
                        await registry.bind(
                            ResourceBinding(
                                provider_id=response["id"],
                                principal_id=principal_id,
                                route_id=route_id,
                                account_ref=account_ref,
                                adapter="openai-responses",
                                expires_at=time.time()
                                + generation.snapshot.limits.resource_ttl_seconds,
                                authority_fingerprint=generation.account_key(account_ref),
                                target_fingerprint=generation.target_key(route_id),
                                generation=generation.number,
                            )
                        )
        yield chunk


async def _finalize_stream(
    chunks: AsyncIterator[bytes],
    reservation: AdmissionReservation,
) -> AsyncIterator[bytes]:
    completed = False
    try:
        async for chunk in chunks:
            yield chunk
        completed = True
    finally:
        if completed:
            await reservation.finalize(actual_cost=None)
        else:
            await reservation.release()


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

    reservation: AdmissionReservation | None = None
    handed_off = False
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
        route = request.state.gateway_generation.authorizer.authorize(
            principal,
            scope="inference",
            protocol=protocol,
            public_model=requested_model,
            transport="http-stream"
            if payload.get("stream") is True
            or "streamGenerateContent" in request.url.path
            or "response-stream" in request.url.path
            else "http-json",
            features=requested_features(payload),
        )
        resource_binding = None
        if protocol == "openai-responses":
            previous_response_id = payload.get("previous_response_id")
            if previous_response_id is not None:
                if not isinstance(previous_response_id, str):
                    raise GatewayAuthorizationError(
                        status_code=400,
                        code="gateway_request_invalid",
                        message="previous_response_id must be a string",
                    )
                resource_binding = await request.app.state.gateway_runtime.resources.authorize(
                    previous_response_id,
                    principal_id=principal.id,
                    route_id=route.id,
                    now=time.time(),
                )
        if resource_binding is not None:
            request.state.gateway_generation.validate_binding(resource_binding)
        target_protocol = protocol
        translated = protocol not in route.native_protocols
        translated_stream = False
        if translated:
            if route.translation != "qualified" or len(route.native_protocols) != 1:
                raise GatewayAuthorizationError(
                    status_code=400,
                    code="gateway_unsupported_capability",
                    message="Route does not qualify this protocol translation",
                )
            target_protocol = route.native_protocols[0]
            from headroom.proxy.gateway.protocols import translate

            translated_payload = translate(protocol, target_protocol, payload)
            translated_stream = translated_payload.get("stream") is True
            if target_protocol in ("openai-chat", "anthropic-messages"):
                translated_payload["model"] = route.upstream_model
            outbound_body = json.dumps(
                translated_payload, ensure_ascii=False, separators=(",", ":")
            ).encode()
        else:
            outbound_body = body

        upstream_paths = {
            "openai-chat": "/v1/chat/completions",
            "openai-responses": "/v1/responses",
            "anthropic-messages": "/v1/messages",
            "gemini-generate": f"/v1beta/models/{route.upstream_model}:generateContent",
        }
        upstream_path = upstream_paths.get(target_protocol, request.url.path)
        if (
            not translated
            and public_model is not None
            and route.public_model != route.upstream_model
        ):
            upstream_path = upstream_path.replace(route.public_model, route.upstream_model, 1)
        target = route_target(route, upstream_path)
        safe_query = [
            (name, value)
            for name, value in parse_qsl(request.url.query, keep_blank_values=True)
            if name.casefold() not in _QUERY_CREDENTIAL_NAMES
        ]
        if safe_query:
            target += "?" + urlencode(safe_query)
        if not translated and public_model is None:
            plan = GatewayDispatcher.resolve_body(
                body,
                public_model=route.public_model,
                upstream_model=route.upstream_model,
                declared_contract=route.body_contract,
            )
            outbound_body = plan.body
        reservation = await request.app.state.gateway_runtime.admission.reserve(
            AdmissionRequest(principal.id, estimated_cost=None)
        )
        selection = request.app.state.gateway_runtime.router.select(
            route,
            principal,
            resource_binding=resource_binding,
            eligible_accounts=request.state.gateway_generation.catalog.eligible_accounts(
                route,
                protocol=protocol,
                transport="http-stream"
                if payload.get("stream") is True
                or "streamGenerateContent" in request.url.path
                or "response-stream" in request.url.path
                else "http-json",
                features=requested_features(payload),
            ),
            authority_keys=dict(request.state.gateway_generation.authorities),
            target_key=request.state.gateway_generation.target_key(route.id),
        )
        lease = await request.state.gateway_generation.broker.acquire(
            route,
            account_ref=selection.account_ref,
        )
        client_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _REQUEST_HEADER_DENYLIST
        }
        request.state.gateway = GatewayRequestContext(
            principal=principal,
            route=route,
            ingress_protocol=protocol,
            request_id=request.headers.get("x-request-id", ""),
            generation=request.state.gateway_generation,
            catalog=request.state.gateway_generation.catalog,
            account_selection=selection,
        )
        upstream = await _send_managed(
            request, proxy, route, lease, target, client_headers, outbound_body
        )
        response_headers = _safe_response_headers(upstream)
        if translated and translated_stream:
            from headroom.proxy.gateway.protocols.events import translate_sse_stream

            handed_off = True
            return StreamingResponse(
                _finalize_stream(
                    translate_sse_stream(
                        target_protocol,
                        protocol,
                        _iter_upstream_bytes(upstream),
                        public_model=route.public_model,
                    ),
                    reservation,
                ),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type="text/event-stream",
                background=BackgroundTask(upstream.aclose),
            )
        if translated:
            from headroom.proxy.gateway.protocols import translate_response

            upstream_body = await upstream.aread()
            upstream_payload = json.loads(upstream_body)
            if not isinstance(upstream_payload, dict):
                raise GatewayAuthorizationError(
                    status_code=502,
                    code="gateway_upstream_invalid",
                    message="Upstream response must be a JSON object",
                )
            translated_response = translate_response(
                target_protocol,
                protocol,
                upstream_payload,
                public_model=route.public_model,
            )
            await upstream.aclose()
            await reservation.finalize(actual_cost=None)
            return JSONResponse(
                translated_response,
                status_code=upstream.status_code,
                headers=response_headers,
            )
        native_stream = payload.get("stream") is True
        if protocol == "openai-responses" and not native_stream:
            upstream_body = await upstream.aread()
            if 200 <= upstream.status_code < 300:
                try:
                    response_payload = json.loads(upstream_body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    response_payload = None
                if isinstance(response_payload, dict) and isinstance(
                    response_payload.get("id"), str
                ):
                    await request.app.state.gateway_runtime.resources.bind(
                        ResourceBinding(
                            provider_id=response_payload["id"],
                            principal_id=principal.id,
                            route_id=route.id,
                            account_ref=lease.account_ref,
                            adapter="openai-responses",
                            expires_at=time.time()
                            + request.state.gateway_generation.snapshot.limits.resource_ttl_seconds,
                            authority_fingerprint=request.state.gateway_generation.account_key(
                                lease.account_ref
                            ),
                            target_fingerprint=request.state.gateway_generation.target_key(
                                route.id
                            ),
                            generation=request.state.gateway_generation.number,
                        )
                    )
            await upstream.aclose()
            await reservation.finalize(actual_cost=None)
            return Response(
                content=upstream_body,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=None,
            )
        response_chunks: AsyncIterator[bytes] = _iter_upstream_bytes(upstream)
        if protocol == "openai-responses":
            response_chunks = _bind_response_stream(
                response_chunks,
                registry=request.app.state.gateway_runtime.resources,
                principal_id=principal.id,
                route_id=route.id,
                account_ref=lease.account_ref,
                generation=request.state.gateway_generation,
            )
        handed_off = True
        return StreamingResponse(
            _finalize_stream(response_chunks, reservation),
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
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "gateway_error",
                    "code": "gateway_upstream_error",
                    "message": "Upstream request failed",
                }
            },
        )
    finally:
        if reservation is not None and not handed_off:
            await reservation.release()


def gateway_model_catalog(
    request: Request, model_id: str | None = None, *, protocol: str | None = None
) -> JSONResponse:
    principal = request.state.gateway_principal
    catalog = request.state.gateway_generation.catalog
    if "models" not in principal.scopes:
        return JSONResponse(status_code=403, content={"error": {"code": "gateway_scope_forbidden"}})
    routes = catalog.visible_routes(principal)
    if protocol:
        routes = tuple(r for r in routes if protocol in r.protocols)
    if model_id is not None:
        routes = tuple(r for r in routes if r.id == model_id)
        if not routes:
            return JSONResponse(
                status_code=404, content={"error": {"code": "gateway_model_unavailable"}}
            )
    data = []
    for route in routes:
        item = {
            "id": route.id,
            "object": "model",
            "owned_by": "headroom-gateway",
            "headroom": {
                "protocols": route.protocols,
                "body_contract": route.body_contract,
                "catalog_revision": catalog.revision,
                "generation": catalog.generation,
                "provenance": route.provenance,
                "state": route.state,
                "capabilities": [
                    {"protocol": p, "transport": t, "features": f} for p, t, f in route.capabilities
                ],
                "cost_available": route.tariff_revision is not None,
                "tariff_revision": route.tariff_revision,
            },
        }
        if protocol == "gemini-generate":
            item["name"] = "models/" + route.id
            item["supportedGenerationMethods"] = ["generateContent"] + (
                ["streamGenerateContent"]
                if any(t == "http-stream" for p, t, _f in route.capabilities if p == protocol)
                else []
            )
        data.append(item)
    return JSONResponse(
        data[0]
        if model_id is not None
        else {"models": data}
        if protocol == "gemini-generate"
        else {"object": "list", "data": data}
    )


async def dispatch_stateful_response_http(
    request: Request,
    proxy: Any,
    sub_path: str,
) -> Response:
    """Authorize an existing Responses resource before any credential or I/O."""

    reservation: AdmissionReservation | None = None
    try:
        response_id = sub_path.split("/", 1)[0]
        principal = request.state.gateway_principal
        if "inference" not in principal.scopes:
            raise GatewayAuthorizationError(
                status_code=403,
                code="gateway_scope_denied",
                message="Gateway scope denied",
            )
        binding = await request.app.state.gateway_runtime.resources.authorize(
            response_id,
            principal_id=principal.id,
            route_id=None,
            allowed_route_ids=principal.routes,
            now=time.time(),
        )
        route = request.state.gateway_generation.catalog.route_for_id(binding.route_id)
        if route is None:
            raise GatewayAuthorizationError(
                status_code=404,
                code="gateway_resource_not_found",
                message="Stateful resource not found",
            )
        request.state.gateway_generation.validate_binding(binding)
        reservation = await request.app.state.gateway_runtime.admission.reserve(
            AdmissionRequest(principal.id, estimated_cost=None)
        )
        selection = request.app.state.gateway_runtime.router.select(
            route,
            principal,
            resource_binding=binding,
            eligible_accounts=request.state.gateway_generation.catalog.eligible_accounts(route),
            authority_keys=dict(request.state.gateway_generation.authorities),
            target_key=request.state.gateway_generation.target_key(route.id),
        )
        lease = await request.state.gateway_generation.broker.acquire(
            route,
            account_ref=selection.account_ref,
        )
        target = route_target(route, request.url.path)
        client_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _REQUEST_HEADER_DENYLIST
        }
        body = await request.body()
        upstream = await _send_managed(request, proxy, route, lease, target, client_headers, body)
        response_headers = _safe_response_headers(upstream)
        upstream_body = await upstream.aread()
        await upstream.aclose()
        if request.method == "DELETE" and 200 <= upstream.status_code < 300:
            await request.app.state.gateway_runtime.resources.delete(
                response_id,
                principal_id=principal.id,
                route_id=route.id,
            )
        await reservation.finalize(actual_cost=None)
        return Response(
            content=upstream_body,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=None,
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
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "gateway_error",
                    "code": "gateway_upstream_error",
                    "message": "Upstream request failed",
                }
            },
        )
    finally:
        if reservation is not None:
            await reservation.release()

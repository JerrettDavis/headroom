"""Gateway-owned OpenAI Responses WebSocket authorization and relay."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from headroom.proxy.gateway.admission import AdmissionRequest, AdmissionReservation
from headroom.proxy.gateway.auth import GatewayAuthorizer
from headroom.proxy.gateway.capabilities import requested_features
from headroom.proxy.gateway.config import RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal, GatewayRequestContext
from headroom.proxy.gateway.dispatch import route_target
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError, GatewayPublicError
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.transport import websocket_connection

_MAX_FRAME_BYTES = 1_048_576


def routed_response_create_frame(frame: str, route: RouteConfig) -> str:
    if route.body_contract == "strict-native" or route.public_model == route.upstream_model:
        return frame
    envelope = json.loads(frame)
    payload = envelope.get("response", envelope)
    payload["model"] = route.upstream_model
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def authorize_response_create_frame(
    frame: str,
    principal: GatewayPrincipal,
    authorizer: GatewayAuthorizer,
    *,
    expected_route_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Parse and authorize exactly one generation frame."""

    if len(frame.encode("utf-8")) > _MAX_FRAME_BYTES:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_frame_too_large",
            message="WebSocket generation frame exceeds the gateway bound",
        )
    try:
        envelope = json.loads(frame)
    except json.JSONDecodeError as exc:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_request_invalid",
            message="WebSocket generation frame is not valid JSON",
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("type") != "response.create":
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_frame_invalid",
            message="Expected a response.create generation frame",
        )
    payload = envelope.get("response", envelope)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_model_required",
            message="A string model is required for every generation turn",
        )
    route = authorizer.authorize(
        principal,
        scope="inference",
        protocol="openai-responses",
        public_model=payload["model"],
        transport="websocket",
        features=requested_features(payload),
    )
    if expected_route_id is not None and route.id != expected_route_id:
        raise GatewayAuthorizationError(
            status_code=403,
            code="gateway_websocket_affinity",
            message="WebSocket generation route does not match session affinity",
        )
    return route, payload


async def dispatch_native_responses_websocket(websocket: WebSocket, proxy: Any) -> None:
    """Relay a bounded session while authorizing every response.create turn."""

    runtime = websocket.app.state.gateway_runtime
    generation = runtime.capture()
    principal = websocket.scope.get("gateway_principal")
    if not isinstance(principal, GatewayPrincipal):
        await websocket.close(code=1008, reason="gateway authentication required")
        return
    session_principal_id = principal.id
    await websocket.accept()
    reservation: AdmissionReservation | None = None
    try:
        first_frame = await websocket.receive_text()
        generation = runtime.capture()
        current_principal = generation.authenticator.authenticate(websocket.headers)
        if current_principal.id != principal.id:
            raise GatewayAuthorizationError(
                status_code=403, code="gateway_route_forbidden", message="Session identity changed"
            )
        principal = current_principal
        route, first_payload = authorize_response_create_frame(
            first_frame,
            principal,
            generation.authorizer,
        )
        previous = first_payload.get("previous_response_id")
        binding = None
        if previous is not None:
            if not isinstance(previous, str):
                raise GatewayAuthorizationError(
                    status_code=400,
                    code="gateway_request_invalid",
                    message="previous_response_id must be a string",
                )
            binding = await runtime.resources.authorize(
                previous,
                principal_id=principal.id,
                route_id=route.id,
                now=time.time(),
            )
        if binding is not None:
            generation.validate_binding(binding)
        reservation = await runtime.admission.reserve(
            AdmissionRequest(principal.id, estimated_cost=None)
        )
        selection = runtime.router.select(
            route,
            principal,
            resource_binding=binding,
            eligible_accounts=generation.catalog.eligible_accounts(
                route,
                protocol="openai-responses",
                transport="websocket",
                features=requested_features(first_payload),
            ),
            authority_keys=dict(generation.authorities),
            target_key=generation.target_key(route.id),
        )
        broker = generation.broker
        lease = await broker.acquire(
            route,
            account_ref=selection.account_ref,
        )
        https_target = route_target(route, "/v1/responses")
        destination = await asyncio.to_thread(
            generation.egress_policy.authorize, lease, https_target, route=route
        )
        headers = build_managed_upstream_headers(
            dict(websocket.headers.items()),
            lease,
            https_target,
            method="GET",
            resolved_addresses=destination.addresses,
            route=route,
        )
        headers.pop("host", None)
        async with websocket_connection(destination, headers, generation.tls_context) as upstream:
            session_generation = generation
            websocket.scope["gateway"] = GatewayRequestContext(
                principal, route, "openai-responses", "", generation, generation.catalog, selection
            )
            first_frame = routed_response_create_frame(first_frame, route)
            await upstream.send(first_frame)

            async def client_to_upstream() -> None:
                nonlocal reservation, generation, principal
                while True:
                    frame = await websocket.receive_text()
                    try:
                        parsed = json.loads(frame)
                    except json.JSONDecodeError as exc:
                        raise GatewayAuthorizationError(
                            status_code=400,
                            code="gateway_request_invalid",
                            message="WebSocket frame is not valid JSON",
                        ) from exc
                    if isinstance(parsed, dict) and parsed.get("type") == "response.create":
                        generation = runtime.capture()
                        current_principal = generation.authenticator.authenticate(websocket.headers)
                        if current_principal.id != session_principal_id:
                            raise GatewayAuthorizationError(
                                status_code=403,
                                code="gateway_route_forbidden",
                                message="Session identity changed",
                            )
                        principal = current_principal
                        turn_route, turn_payload = authorize_response_create_frame(
                            frame,
                            principal,
                            generation.authorizer,
                            expected_route_id=route.id,
                        )
                        if (
                            generation.target_key(turn_route.id)
                            != session_generation.target_key(route.id)
                            or generation.account_key(lease.account_ref)
                            != session_generation.account_key(lease.account_ref)
                            or lease.account_ref
                            not in generation.catalog.eligible_accounts(
                                turn_route,
                                protocol="openai-responses",
                                transport="websocket",
                                features=requested_features(turn_payload),
                            )
                        ):
                            raise GatewayAuthorizationError(
                                status_code=403,
                                code="gateway_websocket_affinity",
                                message="Session authority is unavailable",
                            )
                        websocket.scope["gateway"] = GatewayRequestContext(
                            principal,
                            turn_route,
                            "openai-responses",
                            "",
                            generation,
                            generation.catalog,
                            selection,
                        )
                        if reservation is not None:
                            raise GatewayAuthorizationError(
                                status_code=409,
                                code="gateway_generation_in_progress",
                                message="A WebSocket generation is already in progress",
                            )
                        previous_id = turn_payload.get("previous_response_id")
                        if previous_id is not None:
                            if not isinstance(previous_id, str):
                                raise GatewayAuthorizationError(
                                    status_code=400,
                                    code="gateway_request_invalid",
                                    message="previous_response_id must be a string",
                                )
                            owned = await runtime.resources.authorize(
                                previous_id,
                                principal_id=principal.id,
                                route_id=turn_route.id,
                                now=time.time(),
                            )
                            generation.validate_binding(owned)
                            if owned.account_ref != lease.account_ref:
                                raise GatewayAuthorizationError(
                                    status_code=403,
                                    code="gateway_websocket_affinity",
                                    message="Stateful resource account does not match session affinity",
                                )
                        reservation = await runtime.admission.reserve(
                            AdmissionRequest(principal.id, estimated_cost=None)
                        )
                    elif not isinstance(parsed, dict) or parsed.get("type") != "response.cancel":
                        raise GatewayAuthorizationError(
                            status_code=400,
                            code="gateway_frame_invalid",
                            message="Unsupported WebSocket client frame",
                        )
                    if (
                        isinstance(parsed, dict)
                        and parsed.get("type") == "response.create"
                        and turn_route.body_contract == "routed-native"
                    ):
                        frame = routed_response_create_frame(frame, turn_route)
                    await upstream.send(frame)

            async def upstream_to_client() -> None:
                nonlocal reservation
                async for frame in upstream:
                    if isinstance(frame, bytes):
                        if len(frame) > _MAX_FRAME_BYTES:
                            raise GatewayAuthorizationError(
                                status_code=502,
                                code="gateway_frame_too_large",
                                message="Upstream WebSocket frame exceeded the gateway bound",
                            )
                        await websocket.send_bytes(frame)
                        continue
                    if len(frame.encode("utf-8")) > _MAX_FRAME_BYTES:
                        raise GatewayAuthorizationError(
                            status_code=502,
                            code="gateway_frame_too_large",
                            message="Upstream WebSocket frame exceeded the gateway bound",
                        )
                    with contextlib.suppress(json.JSONDecodeError):
                        event = json.loads(frame)
                        response = event.get("response") if isinstance(event, dict) else None
                        if isinstance(event, dict) and (
                            event.get("type") in {"error", "response.failed", "response.error"}
                            or isinstance(response, dict)
                            and response.get("error")
                        ):
                            raise GatewayPublicError(
                                status_code=502,
                                code="gateway_upstream_error",
                                message="Upstream request failed",
                            )
                        if (
                            isinstance(response, dict)
                            and event.get("type") == "response.created"
                            and isinstance(response.get("id"), str)
                        ):
                            await runtime.resources.bind(
                                ResourceBinding(
                                    provider_id=response["id"],
                                    principal_id=principal.id,
                                    route_id=route.id,
                                    account_ref=lease.account_ref,
                                    adapter="openai-responses",
                                    expires_at=time.time()
                                    + generation.snapshot.limits.resource_ttl_seconds,
                                    authority_fingerprint=generation.account_key(lease.account_ref),
                                    target_fingerprint=generation.target_key(route.id),
                                    generation=generation.number,
                                )
                            )
                        if (
                            isinstance(event, dict)
                            and event.get("type")
                            in {"response.completed", "response.failed", "response.cancelled"}
                            and reservation is not None
                        ):
                            await reservation.finalize(actual_cost=None)
                            reservation = None
                    await websocket.send_text(frame)

            client_task = asyncio.create_task(client_to_upstream())
            upstream_task = asyncio.create_task(upstream_to_client())
            tasks = {client_task, upstream_task}
            try:
                done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except GatewayPublicError as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "error": {"code": exc.code, "message": exc.message}}
            )
            await websocket.close(code=1008, reason=exc.code)
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "code": "gateway_upstream_error",
                        "message": "Upstream request failed",
                    },
                }
            )
            await websocket.close(code=1011, reason="gateway_upstream_error")
    finally:
        if reservation is not None:
            await reservation.release()

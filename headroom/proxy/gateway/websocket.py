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
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError, GatewayPublicError
from headroom.proxy.gateway.resources import ResourceBinding

_MAX_FRAME_BYTES = 1_048_576


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

    principal = websocket.scope.get("gateway_principal")
    if not isinstance(principal, GatewayPrincipal):
        await websocket.close(code=1008, reason="gateway authentication required")
        return
    await websocket.accept()
    reservation: AdmissionReservation | None = None
    try:
        first_frame = await websocket.receive_text()
        route, first_payload = authorize_response_create_frame(
            first_frame,
            principal,
            websocket.app.state.gateway_authorizer,
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
            binding = await websocket.app.state.gateway_resource_registry.authorize(
                previous,
                principal_id=principal.id,
                route_id=route.id,
                now=time.time(),
            )
        reservation = await websocket.app.state.gateway_admission.reserve(
            AdmissionRequest(principal.id, estimated_cost=None)
        )
        selection = websocket.app.state.gateway_account_router.select(
            route,
            principal,
            resource_binding=binding,
        )
        broker = websocket.app.state.gateway_credential_broker
        lease = await broker.acquire(
            route,
            account_ref=selection.account_ref,
        )
        https_target = route.upstream_origin.rstrip("/") + "/v1/responses"
        headers = build_managed_upstream_headers(
            dict(websocket.headers.items()),
            lease,
            https_target,
            method="GET",
        )
        headers.pop("host", None)
        ws_target = "wss://" + https_target.removeprefix("https://")

        import websockets

        async with websockets.connect(
            ws_target,
            additional_headers=headers,
            max_size=_MAX_FRAME_BYTES,
            ping_interval=20,
            ping_timeout=None,
        ) as upstream:
            await upstream.send(first_frame)

            async def client_to_upstream() -> None:
                nonlocal reservation
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
                        turn_route, turn_payload = authorize_response_create_frame(
                            frame,
                            principal,
                            websocket.app.state.gateway_authorizer,
                            expected_route_id=route.id,
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
                            owned = await websocket.app.state.gateway_resource_registry.authorize(
                                previous_id,
                                principal_id=principal.id,
                                route_id=turn_route.id,
                                now=time.time(),
                            )
                            if owned.account_ref != lease.account_ref:
                                raise GatewayAuthorizationError(
                                    status_code=403,
                                    code="gateway_websocket_affinity",
                                    message="Stateful resource account does not match session affinity",
                                )
                        reservation = await websocket.app.state.gateway_admission.reserve(
                            AdmissionRequest(principal.id, estimated_cost=None)
                        )
                    elif not isinstance(parsed, dict) or parsed.get("type") != "response.cancel":
                        raise GatewayAuthorizationError(
                            status_code=400,
                            code="gateway_frame_invalid",
                            message="Unsupported WebSocket client frame",
                        )
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
                        if (
                            isinstance(response, dict)
                            and event.get("type") == "response.created"
                            and isinstance(response.get("id"), str)
                        ):
                            await websocket.app.state.gateway_resource_registry.bind(
                                ResourceBinding(
                                    provider_id=response["id"],
                                    principal_id=principal.id,
                                    route_id=route.id,
                                    account_ref=lease.account_ref,
                                    adapter="openai-responses",
                                    expires_at=None,
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
                done, _pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
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
    finally:
        if reservation is not None:
            await reservation.release()

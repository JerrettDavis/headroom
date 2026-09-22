"""Explicit incremental event mappings for qualified directions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Literal

from headroom.proxy.gateway.errors import GatewayAuthorizationError


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: Literal["text_delta", "tool_argument_delta", "finish"]
    index: int
    data: str
    call_id: str | None = None
    tool_name: str | None = None


def translate_event(
    source_protocol: str,
    target_protocol: str,
    event: StreamEvent,
) -> tuple[dict[str, object], ...]:
    if (
        source_protocol == "openai-chat"
        and target_protocol == "anthropic-messages"
        and event.kind == "tool_argument_delta"
    ):
        return (
            {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "input_json_delta", "partial_json": event.data},
            },
        )
    raise GatewayAuthorizationError(
        status_code=400,
        code="gateway_unsupported_capability",
        message="Streaming translation direction is unsupported",
    )


async def translate_sse_stream(
    source_protocol: str,
    target_protocol: str,
    chunks: AsyncIterable[bytes],
    *,
    public_model: str,
) -> AsyncIterator[bytes]:
    """Translate complete SSE events while retaining only one bounded partial event."""

    if (source_protocol, target_protocol) != ("anthropic-messages", "openai-chat"):
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Streaming translation direction is unsupported",
        )
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
            if not data:
                continue
            try:
                event = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GatewayAuthorizationError(
                    status_code=502,
                    code="gateway_upstream_invalid",
                    message="Upstream stream event is invalid JSON",
                ) from exc
            if event.get("type") == "content_block_delta":
                delta = event.get("delta")
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    payload = {
                        "object": "chat.completion.chunk",
                        "model": public_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta.get("text", "")},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield (
                        b"data: "
                        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
                        + b"\n\n"
                    )
            elif event.get("type") == "message_stop":
                yield b"data: [DONE]\n\n"
    if buffer.strip():
        raise GatewayAuthorizationError(
            status_code=502,
            code="gateway_upstream_truncated",
            message="Upstream stream ended during an event",
        )

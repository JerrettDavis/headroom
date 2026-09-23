"""Explicit incremental event mappings for qualified directions."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
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
) -> AsyncGenerator[bytes, None]:
    """Translate complete SSE events while retaining only one bounded partial event."""

    if (source_protocol, target_protocol) != ("anthropic-messages", "openai-chat"):
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Streaming translation direction is unsupported",
        )
    from headroom.proxy.gateway.streaming import SSEFrames, event_data

    frames = SSEFrames(1_048_576)
    async for chunk in chunks:
        for raw_event in frames.feed(chunk):
            data = event_data(raw_event)
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
    if frames.pending:
        raise GatewayAuthorizationError(
            status_code=502,
            code="gateway_upstream_truncated",
            message="Upstream stream ended during an event",
        )

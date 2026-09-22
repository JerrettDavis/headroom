"""Directed, fail-closed protocol translation registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.protocols.anthropic import decode_anthropic, encode_anthropic
from headroom.proxy.gateway.protocols.gemini import decode_gemini, encode_gemini
from headroom.proxy.gateway.protocols.openai import decode_openai, encode_openai

Decoder = Callable[[dict[str, Any]], object]
Encoder = Callable[[Any], dict[str, Any]]

_DECODERS: dict[str, Decoder] = {
    "openai-chat": decode_openai,
    "anthropic-messages": decode_anthropic,
    "gemini-generate": decode_gemini,
}
_ENCODERS: dict[str, Encoder] = {
    "openai-chat": encode_openai,
    "anthropic-messages": encode_anthropic,
    "gemini-generate": encode_gemini,
}


def translate(
    source_protocol: str,
    target_protocol: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        decoder = _DECODERS[source_protocol]
        encoder = _ENCODERS[target_protocol]
    except KeyError as exc:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Translation direction is unsupported",
        ) from exc
    return encoder(decoder(payload))


def translate_response(
    source_protocol: str,
    target_protocol: str,
    payload: dict[str, Any],
    *,
    public_model: str,
) -> dict[str, Any]:
    """Translate a qualified non-streaming response without inventing usage."""

    if source_protocol == "openai-chat" and target_protocol == "anthropic-messages":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _unsupported_response()
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            _unsupported_response()
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, str):
            _unsupported_response()
        raw_finish = choice.get("finish_reason")
        if raw_finish is not None and not isinstance(raw_finish, str):
            _unsupported_response()
        stop_reason = (
            None
            if raw_finish is None
            else {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
                raw_finish
            )
        )
        if raw_finish is not None and stop_reason is None:
            _unsupported_response()
        openai_result: dict[str, Any] = {
            "id": payload.get("id"),
            "type": "message",
            "role": "assistant",
            "model": public_model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
        }
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                openai_result["usage"] = {"input_tokens": prompt, "output_tokens": completion}
        return openai_result

    if source_protocol == "gemini-generate" and target_protocol == "openai-chat":
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            _unsupported_response()
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            _unsupported_response()
        content = candidate.get("content")
        if not isinstance(content, dict) or set(content) != {"role", "parts"}:
            _unsupported_response()
        parts = content.get("parts")
        if content.get("role") != "model" or not isinstance(parts, list):
            _unsupported_response()
        gemini_text_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict) or set(part) != {"text"} or not isinstance(
                part["text"], str
            ):
                _unsupported_response()
            gemini_text_parts.append(part["text"])
        raw_finish = candidate.get("finishReason")
        if raw_finish is not None and not isinstance(raw_finish, str):
            _unsupported_response()
        finish = (
            None
            if raw_finish is None
            else {"STOP": "stop", "MAX_TOKENS": "length"}.get(raw_finish)
        )
        if raw_finish is not None and finish is None:
            _unsupported_response()
        gemini_result: dict[str, Any] = {
            "id": payload.get("responseId"),
            "object": "chat.completion",
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(gemini_text_parts)},
                    "finish_reason": finish,
                }
            ],
        }
        usage = payload.get("usageMetadata")
        if isinstance(usage, dict):
            prompt = usage.get("promptTokenCount")
            completion = usage.get("candidatesTokenCount")
            total = usage.get("totalTokenCount")
            if all(isinstance(value, int) for value in (prompt, completion, total)):
                gemini_result["usage"] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                }
        return gemini_result

    if source_protocol == "anthropic-messages" and target_protocol == "openai-chat":
        content = payload.get("content")
        if not isinstance(content, list):
            _unsupported_response()
        anthropic_text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or set(block) != {"type", "text"}:
                _unsupported_response()
            if block["type"] != "text" or not isinstance(block["text"], str):
                _unsupported_response()
            anthropic_text_parts.append(block["text"])
        raw_stop_reason = payload.get("stop_reason")
        if raw_stop_reason is not None and not isinstance(raw_stop_reason, str):
            _unsupported_response()
        stop_reason = (
            None
            if raw_stop_reason is None
            else {
                "end_turn": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }.get(raw_stop_reason)
        )
        if raw_stop_reason is not None and stop_reason is None:
            _unsupported_response()
        anthropic_result: dict[str, Any] = {
            "id": payload.get("id"),
            "object": "chat.completion",
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(anthropic_text_parts)},
                    "finish_reason": stop_reason,
                }
            ],
        }
        usage = payload.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                anthropic_result["usage"] = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
        return anthropic_result
    _unsupported_response()


def _unsupported_response() -> NoReturn:
    raise GatewayAuthorizationError(
        status_code=502,
        code="gateway_upstream_semantics_unsupported",
        message="Upstream response cannot be represented in the ingress protocol",
    )


__all__ = [
    "decode_anthropic",
    "decode_gemini",
    "decode_openai",
    "encode_anthropic",
    "encode_gemini",
    "encode_openai",
    "translate",
    "translate_response",
]

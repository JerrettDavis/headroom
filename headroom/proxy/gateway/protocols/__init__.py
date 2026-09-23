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
    if (source_protocol, target_protocol) not in {
        ("openai-chat", "anthropic-messages"),
        ("anthropic-messages", "openai-chat"),
        ("gemini-generate", "openai-chat"),
    }:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Translation direction is unsupported",
        )
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

    if source_protocol == "openai-chat" and target_protocol == "gemini-generate":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _unsupported_response()
        choice = choices[0]
        if (
            set(choice) - {"index", "message", "finish_reason", "logprobs"}
            or choice.get("logprobs") is not None
        ):
            _unsupported_response()
        message = choice.get("message")
        if not isinstance(message, dict) or set(message) - {"role", "content", "refusal"}:
            _unsupported_response()
        text = message.get("content")
        if message.get("role") != "assistant" or not isinstance(text, str):
            _unsupported_response()
        raw_finish = choice.get("finish_reason")
        if not isinstance(raw_finish, str):
            _unsupported_response()
        finish = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "SAFETY"}.get(
            raw_finish
        )
        if finish is None:
            _unsupported_response()
        if message.get("refusal"):
            _unsupported_response()
        result: dict[str, Any] = {
            "responseId": payload.get("id"),
            "modelVersion": public_model,
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                }
            ],
        }
        usage = payload.get("usage")
        if isinstance(usage, dict):
            result["usageMetadata"] = mapped_usage("openai-chat", "gemini-generate", usage)
        return result

    if source_protocol == "openai-chat" and target_protocol == "anthropic-messages":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _unsupported_response()
        choice = choices[0]
        if (
            set(choice) - {"index", "message", "finish_reason", "logprobs"}
            or choice.get("logprobs") is not None
        ):
            _unsupported_response()
        message = choice.get("message")
        if not isinstance(message, dict) or set(message) - {"role", "content", "refusal"}:
            _unsupported_response()
        content = message.get("content")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            if content:
                _unsupported_response()
            content = refusal
        if message.get("role") != "assistant" or not isinstance(content, str):
            _unsupported_response()
        raw_finish = choice.get("finish_reason")
        if raw_finish is not None and not isinstance(raw_finish, str):
            _unsupported_response()
        stop_reason = (
            None
            if raw_finish is None
            else {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(
                raw_finish
            )
        )
        if refusal:
            stop_reason = "refusal"
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
            openai_result["usage"] = mapped_usage(source_protocol, target_protocol, usage)
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
            if (
                not isinstance(part, dict)
                or set(part) != {"text"}
                or not isinstance(part["text"], str)
            ):
                _unsupported_response()
            gemini_text_parts.append(part["text"])
        raw_finish = candidate.get("finishReason")
        if raw_finish is not None and not isinstance(raw_finish, str):
            _unsupported_response()
        finish = (
            None
            if raw_finish is None
            else {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}.get(
                raw_finish
            )
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
                "refusal": "content_filter",
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
            anthropic_result["usage"] = mapped_usage(source_protocol, target_protocol, usage)
        return anthropic_result
    _unsupported_response()


def mapped_usage(source: str, target: str, usage: dict[str, Any]) -> dict[str, Any]:
    """Map documented token counts; absent counts stay absent, never zero-filled."""
    result: dict[str, Any] = {}
    if source == "openai-chat":
        fields = (
            {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}
            if target == "anthropic-messages"
            else {
                "prompt_tokens": "promptTokenCount",
                "completion_tokens": "candidatesTokenCount",
                "total_tokens": "totalTokenCount",
            }
        )
    else:
        fields = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}
    for source_key, target_key in fields.items():
        if source_key in usage:
            value = usage[source_key]
            if type(value) is not int or value < 0:
                _unsupported_response()
            result[target_key] = value
    if source == "anthropic-messages" and "prompt_tokens" in result:
        for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if name in usage:
                value = usage[name]
                if type(value) is not int or value < 0:
                    _unsupported_response()
                result["prompt_tokens"] += value
        if "cache_read_input_tokens" in usage:
            result["prompt_tokens_details"] = {"cached_tokens": usage["cache_read_input_tokens"]}
        if "completion_tokens" in result:
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    if source == "openai-chat":
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached is not None:
            if type(cached) is not int or cached < 0 or cached > usage.get("prompt_tokens", 0):
                _unsupported_response()
            if target == "gemini-generate":
                result["cachedContentTokenCount"] = cached
            else:
                result["input_tokens"] -= cached
                result["cache_read_input_tokens"] = cached
    return result


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

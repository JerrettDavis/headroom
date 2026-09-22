"""Qualified Gemini GenerateContent request adapter."""

from __future__ import annotations

from typing import Any, Literal, cast

from headroom.proxy.gateway.protocols.content import ContentBlock, Conversation, Message
from headroom.proxy.gateway.protocols.openai import _reject_unknown, _unsupported

_FIELDS = frozenset({"contents", "systemInstruction", "generationConfig"})


def decode_gemini(payload: dict[str, Any]) -> Conversation:
    _reject_unknown(payload, _FIELDS)
    system_instruction = payload.get("systemInstruction")
    system: tuple[ContentBlock, ...] = ()
    if system_instruction is not None:
        if not isinstance(system_instruction, dict) or set(system_instruction) != {"parts"}:
            _unsupported("systemInstruction")
        system = _parts(system_instruction["parts"])
    raw_contents = payload.get("contents")
    if not isinstance(raw_contents, list):
        _unsupported("contents")
    messages: list[Message] = []
    for item in raw_contents:
        if not isinstance(item, dict) or set(item) != {"role", "parts"}:
            _unsupported("content")
        role = {"model": "assistant", "user": "user"}.get(item["role"])
        if role is None:
            _unsupported("content role")
        messages.append(
            Message(
                role=cast(Literal["user", "assistant"], role),
                content=_parts(item["parts"]),
            )
        )
    generation = payload.get("generationConfig", {})
    if not isinstance(generation, dict):
        _unsupported("generationConfig")
    _reject_unknown(generation, frozenset({"maxOutputTokens", "temperature"}))
    max_tokens = generation.get("maxOutputTokens")
    temperature = generation.get("temperature")
    return Conversation(
        model=None,
        system=system,
        messages=tuple(messages),
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
    )


def encode_gemini(conversation: Conversation) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contents": [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": block.text} for block in message.content],
            }
            for message in conversation.messages
        ]
    }
    if conversation.system:
        result["systemInstruction"] = {
            "parts": [{"text": block.text} for block in conversation.system]
        }
    generation: dict[str, Any] = {}
    if conversation.max_tokens is not None:
        generation["maxOutputTokens"] = conversation.max_tokens
    if conversation.temperature is not None:
        generation["temperature"] = conversation.temperature
    if generation:
        result["generationConfig"] = generation
    return result


def _parts(value: object) -> tuple[ContentBlock, ...]:
    if not isinstance(value, list):
        _unsupported("parts")
    blocks: list[ContentBlock] = []
    for part in value:
        if not isinstance(part, dict) or set(part) != {"text"} or not isinstance(part["text"], str):
            _unsupported("part")
        blocks.append(ContentBlock(kind="text", text=part["text"]))
    return tuple(blocks)

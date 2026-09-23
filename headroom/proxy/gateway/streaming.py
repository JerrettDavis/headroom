"""Pull-driven, bounded HTTP entity observation without native reserialization."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from headroom.proxy.gateway.config import LimitsConfig
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.usage import UsageObservation, normalize_usage


def stream_error(reason: str) -> GatewayPublicError:
    return GatewayPublicError(
        status_code=502, code="gateway_stream_" + reason, message="Upstream stream failed"
    )


class SSEFrames:
    """One bounded partial frame; output retains line endings and field bytes."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.pending = bytearray()

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline + 1
            if len(self.pending) + end - start > self.maximum:
                raise stream_error("too_large")
            self.pending.extend(chunk[start:end])
            start = end
            if self.pending.endswith((b"\n\n", b"\n\r\n")) or self.pending in (b"\n", b"\r\n"):
                frame = bytes(self.pending)
                self.pending.clear()
                yield frame


def event_data(frame: bytes) -> str:
    try:
        lines = frame.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise stream_error("malformed") from None
    if any(
        line.partition(":")[0] == "event" and line.partition(":")[2].removeprefix(" ") == "error"
        for line in lines
    ):
        raise stream_error("upstream_error")
    return "\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:"))


class StreamObserver:
    def __init__(self, protocol: str, limits: LimitsConfig, deadline: float) -> None:
        self.protocol, self.limits, self.deadline = protocol, limits, deadline
        self.frames = SSEFrames(limits.max_frame_bytes)
        self.usage = UsageObservation()
        self.terminal: str | None = None
        self.last_event: dict[str, Any] | None = None
        self._choices: dict[int, bool] = {}
        self._content_at = time.monotonic()
        self._partial_at: float | None = None

    def inspect(self, frame: bytes) -> bool:
        self.last_event = None
        data = event_data(frame)
        if not data:
            return False
        if data == "[DONE]":
            if (
                self.protocol != "openai-chat"
                or not self._choices
                or not all(self._choices.values())
            ):
                raise stream_error("truncated")
            self.terminal = "success"
            return True
        try:
            event = json.loads(data)
        except (ValueError, RecursionError):
            raise stream_error("malformed") from None
        if not isinstance(event, dict):
            raise stream_error("malformed")
        self.last_event = event
        self.usage = normalize_usage(self.protocol, event, previous=self.usage)
        kind = event.get("type")
        if event.get("error") or kind in {
            "error",
            "response.failed",
            "response.error",
            "response.incomplete",
            "response.cancelled",
            "refusal",
        }:
            self.terminal = "failed"
            raise stream_error("upstream_error")
        if kind in {"ping", "heartbeat"}:
            return False
        if self.protocol == "openai-chat":
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                raise stream_error("malformed")
            for choice in choices:
                if not isinstance(choice, dict) or type(choice.get("index", 0)) is not int:
                    raise stream_error("malformed")
                self._choices[choice.get("index", 0)] = choice.get("finish_reason") is not None
                if choice.get("finish_reason") in {"content_filter", "length"} or choice.get(
                    "delta", {}
                ).get("refusal"):
                    self.terminal = "failed"
                    raise stream_error("upstream_error")
        elif self.protocol == "openai-responses" and kind == "response.completed":
            self.terminal = "success"
        elif self.protocol == "anthropic-messages":
            if event.get("delta", {}).get("stop_reason") in {
                "refusal",
                "max_tokens",
                "model_context_window_exceeded",
            }:
                self.terminal = "failed"
                raise stream_error("upstream_error")
            if kind == "message_stop":
                self.terminal = "success"
        elif self.protocol in {"gemini-generate", "vertex-generate"}:
            candidates = event.get("candidates", [])
            if candidates and all(
                isinstance(c, dict) and c.get("finishReason") for c in candidates
            ):
                if any(c["finishReason"] != "STOP" for c in candidates):
                    self.terminal = "failed"
                    raise stream_error("upstream_error")
                self.terminal = "success"
        return True

    async def observe(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        iterator = chunks.__aiter__()
        try:
            while True:
                due = min(self.deadline, self._content_at + self.limits.stream_content_idle_seconds)
                if self._partial_at is not None:
                    due = min(due, self._partial_at + self.limits.partial_frame_seconds)
                try:
                    chunk = await asyncio.wait_for(anext(iterator), max(0, due - time.monotonic()))
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise stream_error("timeout") from None
                had_partial = bool(self.frames.pending)
                for frame in self.frames.feed(chunk):
                    self._partial_at = None
                    if self.inspect(frame):
                        self._content_at = time.monotonic()
                    yield frame
                    if self.terminal == "success":
                        return
                if self.frames.pending and (not had_partial or self._partial_at is None):
                    self._partial_at = time.monotonic()
            if self.frames.pending or self.terminal is None:
                raise stream_error("truncated")
        finally:
            await iterator.aclose()  # type: ignore[attr-defined]


async def observed_body(chunks: AsyncIterator[bytes], *, maximum: int, deadline: float) -> bytes:
    body = bytearray()
    iterator = chunks.__aiter__()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(anext(iterator), max(0, deadline - time.monotonic()))
            except StopAsyncIteration:
                return bytes(body)
            except TimeoutError:
                raise stream_error("timeout") from None
            if len(body) + len(chunk) > maximum:
                raise stream_error("too_large")
            body.extend(chunk)
    finally:
        await iterator.aclose()  # type: ignore[attr-defined]

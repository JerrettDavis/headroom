"""T036–T040: real process, verified TLS and client/upstream barriers."""

import threading

import pytest

from tests.unified_gateway.process.http_harness import frame, http_process, stream_headers

CHAT_FIRST = 'data: {"choices":[{"index":0,"delta":{"content":"雪","tool_calls":[{"function":{"arguments":"{}"}}]},"finish_reason":null}]}\r\n\r\n'.encode()
CHAT_END = b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\r\n\r\ndata: [DONE]\r\n\r\n'


@pytest.mark.parametrize("translated", [False, True])
def test_first_event_precedes_upstream_completion_barrier(local_pki, tmp_path, translated):  # noqa: F811
    release = threading.Event()
    finished = threading.Event()
    first = (
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\r\n\r\n'
        if translated
        else CHAT_FIRST
    )
    final = (
        b'data: {"type":"message_delta","usage":{"output_tokens":2}}\r\n\r\ndata: {"type":"message_stop"}\r\n\r\n'
        if translated
        else CHAT_END
    )

    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, first)
        assert release.wait(10)
        frame(upstream, final)
        finished.set()

    def configure(raw):
        if translated:
            route = raw["routes"][0]
            route.update(
                native_protocols=["anthropic-messages"],
                ingress_protocols=["openai-chat"],
                translation="qualified",
                capabilities={"openai-chat": {"http-stream": {"features": ["text"]}}},
            )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "fixture-model", "messages": [], "stream": True},
            ) as response:
                assert response.status_code == 200
                iterator = response.iter_raw()
                initial = next(iterator)
                assert b"hello" in initial if translated else CHAT_FIRST == initial
                assert not finished.is_set()
                release.set()
                remainder = b"".join(iterator)
                assert b"[DONE]" in remainder
            probe = client.get("/__test/idle").json()
            assert probe["totals"]["logical_requests"] == 1
            assert probe["totals"]["attempts"] == 1
            assert len(calls) == 1
            assert probe["active"] == 0
            if translated:
                assert probe["ledger"]["known_micro_usd"] == 4
                assert probe["ledger"]["unknown_charge_count"] == 1
            else:
                assert probe["ledger"]["known_micro_usd"] == 7
        finally:
            release.set()


def test_fragmented_unicode_tool_events_cross_real_sockets(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        stream_headers(upstream)
        for byte in CHAT_FIRST + CHAT_END:
            frame(upstream, bytes([byte]))

    with http_process(local_pki, tmp_path, handler) as (client, _, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert response.content == CHAT_FIRST + CHAT_END
        assert client.get("/__test/idle").json()["ledger"]["known_micro_usd"] == 7


@pytest.mark.parametrize(
    "tail", [b"", b"data: {", b"data: \xe9", b"data: {invalid}\n\n", b"data: " + b"x" * 1100]
)
def test_eof_without_terminal_is_failed(local_pki, tmp_path, tail):  # noqa: F811
    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, CHAT_FIRST + tail)

    with http_process(
        local_pki,
        tmp_path,
        handler,
        configure=lambda raw: raw["limits"].update(max_frame_bytes=1024),
    ) as (client, calls, output, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert b'"code":"gateway_upstream_error"' in response.content
        assert b"[DONE]" not in response.content
        probe = client.get("/__test/idle").json()
        assert probe["active"] == 0
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert len(calls) == 1
        assert "fixture-key" not in "".join(output)


@pytest.mark.parametrize(
    "prefix,absolute", [(b": ping\n\n", False), (b"data: {", False), (b": ping\n\n", True)]
)
def test_heartbeat_and_partial_line_deadlines_fire_without_new_bytes(
    local_pki, tmp_path, prefix, absolute
):  # noqa: F811
    closed = threading.Event()

    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, prefix)
        upstream.connection.settimeout(4)
        assert upstream.connection.recv(1) == b""
        closed.set()

    def configure(raw):
        if absolute:
            raw["limits"].update(
                request_deadline_seconds=0.3,
                stream_content_idle_seconds=10,
                partial_frame_seconds=10,
            )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, _, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert b'"code":"gateway_upstream_error"' in response.content
        assert closed.wait(5)
        assert client.get("/__test/idle").json()["active"] == 0


@pytest.mark.parametrize("expose", [False, True])
def test_disconnect_cancels_upstream_and_finalizes_once(local_pki, tmp_path, expose):  # noqa: F811
    closed = threading.Event()

    def handler(upstream):
        stream_headers(upstream)
        if expose:
            frame(upstream, CHAT_FIRST)
        upstream.connection.settimeout(5)
        try:
            assert upstream.connection.recv(1) == b""
        except (OSError, ConnectionResetError):
            pass
        closed.set()

    with http_process(local_pki, tmp_path, handler) as (client, calls, _, _):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fixture-model", "messages": [], "stream": True},
        ) as response:
            if expose:
                assert next(response.iter_raw()) == CHAT_FIRST
        assert closed.wait(6)
        probe = client.get("/__test/idle").json()
        assert probe["active"] == 0
        assert probe["queued"] == 0
        assert probe["totals"]["attempts"] == len(calls) == 1
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30

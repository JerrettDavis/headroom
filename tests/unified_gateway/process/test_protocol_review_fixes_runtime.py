"""Review regressions over actual gateway/TLS upstream sockets."""

import json

import pytest

from tests.unified_gateway.process.http_harness import (
    frame,
    http_process,
    json_response,
    stream_headers,
)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.refusal.delta", "delta": "REFUSAL_SECRET"},
        {"type": "response.refusal.done", "refusal": "REFUSAL_SECRET"},
        {
            "type": "response.content_part.added",
            "part": {"type": "refusal", "refusal": "REFUSAL_SECRET"},
        },
        {
            "type": "response.content_part.done",
            "part": {"type": "refusal", "refusal": "REFUSAL_SECRET"},
        },
        {
            "type": "response.output_item.added",
            "item": {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "REFUSAL_SECRET"}],
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "REFUSAL_SECRET"}],
            },
        },
    ],
    ids=["delta", "done", "part-added", "part-done", "item-added", "item-done"],
)
def test_native_responses_refusal_lifecycle_never_leaks_and_retains_terminal_usage(
    local_pki, tmp_path, event
):
    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, b"data: " + json.dumps(event).encode() + b"\n\n")
        frame(
            upstream,
            b'data: {"type":"response.completed","response":{"id":"refused","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2}}}\n\n',
        )

    with http_process(local_pki, tmp_path, handler) as (client, calls, _, _):
        with client.stream(
            "POST", "/v1/responses", json={"model": "fixture-model", "input": "hi", "stream": True}
        ) as response:
            chunks = list(response.iter_bytes())
        assert chunks
        for chunk in chunks:
            assert b"REFUSAL_SECRET" not in chunk
            assert b"response.refusal" not in chunk
        wire = b"".join(chunks)
        assert len(wire) < 512 and b"gateway_upstream_error" in wire
        assert b"response.completed" not in wire
        probe = client.get("/__test/idle").json()
        assert len(calls) == 1
        assert probe["operations"][0]["terminal"] == "failed"
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert probe["active"] == probe["queued"] == 0


@pytest.mark.parametrize(
    "protocol,alias,streamed",
    [
        ("gemini-generate", "models", False),
        ("gemini-generate", "v1beta", True),
        ("vertex-generate", "projects", False),
        ("vertex-generate", "p", True),
        ("vertex-generate", "locations", False),
        ("vertex-generate", "r", True),
        ("vertex-generate", "publishers", False),
        ("vertex-generate", "google", True),
        ("vertex-generate", "models", False),
        ("vertex-generate", "v1", True),
        ("bedrock-invoke", "model", False),
        ("gemini-generate", "streamGenerateContent", False),
        ("vertex-generate", "response-stream", False),
    ],
)
def test_native_path_alias_changes_only_model_component(
    local_pki, tmp_path, protocol, alias, streamed
):
    method = "streamGenerateContent" if streamed else "generateContent"
    stem = (
        "/v1beta/models/"
        if protocol == "gemini-generate"
        else "/v1/projects/p/locations/r/publishers/google/models/"
    )
    path = stem + alias + ":" + method
    expected = "/private/google/" + stem.split("/", 2)[2] + "provider-model:" + method
    if protocol == "bedrock-invoke":
        path, expected = "/model/model/invoke", "/model/provider-model/invoke"

    def configure(raw):
        prefix = "/model/" if protocol == "bedrock-invoke" else "/private/google/"
        raw["routes"][0].update(
            public_model=alias,
            upstream_model="provider-model",
            upstream_path_prefix=prefix,
            ingress_protocols=[protocol],
            native_protocols=[protocol],
            capabilities={
                protocol: {"http-stream" if streamed else "http-json": {"features": ["text"]}}
            },
        )
        raw["credentials"][0]["allowed_path_prefixes"] = [prefix]

    def handler(upstream):
        if streamed:
            stream_headers(upstream)
            frame(upstream, b'data: {"candidates":[{"index":0,"finishReason":"STOP"}]}\n\n')
        else:
            json_response(upstream, b"{}")

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post(path + ("?alt=sse" if streamed else ""), json={"contents": []})
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == expected + ("?alt=sse" if streamed else "")

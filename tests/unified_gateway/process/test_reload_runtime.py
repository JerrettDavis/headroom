"""Reload/revocation oracles through actual admitted HTTP and WS work."""

import concurrent.futures
import json
import threading

import httpx
import pytest
from websockets.exceptions import ConnectionClosed

from tests.unified_gateway.process.http_harness import http_process, json_response
from tests.unified_gateway.process.test_websocket_runtime import ADMIN, CREATE, configure, socket


def test_atomic_reload_preserves_inflight_snapshot_and_ledger(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        if json.loads(upstream.body).get("input") == "a":
            accepted.set()
            assert release.wait(10)
        json_response(
            upstream,
            json.dumps(
                {
                    "id": "resp_" + json.loads(upstream.body)["input"],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                }
            ).encode(),
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, raw):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                assert accepted.wait(5)
                raw["routes"][0]["pricing"].update(revision="b", output_usd_per_million="5")
                (tmp_path / "http-runtime.json").write_text(json.dumps(raw))
                assert client.post("/admin/gateway/reload", headers=ADMIN).json()["applied"]
                assert (
                    client.post(
                        "/v1/responses", json={"model": "fixture-model", "input": "b"}
                    ).status_code
                    == 200
                )
                during = client.get("/__test/probe").json()
                assert during["ledger"]["reserved_micro_usd"] == 30
                assert during["ledger"]["known_micro_usd"] == 13
                assert not first.done()
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 20
        assert [op["generation"] for op in probe["operations"]] == [1, 2]
        assert [op["tariff"] for op in probe["operations"]] == ["fixture", "b"]
        assert len(calls) == 2


@pytest.mark.parametrize("phase", ["queued", "acquire"])
@pytest.mark.parametrize("selector", ["principal_id", "route_id", "account_id", "grant"])
def test_revoke_denies_waiters_and_cancels_http_ws_and_acquire(
    local_pki, tmp_path, phase, selector
):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)
        try:
            json_response(upstream, b'{"usage":{"input_tokens":3,"output_tokens":2}}')
        except OSError:
            pass

    def policy(raw):
        configure(raw)
        raw["admission"].update(max_concurrency=1, queue_limit=2, queue_timeout_seconds=4)
        raw["client_auth"]["principals"][0]["admission"] = {
            "max_concurrency": 1,
            "queue_limit": 2,
            "queue_timeout_seconds": 4,
        }
        spare = dict(raw["routes"][0], id="spare", public_model="spare")
        raw["routes"].append(spare)
        raw["client_auth"]["principals"][0]["routes"].append("spare")

    with http_process(
        local_pki, tmp_path, handler, configure=policy, acquire_barrier=phase == "acquire"
    ) as (client, calls, _, raw):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                if phase == "queued":
                    assert accepted.wait(5)
                else:
                    assert client.post("/__test/acquire-entered").json()["entered"]
                with socket(client) as ws:
                    ws.send(CREATE)
                    assert client.post("/__test/queue-entered").json()["queued"] == 1
                    if selector == "grant":
                        raw["client_auth"]["principals"][0]["routes"] = ["spare"]
                        (tmp_path / "http-runtime.json").write_text(json.dumps(raw))
                        response = client.post("/admin/gateway/reload", headers=ADMIN)
                    else:
                        value = {
                            "principal_id": raw["client_auth"]["principals"][0]["id"],
                            "route_id": raw["routes"][0]["id"],
                            "account_id": raw["credentials"][0]["id"],
                        }[selector]
                        response = client.post(
                            "/admin/gateway/revoke", headers=ADMIN, json={selector: value}
                        )
                    assert response.status_code == 200, response.text
                    try:
                        assert json.loads(ws.recv(timeout=3))["type"] == "error"
                    except ConnectionClosed:
                        pass
                client.post("/__test/release-acquire", headers=ADMIN)
                probe = client.get("/__test/idle", headers=ADMIN).json()
                assert probe["active"] == probe["queued"] == probe["owned"] == 0
                assert len(calls) == (1 if phase == "queued" else 0)
                assert probe["ledger"]["unknown_charge_count"] == (1 if phase == "queued" else 0)
            finally:
                release.set()
            try:
                assert pending.result(timeout=5).status_code != 200
            except httpx.HTTPError:
                pass

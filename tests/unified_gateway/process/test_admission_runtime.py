"""HTTP subset of T052–T054: shared ledger and every accepted attempt."""

import concurrent.futures
import copy
import threading

import pytest

from tests.unified_gateway.process.http_harness import http_process, json_response


def test_two_routes_race_one_remaining_budget_reservation(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)
        json_response(upstream, b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}')

    def configure(raw):
        raw["admission"].update(budget_usd="0.000030", unknown_cost_policy="block")
        route = copy.deepcopy(raw["routes"][0])
        route.update(id="second", public_model="second-model", upstream_model="second-model")
        raw["routes"].append(route)
        raw["client_auth"]["principals"][0]["routes"].append("second")

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
            )
            try:
                assert accepted.wait(5), pending.result(timeout=1).text
                denied = client.post(
                    "/v1/chat/completions", json={"model": "second-model", "messages": []}
                )
                assert denied.status_code == 403
                assert "budget" in denied.json()["error"]["code"]
                assert len(calls) == 1
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["active"] == probe["queued"] == 0


def test_retry_accounting_keeps_known_and_unknown_attempts(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        # Partial usage must keep the remainder of the accepted reservation.
        json_response(upstream, b'{"usage":{"prompt_tokens":3}}')

    with http_process(local_pki, tmp_path, handler, fail_connect=True) as (client, _, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert response.status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["totals"]["attempts"] == 2
        assert probe["ledger"]["known_micro_usd"] == 3
        assert probe["ledger"]["unresolved_micro_usd"] == 27
        assert probe["ledger"]["unknown_charge_count"] == 1


def test_strict_budget_resource_read_does_not_reserve_generation_twice(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1","usage":{"input_tokens":3,"output_tokens":2}}')

    def configure(raw):
        raw["admission"].update(budget_usd="0.000030", unknown_cost_policy="block")

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post("/v1/responses", json={"model": "fixture-model", "input": "hello"})
        assert response.status_code == 200
        assert client.get("/v1/responses/resp_1").status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert len(calls) == probe["totals"]["attempts"] == 2


def test_noisy_tenant_cannot_consume_reserved_other_tenant_slots(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        if not accepted.is_set():
            accepted.set()
            assert release.wait(10)
        json_response(upstream, b'{"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    def configure(raw):
        raw["admission"]["max_concurrency"] = 2
        other = copy.deepcopy(raw["client_auth"]["principals"][0])
        other.update(
            id="other",
            secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B",
            admission={"reserved_concurrency": 1},
        )
        raw["client_auth"]["principals"].append(other)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
            )
            try:
                assert accepted.wait(5)
                denied = client.post(
                    "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
                )
                assert denied.status_code == 429
                other = client.post(
                    "/v1/chat/completions",
                    headers={"authorization": "Bearer client-b"},
                    json={"model": "fixture-model", "messages": []},
                )
                assert other.status_code == 200
                assert len(calls) == 2
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        assert client.get("/__test/idle").json()["active"] == 0


@pytest.mark.parametrize(
    "body,status",
    [(b'{"error":{"message":"secret"}}', 400), (b"x" * 2000, 200), (b"{invalid", 200)],
)
def test_terminal_path_leak_matrix(local_pki, tmp_path, body, status):  # noqa: F811
    def handler(upstream):
        json_response(upstream, body, status)

    with http_process(
        local_pki,
        tmp_path,
        handler,
        configure=lambda raw: raw["limits"].update(max_observed_json_bytes=1024),
    ) as (client, calls, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert response.status_code == 502
        assert b"secret" not in response.content
        probe = client.get("/__test/idle").json()
        assert probe["active"] == probe["queued"] == 0
        assert probe["totals"]["attempts"] == len(calls) == 1
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30

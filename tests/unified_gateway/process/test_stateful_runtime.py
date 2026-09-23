"""HTTP ownership/affinity subset; WebSocket turn cases remain a later batch."""

import copy

from tests.unified_gateway.process.http_harness import http_process, json_response


def test_other_principal_cannot_read_continue_cancel_delete(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1","usage":{"input_tokens":3,"output_tokens":2}}')

    def configure(raw):
        other = copy.deepcopy(raw["client_auth"]["principals"][0])
        other.update(id="other", secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B")
        raw["client_auth"]["principals"].append(other)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "input": "hello"}
            ).status_code
            == 200
        )
        headers = {"authorization": "Bearer client-b"}
        assert client.get("/v1/responses/resp_1", headers=headers).status_code == 404
        assert client.post("/v1/responses/resp_1/cancel", headers=headers).status_code == 404
        assert client.delete("/v1/responses/resp_1", headers=headers).status_code == 404
        assert (
            client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "fixture-model", "previous_response_id": "resp_1"},
            ).status_code
            == 404
        )
        assert len(calls) == 1


def test_affinity_does_not_migrate_when_owner_unavailable(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1"}')

    def configure(raw):
        account = raw["credentials"][0]
        account.update(owner_group="owner", billing_group="billing")
        other = copy.deepcopy(account)
        other["id"] = "other-key"
        raw["credentials"].append(other)
        raw["routes"][0]["credentials"].append("other-key")
        raw["routes"][0]["catalog"]["entitlements"]["other-key"] = "allowed"

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "input": "hello"}
            ).status_code
            == 200
        )
        assert client.post("/__test/cool/internal-key/internal-native").status_code == 200
        assert client.get("/v1/responses/resp_1").status_code == 503
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "previous_response_id": "resp_1"}
            ).status_code
            == 503
        )
        assert len(calls) == 1

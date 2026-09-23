"""Actual Uvicorn test process, with one explicit fake-server destination pin."""

import sys
from dataclasses import replace
from pathlib import Path

import httpx
import uvicorn
from fastapi import Request

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.credential_sources.environment import EnvironmentCredentialSource
from headroom.proxy.gateway.egress import EgressPolicy
from headroom.proxy.gateway.transport import PinnedHTTPTransport, tls_context
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def run(path: Path) -> None:
    snapshot = GatewayConfigSnapshot.load(path)
    app = create_app(ProxyConfig(gateway=snapshot, gateway_config_path=path))
    runtime = app.state.gateway_runtime
    observations = []
    counts = {"identity": 0}
    now = [1000.0]
    original_acquire = EnvironmentCredentialSource.acquire

    async def acquire(source, *, now):
        counts["identity"] += 1
        return await original_acquire(source, now=now)

    EnvironmentCredentialSource.acquire = acquire

    def resolve(host, port):
        assert host == "llm.internal.example"
        assert port == httpx.URL(snapshot.routes[0].upstream_origin).port
        return ("10.111.0.10",)

    class LocalFixtureTransport(PinnedHTTPTransport):
        async def handle_async_request(self, request):
            destination = request.extensions["gateway_destination"]
            assert destination.hostname == "llm.internal.example"
            assert destination.addresses == ("10.111.0.10",)
            # The real policy/header boundary has approved the configured private
            # destination. Only this test transport maps that exact endpoint to
            # the fixture socket; TLS still verifies llm.internal.example.
            request.extensions["gateway_destination"] = replace(
                destination, addresses=("127.0.0.1",)
            )
            return await super().handle_async_request(request)

    runtime.dependencies.egress_policy = EgressPolicy(resolver=resolve)
    runtime.dependencies.clock = lambda: now[0]
    runtime.dependencies.http_client = httpx.AsyncClient(
        transport=LocalFixtureTransport(
            verify=tls_context(snapshot),
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ),
        trust_env=False,
        follow_redirects=False,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        response = await call_next(request)
        context = getattr(request.state, "gateway", None)
        if context is not None:
            observations.append(
                {
                    "generation": context.generation.number,
                    "catalog_revision": context.catalog.revision,
                    "tariff": context.route.pricing.revision if context.route.pricing else None,
                }
            )
        return response

    @app.get("/__test/probe")
    async def probe():
        return {**counts, "observations": observations}

    @app.post("/__test/clock/{value}")
    async def advance(value: float):
        now[0] = value
        return {"now": now[0]}

    @app.post("/__test/expire-source")
    async def expire_source():
        broker = runtime.capture().broker
        for account, lease in tuple(broker._leases.items()):
            broker._leases[account] = replace(lease, expires_at=0)
        return {"expired": True}

    @app.on_event("shutdown")
    async def close_fixture_client():
        await runtime.dependencies.http_client.aclose()

    app.router.routes.sort(
        key=lambda route: 0 if getattr(route, "path", "").startswith("/__test/") else 1
    )
    uvicorn.run(
        app, host="127.0.0.1", port=snapshot.runtime.port, log_level="error", access_log=False
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]))

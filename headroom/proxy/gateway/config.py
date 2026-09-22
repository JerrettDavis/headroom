"""Immutable version-1 gateway configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Protocol = Literal[
    "openai-chat",
    "openai-responses",
    "anthropic-messages",
    "gemini-generate",
    "vertex-generate",
    "bedrock-invoke",
]
Provider = Literal["openai", "anthropic", "gemini", "vertex", "bedrock", "compatible"]


class FrozenModel(BaseModel):
    """Strict immutable base for one published configuration snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(FrozenModel):
    profile: Literal["gateway"]
    engine: Literal["python"]
    bind: Literal["127.0.0.1"]
    port: int = Field(ge=1, le=65535)
    workers: Literal[1]
    remote_enabled: Literal[False]
    stateless: bool


class TransformConfig(FrozenModel):
    mode: Literal["off"]


class PrivacyConfig(FrozenModel):
    beacon: Literal[False]
    payload_logging: Literal[False]
    metrics: Literal["off", "local"]


class PrincipalConfig(FrozenModel):
    id: Identifier
    secret_ref: Annotated[str, Field(pattern=r"^env:[A-Z][A-Z0-9_]*$")]
    scopes: tuple[Literal["inference", "models"], ...] = Field(min_length=1)
    routes: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_values(self) -> PrincipalConfig:
        _require_unique("principal scopes", self.scopes)
        _require_unique("principal routes", self.routes)
        return self


class ClientAuthConfig(FrozenModel):
    required: Literal[True]
    principals: tuple[PrincipalConfig, ...] = Field(min_length=1)


class EnvironmentSource(FrozenModel):
    kind: Literal["env"]
    ref: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]


class GcpAdcSource(FrozenModel):
    kind: Literal["gcp-adc"]
    project: Annotated[str, Field(min_length=1)]


class AwsChainSource(FrozenModel):
    kind: Literal["aws-chain"]
    region: Annotated[str, Field(min_length=1)]
    profile: Annotated[str, Field(min_length=1)]


class NoCredentialSource(FrozenModel):
    kind: Literal["none"]


CredentialSource = Annotated[
    EnvironmentSource | GcpAdcSource | AwsChainSource | NoCredentialSource,
    Field(discriminator="kind"),
]


class CredentialConfig(FrozenModel):
    id: Identifier
    provider: Provider
    source: CredentialSource
    refresh_owner: Literal["none", "sdk"]
    allowed_origins: tuple[Annotated[str, Field(pattern=r"^https://")], ...] = Field(min_length=1)
    allowed_path_prefixes: tuple[Annotated[str, Field(pattern=r"^/")], ...] = Field(min_length=1)
    enabled: bool

    @model_validator(mode="after")
    def validate_source_contract(self) -> CredentialConfig:
        if self.source.kind in {"gcp-adc", "aws-chain"} and self.refresh_owner != "sdk":
            raise ValueError(f"{self.source.kind} requires refresh_owner='sdk'")
        if self.source.kind in {"env", "none"} and self.refresh_owner != "none":
            raise ValueError(f"{self.source.kind} requires refresh_owner='none'")
        if self.provider == "vertex" and self.source.kind != "gcp-adc":
            raise ValueError("vertex credentials require gcp-adc")
        if self.provider == "bedrock" and self.source.kind != "aws-chain":
            raise ValueError("bedrock credentials require aws-chain")
        if self.source.kind == "none" and self.provider != "compatible":
            raise ValueError("credential source 'none' is restricted to compatible providers")
        _require_unique("allowed origins", self.allowed_origins)
        _require_unique("allowed path prefixes", self.allowed_path_prefixes)
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if parsed.scheme != "https" or not parsed.hostname or parsed.path not in {"", "/"}:
                raise ValueError(f"invalid credential origin: {origin}")
        return self


class RetryConfig(FrozenModel):
    max_attempts: Literal[1]
    ambiguous_commit: Literal["never"]
    after_output: Literal["never"]


class BillingConfig(FrozenModel):
    allow_paid_fallback: Literal[False]


class RouteConfig(FrozenModel):
    id: Identifier
    public_model: Annotated[str, Field(min_length=1)]
    upstream_model: Annotated[str, Field(min_length=1)]
    provider: Provider
    upstream_origin: Annotated[str, Field(pattern=r"^https://")]
    upstream_path_prefix: Annotated[str, Field(pattern=r"^/")]
    credentials: tuple[Identifier, ...] = Field(min_length=1)
    ingress_protocols: tuple[Protocol, ...] = Field(min_length=1)
    native_protocols: tuple[Protocol, ...] = Field(min_length=1)
    translation: Literal["disabled"]
    body_contract: Literal["strict-native", "routed-native"]
    private_network: bool
    retry: RetryConfig
    billing: BillingConfig

    @model_validator(mode="after")
    def unique_values(self) -> RouteConfig:
        _require_unique("route credentials", self.credentials)
        _require_unique("ingress protocols", self.ingress_protocols)
        _require_unique("native protocols", self.native_protocols)
        return self


class TransportConfig(FrozenModel):
    ca_bundle: str | None = None


class GatewayConfigSnapshot(FrozenModel):
    """A validated configuration generation with no resolved secret material."""

    version: Literal[1]
    runtime: RuntimeConfig
    transforms: TransformConfig
    privacy: PrivacyConfig
    client_auth: ClientAuthConfig
    credentials: tuple[CredentialConfig, ...] = Field(min_length=1)
    routes: tuple[RouteConfig, ...] = Field(min_length=1)
    transport: TransportConfig = TransportConfig()

    @classmethod
    def load(cls, path: Path) -> GatewayConfigSnapshot:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def redacted_dict(self) -> dict[str, object]:
        # Version 1 stores references only. Credential values are deliberately not
        # accepted by any model, so a regular dump is already secret-free.
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def validate_references(self) -> GatewayConfigSnapshot:
        credential_ids = _index_unique("credential id", self.credentials, key=lambda item: item.id)
        route_ids = _index_unique("route id", self.routes, key=lambda item: item.id)
        _index_unique("principal id", self.client_auth.principals, key=lambda item: item.id)
        _index_unique("public_model", self.routes, key=lambda item: item.public_model)

        for principal in self.client_auth.principals:
            unknown_routes = sorted(set(principal.routes) - route_ids.keys())
            if unknown_routes:
                raise ValueError(
                    f"principal {principal.id} references unknown route: {unknown_routes}"
                )

        for route in self.routes:
            for credential_id in route.credentials:
                credential = credential_ids.get(credential_id)
                if credential is None:
                    raise ValueError(
                        f"route {route.id} references unknown credential: {credential_id}"
                    )
                if not credential.enabled:
                    raise ValueError(
                        f"route {route.id} references disabled credential: {credential_id}"
                    )
                if credential.provider != route.provider:
                    raise ValueError(
                        f"route {route.id} provider {route.provider} does not match "
                        f"credential {credential_id} provider {credential.provider}"
                    )
                if route.upstream_origin not in credential.allowed_origins:
                    raise ValueError(
                        f"route {route.id} origin is outside credential {credential_id} origins"
                    )
                if not any(
                    route.upstream_path_prefix.startswith(prefix)
                    for prefix in credential.allowed_path_prefixes
                ):
                    raise ValueError(
                        f"route {route.id} path is outside credential {credential_id} prefixes"
                    )
        return self


def gateway_proxy_overrides(snapshot: GatewayConfigSnapshot) -> dict[str, object]:
    """Return authoritative legacy ProxyConfig values for gateway pure mode."""

    return {
        "host": snapshot.runtime.bind,
        "port": snapshot.runtime.port,
        "optimize": False,
        "cache_enabled": False,
        "memory_enabled": False,
        "traffic_learning_enabled": False,
        "ccr_inject_tool": False,
        "ccr_inject_marker": False,
        "ccr_handle_responses": False,
        "code_graph_watcher": False,
        "image_optimize": False,
        "retry_enabled": False,
        "stateless": snapshot.runtime.stateless,
    }


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _index_unique(label: str, values: tuple[object, ...], *, key):  # type: ignore[no-untyped-def]
    result = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ValueError(f"duplicate {label}: {item_key}")
        result[item_key] = value
    return result

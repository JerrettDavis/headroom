"""Bounded, content-free telemetry for the gateway profile."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

IngressProtocol: TypeAlias = Literal[
    "openai-chat",
    "openai-responses",
    "anthropic-messages",
    "gemini-generate",
    "vertex-generate",
    "bedrock-invoke",
]
RouteClass: TypeAlias = Literal["public-api", "private-compatible", "cloud-workload"]
Adapter: TypeAlias = Literal["strict-native", "routed-native", "translated"]
CredentialSource: TypeAlias = Literal["env", "gcp-adc", "aws-chain", "none"]
FailureOrigin: TypeAlias = Literal["none", "client", "gateway", "identity", "network", "upstream"]
RetryReason: TypeAlias = Literal[
    "none", "connect", "rate-limit", "unavailable", "credential-refresh"
]
TerminalResult: TypeAlias = Literal["success", "rejected", "cancelled", "failed", "unknown"]
GatewayDimensions: TypeAlias = tuple[str, str, str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class GatewayEvent:
    ingress_protocol: IngressProtocol
    route_class: RouteClass
    adapter: Adapter
    credential_source: CredentialSource
    failure_origin: FailureOrigin
    retry_reason: RetryReason
    terminal_result: TerminalResult

    def dimensions(self) -> GatewayDimensions:
        return (
            self.ingress_protocol,
            self.route_class,
            self.adapter,
            self.credential_source,
            self.failure_origin,
            self.retry_reason,
            self.terminal_result,
        )


class GatewayObservability:
    """Aggregate events without retaining request or response content."""

    def __init__(self) -> None:
        self._counts: Counter[GatewayDimensions] = Counter()

    def record(self, event: GatewayEvent) -> None:
        self._counts[event.dimensions()] += 1

    def snapshot(self) -> MappingProxyType[GatewayDimensions, int]:
        return MappingProxyType(dict(self._counts))

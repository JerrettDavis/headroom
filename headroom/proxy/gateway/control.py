"""Redacted gateway control-plane value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedactedGatewayStatus:
    service: str
    profile: str
    generation: int
    ready: bool
    route_count: int
    credential_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "profile": self.profile,
            "generation": self.generation,
            "ready": self.ready,
            "route_count": self.route_count,
            "credential_count": self.credential_count,
        }

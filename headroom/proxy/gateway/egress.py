"""Credential-specific destination authorization."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping
from typing import NoReturn
from urllib.parse import urlsplit

from headroom.proxy.gateway.credentials import CredentialLease
from headroom.proxy.gateway.errors import GatewayEgressDenied

_CLIENT_AUTH_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "x-headroom-proxy-token",
        "proxy-authorization",
    }
)


class EgressPolicy:
    """Authorize the final URL and resolved addresses before revealing a secret."""

    def authorize(
        self,
        lease: CredentialLease,
        url: str,
        *,
        resolved_addresses: Iterable[str] = (),
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
            self._deny()
        if not parsed.hostname:
            self._deny()
        port = parsed.port or 443
        origin = f"https://{parsed.hostname.lower()}:{port}"
        if origin not in lease.allowed_origins:
            self._deny()
        if not any(parsed.path.startswith(prefix) for prefix in lease.allowed_path_prefixes):
            self._deny()
        for raw_address in resolved_addresses:
            address = ipaddress.ip_address(raw_address)
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_multicast
                or address.is_unspecified
            ):
                self._deny()

    @staticmethod
    def _deny() -> NoReturn:
        raise GatewayEgressDenied(
            status_code=502,
            code="gateway_egress_denied",
            message="Upstream destination is not authorized for this credential",
        )


def build_managed_upstream_headers(
    client_headers: Mapping[str, str],
    lease: CredentialLease,
    url: str,
    *,
    resolved_addresses: Iterable[str] = (),
) -> dict[str, str]:
    """Strip every caller credential before attaching one authorized lease."""

    EgressPolicy().authorize(lease, url, resolved_addresses=resolved_addresses)
    result = {
        name.lower(): value
        for name, value in client_headers.items()
        if name.lower() not in _CLIENT_AUTH_HEADERS
    }
    result.update(lease.authorization_headers())
    return result

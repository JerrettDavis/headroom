"""Credential-specific destination authorization."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import NoReturn
from urllib.parse import parse_qsl, quote, urlsplit

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
    method: str = "GET",
    body: bytes = b"",
) -> dict[str, str]:
    """Strip every caller credential before attaching one authorized lease."""

    EgressPolicy().authorize(lease, url, resolved_addresses=resolved_addresses)
    result = {
        name.lower(): value
        for name, value in client_headers.items()
        if name.lower() not in _CLIENT_AUTH_HEADERS
    }
    if lease.provider == "bedrock":
        result.update(_aws_sigv4_headers(lease, method, url, result, body))
    else:
        result.update(lease.authorization_headers())
    return result


def _aws_sigv4_headers(
    lease: CredentialLease,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
) -> dict[str, str]:
    """Sign one Bedrock request from an opaque AWS SDK credential object."""

    credentials = lease.secret._reveal_for_egress()
    access_key = getattr(credentials, "access_key", None)
    secret_key = getattr(credentials, "secret_key", None)
    session_token = getattr(credentials, "token", None)
    if not isinstance(access_key, str) or not isinstance(secret_key, str):
        raise TypeError("AWS credential object is missing signing fields")

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port not in (None, 443):
        host = f"{host}:{parsed.port}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date = timestamp[:8]
    region = host.removeprefix("bedrock-runtime.").removesuffix(".amazonaws.com")
    payload_hash = hashlib.sha256(body).hexdigest()
    signing_headers = {"host": host, "x-amz-date": timestamp}
    if session_token:
        signing_headers["x-amz-security-token"] = str(session_token)
    content_type = headers.get("content-type")
    if content_type:
        signing_headers["content-type"] = content_type.strip()
    canonical_headers = "".join(
        f"{name}:{signing_headers[name]}\n" for name in sorted(signing_headers)
    )
    signed_headers = ";".join(sorted(signing_headers))
    canonical_query = "&".join(
        f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}"
        for key, value in sorted(parse_qsl(parsed.query, keep_blank_values=True))
    )
    canonical_request = "\n".join(
        (
            method.upper(),
            quote(parsed.path or "/", safe="/-_.~"),
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        )
    )
    scope = f"{date}/{region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            timestamp,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        )
    )

    def sign(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode(), hashlib.sha256).digest()

    signing_key = sign(
        sign(sign(sign(("AWS4" + secret_key).encode(), date), region), "bedrock"),
        "aws4_request",
    )
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    result = {
        "authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": timestamp,
    }
    if session_token:
        result["x-amz-security-token"] = str(session_token)
    return result

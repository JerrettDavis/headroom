"""Gateway errors that are safe to map onto public protocol responses."""


class GatewayConfigurationError(ValueError):
    """Raised when a gateway configuration violates a cross-field invariant."""


class GatewayPublicError(Exception):
    """A bounded public error that never embeds secret or provider detail."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class GatewayAuthError(GatewayPublicError):
    """Caller authentication or browser-boundary failure."""


class GatewayAuthorizationError(GatewayPublicError):
    """Authenticated caller lacks the requested scope, route, or capability."""

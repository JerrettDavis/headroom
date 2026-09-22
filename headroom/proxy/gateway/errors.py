"""Gateway errors that are safe to map onto public protocol responses."""


class GatewayConfigurationError(ValueError):
    """Raised when a gateway configuration violates a cross-field invariant."""

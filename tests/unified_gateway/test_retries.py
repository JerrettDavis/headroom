from __future__ import annotations

import pytest

from headroom.proxy.gateway.routing import ProviderContract, RetryDecision, TransportFailure


@pytest.mark.parametrize("exposure", ["ambiguous_acceptance", "first_byte", "tool_event"])
def test_exposed_or_ambiguous_operation_is_never_retried(exposure: str) -> None:
    assert (
        RetryDecision.classify(
            TransportFailure(kind="connect", retry_after=None),
            exposure,
            ProviderContract(max_attempts=2, retryable_failures=frozenset({"connect"})),
        )
        is False
    )


def test_precommit_retry_honors_contract_and_retry_after_bound() -> None:
    contract = ProviderContract(
        max_attempts=2,
        retryable_failures=frozenset({"connect", "rate_limit"}),
        max_retry_after=5.0,
    )

    assert RetryDecision.classify(
        TransportFailure(kind="connect", retry_after=None), "none", contract
    )
    assert RetryDecision.classify(
        TransportFailure(kind="rate_limit", retry_after=4.0), "none", contract
    )
    assert not RetryDecision.classify(
        TransportFailure(kind="rate_limit", retry_after=6.0), "none", contract
    )

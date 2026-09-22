## Summary

Adds a draft Python unified gateway with authenticated callers, route grants,
credential-bound egress, native and qualified translated protocols, stateful
ownership, atomic admission, safe reload/shutdown, and closed-by-default identity
admission.

## Local qualification

- Base: `94206e265203acfd72a3b939e9a964e29175ad50`
- Implementation source: `5968b2739`
- Wheel: `headroom_ai-0.38.0-cp310-abi3-win_amd64.whl`
- SHA-256: `c9060f9c25ecea381c3fadfda1903783b8be8475c906e1552abfedac5b0254a3`
- Gateway suite: 134 passed
- Ruff check/format, gateway mypy, Cargo fmt/tests: passed
- Exact installed-wheel smoke: passed

The full Python suite collected 13,448 tests but did not terminate in an existing
CLI-proxy test after reaching 20%; pre-existing bundled-tools and Windows wrapper
failures had appeared. See `LOCAL_QUALIFICATION.md` for exact boundaries.

## Unsupported and external gates

- Native subscription identities remain unavailable and negatively tested.
- No paid/live-provider request was run.
- Provider entitlement, real expiry/refresh, OS keychain behavior, and official
  release qualification remain external `not_run` gates.
- This PR does not release, merge, or enable auto-merge.

Human merge approval required: yes

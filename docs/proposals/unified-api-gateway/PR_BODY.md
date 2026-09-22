## Summary

Adds a draft Python unified gateway with authenticated callers, route grants,
credential-bound egress, native and qualified translated protocols, stateful
ownership, atomic admission, safe reload/shutdown, and closed-by-default identity
admission.

## Local qualification

- Base: `94206e265203acfd72a3b939e9a964e29175ad50`
- Qualified source: `e45c97fd5`
- Wheel: `headroom_ai-0.38.0-cp310-abi3-win_amd64.whl`
- SHA-256: `f9982fd52df59f03a1f376b6db038d1adf928ed1df74038b132cb43aa3ae02eb`
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

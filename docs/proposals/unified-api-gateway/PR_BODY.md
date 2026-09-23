## Description

Adds an opt-in Python unified API gateway for authenticated callers. It enforces route grants and credential-bound egress, supports native OpenAI, Anthropic, and Gemini protocols plus explicitly qualified translations, owns stateful connections, performs atomic admission, and provides safe reload and shutdown behavior.

The implementation is closed by default for provider identities that have not been qualified. This is a draft: native subscription identities, paid/live-provider traffic, and official release qualification remain outside the evidence collected here.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [x] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)

## Changes Made

- Added the opt-in gateway runtime, caller authentication and authorization, credential broker, bounded egress, native dispatch, qualified protocol translation, state ownership, and atomic budget/concurrency admission.
- Added safe configuration loading and reload, readiness, shutdown, privacy-preserving telemetry, provider/OAuth discovery, and closed-by-default provider identity admission.
- Added CLI and SDK process acceptance coverage, scenario/requirement ledgers, schemas and examples, and an exact-wheel artifact verifier.
- Documented the design, implementation plan, capability boundaries, and local qualification evidence under `docs/proposals/unified-api-gateway/`.

## Testing

- [ ] Unit tests pass (`pytest`)
- [x] Linting passes (`ruff check .`)
- [x] Type checking passes (`mypy headroom`)
- [x] New tests added for new functionality
- [x] Manual testing performed

### Test Output

```text
Qualified source: e45c97fd5
Base: 94206e265203acfd72a3b939e9a964e29175ad50
Artifact: headroom_ai-0.38.0-cp310-abi3-win_amd64.whl
SHA-256: f9982fd52df59f03a1f376b6db038d1adf928ed1df74038b132cb43aa3ae02eb

Gateway suite: 134 passed
Ruff check/format: passed
Gateway mypy: passed
Cargo fmt/tests: passed
Exact installed-wheel smoke: passed

The full Python suite collected 13,448 tests but did not terminate after
reaching 20% in an existing CLI-proxy test. Pre-existing bundled-tools
Windows wrapper failures also appeared. See LOCAL_QUALIFICATION.md for the
exact commands, scopes, and boundaries.
```

## Real Behavior Proof

- Environment: Windows, CPython 3.13, installed `cp310-abi3-win_amd64` wheel built from `e45c97fd5`, plus repository test environment.
- Exact command / steps: Built the wheel, verified its exact SHA-256 with `scripts/verify_unified_gateway_artifact.py`, installed it with `--no-deps`, ran the gateway unit/process suites and SDK parser smoke tests, and ran Ruff, mypy, Cargo formatting/tests, and the broader Python suite. Exact command lines are recorded in `docs/proposals/unified-api-gateway/LOCAL_QUALIFICATION.md`.
- Observed result: 134 gateway tests passed; real-process OpenAI, Anthropic, and Gemini SDK/wire-shape tests passed; the exact installed-wheel smoke passed; static and Rust checks passed. The broader suite reached 20% before hanging in an existing CLI-proxy test.
- Not tested: Paid or live-provider requests; native subscription identities; provider entitlement; real token expiry/refresh; OS keychain behavior; official retained release-artifact qualification; production activation.

## Runtime Rollout Safety

- Rollout-managed feature(s): Unified API gateway runtime, routes, protocol translation, provider identities, and credential sources.
- Minimum rollout channel: Draft/local qualification only; no production rollout is authorized by this PR.
- Stable/default behavior changed: No. The gateway is opt-in and requires an explicit configuration/profile.
- Kill switch / disable path: Stop invoking the gateway profile/process or remove its explicit gateway configuration; unqualified provider identities remain denied by default.
- Unsafe override required: No unsafe override is required or introduced for qualified local paths.
- Qualification impact: New gateway paths require retained release-artifact and live-provider qualification before production activation; unsupported identity modes remain closed.
- Rollback path: Revert the feature commits or disable the opt-in gateway configuration/profile; existing proxy behavior remains the default.

## Review Readiness

- [x] I have performed a self-review
- [ ] This PR is ready for human review

## Checklist

- [x] My code follows the project's style guidelines
- [x] I have performed a self-review of my code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests covering the changed gateway paths pass locally
- [x] I did **not** edit `CHANGELOG.md` — it is generated by release-please from the Conventional Commit PR title (a CI guard enforces this)

## Screenshots (if applicable)

Not applicable; this change adds a CLI/server gateway and no graphical interface.

## Additional Notes

- This PR does not release, merge, enable auto-merge, or authorize production activation. Human merge approval remains required.
- Native subscription identities are deliberately unavailable and negatively tested.
- Provider entitlement, real expiry/refresh, OS keychain behavior, live-provider traffic, and official release qualification are explicit external `not_run` gates.
- Detailed qualification evidence: `docs/proposals/unified-api-gateway/LOCAL_QUALIFICATION.md`.

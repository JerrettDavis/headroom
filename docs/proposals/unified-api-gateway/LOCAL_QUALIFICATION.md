# Unified API Gateway Local Qualification

Status: draft, local evidence only. No paid or live-provider request was run.

## Qualified source and artifact

- Qualified source commit: `e45c97fd5`
- Base commit: `94206e265203acfd72a3b939e9a964e29175ad50`
- Wheel: `headroom_ai-0.38.0-cp310-abi3-win_amd64.whl`
- Wheel SHA-256: `f9982fd52df59f03a1f376b6db038d1adf928ed1df74038b132cb43aa3ae02eb`
- Platform: Windows amd64, CPython 3.13; wheel ABI is CPython 3.10+ abi3

## Evidence

- Unified gateway suite: 134 passed.
- Task 9 operational batch: 88 passed; one repository-environment Langfuse
  default assertion reproduced independently because tracing is configured at
  collection in this checkout.
- Task 10: 22 new provider-auth tests and 43 existing regressions passed.
- Real-process/coverage checkpoint: 8 passed.
- Ruff check and format, gateway mypy, `cargo fmt --check`, and
  `cargo test --workspace`: passed.
- Exact-wheel verifier: passed outside the source tree after digest verification
  and `--no-deps` installation.

The monolithic Python run collected 13,448 tests after installing the missing
test-only `respx` dependency. It reached 20% before an existing
`tests/test_cli_proxy_improvements.py` process failed to terminate after more
than nine minutes. Earlier failures appeared in the recorded bundled-tools area
and Windows CLI-wrapper tests. They are not counted as gateway success evidence.

## Requirement audit

- R01-R14: proved locally by gateway tests and real-process checks.
- R15: workload identity construction/failure proved locally; real cloud
  identity and expiry evidence is `external_not_run`.
- R16: OpenAI/Anthropic SDK parsers and Gemini wire shape exercised through a
  real process. Live-provider behavior is `external_not_run`.
- R17: static/Rust checks and exact-wheel smoke passed. Official retained
  release qualification is `external_not_run`.
- R18-R19: unavailable native identities are negatively tested; hostile OAuth
  and device-flow behavior is locally tested. No subscription adapter is enabled.

This report does not approve release, merge, auto-merge, or production
activation. Human merge approval is required.

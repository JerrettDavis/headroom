# Deterministic SDK Generation Progress

## Working State

- Approved design: `docs/superpowers/specs/2026-10-01-deterministic-sdk-generation-design.md`
- Approved plan: `docs/superpowers/plans/2026-10-01-deterministic-sdk-generation.md`
- Checkout: detached linked worktree; commits are checkpoints until attached to a feature branch.
- Source bundle: `C:\Users\jd\Downloads\headroom-deterministic-sdk-generation.zip`

## 2026-10-01 — Task 1: Three-Language Baseline

- Installer dry-run correctly refused the detached checkout. The safety check was retained; the checksum-reviewed new-file overlay was installed explicitly and server metadata was reconciled by hand.
- Added import-free AST compiler, deterministic emitter, CLI, conformance runner, Python/TypeScript/Go runtime templates, codegen configuration, tests, and engineering notes.
- Added no-op `sdk_operation` metadata to the real CCR POST and GET retrieval handlers. Existing route decorators, guards, signatures, handler bodies, and response behavior remain unchanged.
- RED: `SourceCompilerTests.test_real_pilot_paths_and_types` failed with `No annotated operations found` before server metadata was added.
- GREEN: the same production-source test passed after metadata integration.
- Windows qualification found the test AST loader relied on the platform default encoding. Root cause was UTF-8 source containing non-CP1252 bytes; the loader now requests UTF-8 explicitly.
- Validation: `python -m unittest discover -s tests/sdkgen -p 'test_*.py' -v` — 37 tests passed; generated fixture snapshot check reported 20 byte-for-byte files.

## Remaining

1. Harden ownership/determinism and generate the current snapshot.
2. Isolate emitter modules.
3. Add and qualify Rust generation.
4. Add and qualify .NET generation.
5. Enforce five-language behavioral conformance.
6. Add CI, repository qualification, and draft-PR notes.

## 2026-10-01 — Task 2: Ownership and Determinism

- Added explicit generated-path validation for absolute paths, parent traversal, Windows drive syntax, backslash aliases, and non-byte emitter output.
- Generation now stages every output before removing stale managed files. A forced staging failure proved an existing stale managed file remains intact.
- Added checks that drift detection is read-only, symlinked output is rejected, and separate output roots are byte-identical.
- RED: unsafe paths failed with a missing validation interface, and a forced temporary-file staging error removed `stale.txt` before the fix.
- GREEN: `OwnershipTests` and `DeterminismTests` passed all 5 cases.
- Generated the real current-source snapshot under `sdk/generated-pilot`.
- Validation: source `generate` emitted 20 files; source `check` matched all 20 byte-for-byte; the complete SDK generator suite passed 42 tests.

## 2026-10-01 — Task 3: Emitter Boundary

- Added a typed emitter registry and independent Python, TypeScript, and Go adapters accepting the canonical document and returning fully qualified output paths.
- Added collision-checked emitter composition; duplicate paths cannot silently overwrite another backend's artifact.
- RED: `EmitterCompositionTests` failed because `render_with_emitters` did not exist.
- GREEN: the collision test passed and the full suite reached 43 passing tests.
- Adding emitter modules intentionally changed only `manifest.json` because generator input provenance includes Python source files. Regeneration left all language artifact bytes unchanged, and source drift checking matched all 20 files.
- The proven legacy emission bodies remain in `emit.py` behind the new adapters for this pilot; Rust and .NET use the stable adapter interface directly.

## 2026-10-01 — Task 4: Rust Generation

- Added deterministic Rust 2021 generation with Serde wire models, manual presence-aware deserialization, unknown-field preservation, async Reqwest client bindings, pinned direct dependencies, and a generated Cargo lockfile.
- The transport rejects non-HTTP base URLs and credentials/query/fragment components, disables redirects, applies a finite timeout, bounds streamed response bodies, performs no automatic retries, percent-encodes path segments, and retains status/headers/raw error bytes without including response bodies in display text.
- Required-nullable fields remain `Option<T>` to callers while custom deserialization rejects omission. Generated unit coverage accepts explicit null and rejects a missing `tool_name`.
- RED: Rust output tests initially failed because no Rust artifacts existed. Cargo then exposed the repository workspace-boundary requirement, and Clippy exposed an oversized error enum variant.
- GREEN: generated model/runtime tests passed; Cargo compiled the async client with the committed lockfile; Clippy passed with warnings denied after boxing the API-error variant.
- Rust build output is directed outside the managed generated tree so ownership checking remains exact. Generated Rust is compiler-owned and not rewritten by rustfmt; Cargo compilation and Clippy are the native static gates.
- Added the Rust shared-wire test source for Task 6 loopback execution.

## 2026-10-01 — Task 5: .NET Generation

- Added deterministic .NET 10/C# generation with nullable reference types, required members, JSON wire-name attributes, `JsonExtensionData`, and a presence-aware `Optional<T>` converter for omitted versus present optional properties.
- Added async `HttpClient` bindings with cancellation tokens, linked finite timeouts, disabled redirects, `ResponseHeadersRead`, streamed bounded reads, path escaping, and structured status/header/raw-body exceptions whose display text omits the payload.
- RED: .NET output tests initially failed because no .NET artifacts existed.
- GREEN: generated-output tests passed; `dotnet build` completed with zero warnings/errors; the executable model test accepted explicit-null `tool_name`, rejected omission, and preserved an unknown nested field.
- `dotnet format --verify-no-changes` formatted 0 of 6 files. Its local `bin`/`obj` build artifacts were removed after an exact `git clean -ndx` preview; all later .NET commands use an external artifacts directory to keep generated ownership exact.
- Source drift checking now matches 30 generated files byte-for-byte.

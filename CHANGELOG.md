# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Features

* **proxy:** per-project savings breakdown on the dashboard for all wrapped agents — Claude Code, Codex, aider, Copilot, and Cursor ([#802](https://github.com/chopratejas/headroom/issues/802)). `headroom wrap claude`/`codex` tag requests with an `X-Headroom-Project` header (launch-directory name); `wrap aider`/`copilot`/`cursor` — whose clients cannot send custom headers — use a `/p/<name>` base-URL prefix the proxy strips. Savings are aggregated per project (persisted, schema v3 with transparent v2 migration), exposed as `savings.per_project` in `/stats` and `projects` in `/stats-history`, and shown in a Per-Project Savings dashboard table.

### Features

* **memory:** opt-in Apple-GPU (MPS) embedding offload via `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`. When set (and Apple MPS is available), the memory embedder runs on the torch sentence-transformers backend on the Apple GPU instead of the default ONNX CPU embedder, freeing the CPU under load. If MPS or the dependencies are unavailable, Headroom logs a warning and uses the existing default embedder selection path (ONNX when available, then the pre-existing local fallback). MPS encode calls are serialized internally (torch-MPS is not thread-safe). Adds the new `[pytorch-mps]` extra (`pip install 'headroom-ai[pytorch-mps]'`). Default behavior is unchanged.

### Bug Fixes

* **ccr:** make retrieval store TTL configurable with `HEADROOM_CCR_TTL_SECONDS`, expose the effective TTL in `/v1/retrieve/stats`, and distinguish expired retrievals from missing hashes.


## [0.26.0](https://github.com/JerrettDavis/headroom/compare/v0.25.0...v0.26.0) (2026-06-12)


### Features

* add differential network capture harness ([#761](https://github.com/JerrettDavis/headroom/issues/761)) ([11ab5f8](https://github.com/JerrettDavis/headroom/commit/11ab5f83a1ccd617a2608349a42feff7f7e72b98))
* add lean-ctx context tool support ([4b06179](https://github.com/JerrettDavis/headroom/commit/4b061792b29f776f36831530ba6fdbfd6d71ad0b))
* add light mode for dashboard ([#834](https://github.com/JerrettDavis/headroom/issues/834)) ([c425893](https://github.com/JerrettDavis/headroom/commit/c425893d123e67c62ee20ff64ae350eb4ea56477))
* add OAuth2 client-credentials upstream-auth proxy extension ([#778](https://github.com/JerrettDavis/headroom/issues/778)) ([#784](https://github.com/JerrettDavis/headroom/issues/784)) ([eb2e50f](https://github.com/JerrettDavis/headroom/commit/eb2e50feb26bacadf8812d6e608a458a990096b9))
* add Vertex AI proxy routing ([#793](https://github.com/JerrettDavis/headroom/issues/793)) ([3c77e52](https://github.com/JerrettDavis/headroom/commit/3c77e52ce431210e6045671cf5f7c66c79f90a32))
* **ci:** X2 — PR-time release dry-run via path-filtered pull_request trigger ([7a06355](https://github.com/JerrettDavis/headroom/commit/7a063554ce03c04cbacd45d548f513f99dcde95a))
* **ci:** X2 — PR-time release dry-run via path-filtered pull_request trigger ([b9f84fa](https://github.com/JerrettDavis/headroom/commit/b9f84fa815a2696a85ed3f430d52b6ccaf36ac82))
* **cli:** comprehensive help text, validation, and exception handling improvements ([#640](https://github.com/JerrettDavis/headroom/issues/640)) ([028efab](https://github.com/JerrettDavis/headroom/commit/028efabb4e611d77118baefb8ffdd13b0edc4fc5))
* compression safety rails — error-output protection, pipeline circuit breaker, library inflation guard ([#851](https://github.com/JerrettDavis/headroom/issues/851)) ([c0cadcc](https://github.com/JerrettDavis/headroom/commit/c0cadccff98e572f126185f371e4de9e241b12e0))
* **copilot:** GitHub Copilot subscription mode through Headroom ([f4dff9b](https://github.com/JerrettDavis/headroom/commit/f4dff9b4885b5c62d79396bbb0847ae3e39a9bd9))
* **dashboard:** per-model savings breakdown and expected-vs-actual cost on historical charts ([#807](https://github.com/JerrettDavis/headroom/issues/807)) ([34dafe6](https://github.com/JerrettDavis/headroom/commit/34dafe69d907c9a2971abc0d801ff9bfa498b3a8))
* detect re-served tool results as over-compression waste signal ([#854](https://github.com/JerrettDavis/headroom/issues/854)) ([5f1d88a](https://github.com/JerrettDavis/headroom/commit/5f1d88ad2701ed186df93d8e2a3980f0329d9dbb))
* **evals:** add zero-cost tool schema compaction integrity eval ([#817](https://github.com/JerrettDavis/headroom/issues/817)) ([53a08c6](https://github.com/JerrettDavis/headroom/commit/53a08c63bf56a76d4fb7b649e37c8e62b0b4cebf))
* gated Markdown-KV compaction formatter (serialization-aware output) ([#859](https://github.com/JerrettDavis/headroom/issues/859)) ([06b2625](https://github.com/JerrettDavis/headroom/commit/06b2625b17b0b032f688d321c6aa30ae3f2b7d96))
* **kompress:** warn on unrecognized HEADROOM_KOMPRESS_BACKEND + document backend selection ([#204](https://github.com/JerrettDavis/headroom/issues/204)) ([6367d0b](https://github.com/JerrettDavis/headroom/commit/6367d0b7228f53b29bbd20f55c1729476ba5ea68))
* **memory:** add opt-in Apple-GPU (MPS) embedding runtime ([#766](https://github.com/JerrettDavis/headroom/issues/766)) ([c71592d](https://github.com/JerrettDavis/headroom/commit/c71592d4214adf1022e4c608518ae0c3ac4aa5e9))
* net-cost cache mutation formula on CompressionPolicy ([#856](https://github.com/JerrettDavis/headroom/issues/856) P1) ([#857](https://github.com/JerrettDavis/headroom/issues/857)) ([d5f5802](https://github.com/JerrettDavis/headroom/commit/d5f58026e2a882bc508acfbddfc9d472100d6e16))
* **perf:** add --format {text,json,csv} to `headroom perf` ([#648](https://github.com/JerrettDavis/headroom/issues/648)) ([9fe4886](https://github.com/JerrettDavis/headroom/commit/9fe4886cf6b612452f7271d3204872f804074c1f))
* **plugins:** Hermes agent headroom_retrieve plugin ([#824](https://github.com/JerrettDavis/headroom/issues/824)) ([058bced](https://github.com/JerrettDavis/headroom/commit/058bcedab838f3b34ac8e38853e1924329efd820))
* probe-based retention scoring of recorded compression events ([#862](https://github.com/JerrettDavis/headroom/issues/862)) ([c2106cb](https://github.com/JerrettDavis/headroom/commit/c2106cbdabb905e1980c6694000c220a5042171c))
* **proxy:** add --exclude-tools flag + HEADROOM_EXCLUDE_TOOLS env var ([1058043](https://github.com/JerrettDavis/headroom/commit/10580439bb4227a3f6f375439b0baae60db2f288))
* **proxy:** add --exclude-tools flag + HEADROOM_EXCLUDE_TOOLS env var ([93cf8af](https://github.com/JerrettDavis/headroom/commit/93cf8af0e0c6b6b7fee0bcfe5b57a514c294a898))
* **proxy:** add CLI opt-outs for CCR injection (compression-only mode) ([#823](https://github.com/JerrettDavis/headroom/issues/823)) ([693d9d2](https://github.com/JerrettDavis/headroom/commit/693d9d20e2b2d9bfce3a0c48314850ee77ff8af3))
* **proxy:** add client (harness) identification — per-harness analytics across every handler ([96a674f](https://github.com/JerrettDavis/headroom/commit/96a674f1b9c7d0b251478bb53541f8fbd6859e1f))
* **proxy:** attribute savings history rollups per provider ([#791](https://github.com/JerrettDavis/headroom/issues/791)) ([0b8b8d9](https://github.com/JerrettDavis/headroom/commit/0b8b8d92de3bd5e0301eadedacfb4b1d20a8de7f))
* **proxy:** log compressed messages alongside original request ([#261](https://github.com/JerrettDavis/headroom/issues/261)) ([2269e40](https://github.com/JerrettDavis/headroom/commit/2269e40bde7e1b9fb0620bd2cec9e33a92834080))
* **proxy:** per-project savings breakdown on the dashboard (claude, codex, aider, copilot, cursor) ([#803](https://github.com/JerrettDavis/headroom/issues/803)) ([914a60a](https://github.com/JerrettDavis/headroom/commit/914a60a2b07caad8488c1e19a5465726b95f83d3))
* **proxy:** show resolved upstream API targets in startup banner ([#586](https://github.com/JerrettDavis/headroom/issues/586)) ([8dbe7ad](https://github.com/JerrettDavis/headroom/commit/8dbe7ad41b3a1d33c01874be5c1cbc68a5e68111)), closes [#583](https://github.com/JerrettDavis/headroom/issues/583)
* **relevance:** weight BM25 score_batch by corpus IDF ([#646](https://github.com/JerrettDavis/headroom/issues/646)) ([88177bd](https://github.com/JerrettDavis/headroom/commit/88177bd7a680490ac85d244c5fff90f21a3be27c))
* support CLAUDE_CODE_USE_FOUNDRY and custom upstream gateways ([#726](https://github.com/JerrettDavis/headroom/issues/726)) ([d90cdce](https://github.com/JerrettDavis/headroom/commit/d90cdce3b69bbf27e0f5feea461766a9d797cf7e))
* support Python 3.14+ via pyo3 abi3 stable ABI ([#516](https://github.com/JerrettDavis/headroom/issues/516)) ([19eac8e](https://github.com/JerrettDavis/headroom/commit/19eac8e00dc9e3911f3afe8e8e5dcc9e00346baa))
* switch Kompress default to kompress-v2-base with weight-only int8 ONNX ([#799](https://github.com/JerrettDavis/headroom/issues/799)) ([74392b2](https://github.com/JerrettDavis/headroom/commit/74392b238e4f76fa061e673d1415fc7fa2830011))
* **transforms:** attribute read_lifecycle + smart_crush tags ([#249](https://github.com/JerrettDavis/headroom/issues/249)) ([8f37426](https://github.com/JerrettDavis/headroom/commit/8f374263d3971c072b5c977375c873864fb05763))


### Bug Fixes

* A0 — fail-loud rust core deployment smoke test ([00ab1ea](https://github.com/JerrettDavis/headroom/commit/00ab1ea74dd91c08c549d6272511762f939cc785))
* A2 — system prompt immutable; memory routes to live-zone tail; cache_aligner detector-only ([704fb2f](https://github.com/JerrettDavis/headroom/commit/704fb2f19df07ec88b19143c2ad423b0b335a7ab))
* A3 — byte-faithful Python forwarders; serialize canonical only when mutated ([f0dcc02](https://github.com/JerrettDavis/headroom/commit/f0dcc027754a68d84861c584e318f3596526410d))
* A5 — strip x-headroom-* from upstream-bound headers (P5-49) ([2e874c5](https://github.com/JerrettDavis/headroom/commit/2e874c5e3ee0755c63b9bd385ab9db63b2a14f9a))
* A6 — anthropic-beta and openai-beta deterministic merge + session-sticky ([aec5ba3](https://github.com/JerrettDavis/headroom/commit/aec5ba32538641dedf4f773ee81764eda878a318))
* A7 — memory tool injection session-sticky for both Anthropic and OpenAI ([8dcd474](https://github.com/JerrettDavis/headroom/commit/8dcd474aca4efd666909da8a7560d5db05d5f778))
* A8 — SSE delta arms, UTF-8 buffer, phase preservation, request-id, 413 ([148ded3](https://github.com/JerrettDavis/headroom/commit/148ded392aef60839a9d4b9dd591601496ef5312))
* A9 — tag protector discards wrap on placeholder loss ([6aacd48](https://github.com/JerrettDavis/headroom/commit/6aacd4805add7b6fe06d18134a2d4e1365960742))
* add Codex wire debug and WS usage metrics ([7c94e72](https://github.com/JerrettDavis/headroom/commit/7c94e728410d8784ecdcddb3d7c9bd6dc079e376))
* add Codex wire debug and WS usage metrics ([2b331e2](https://github.com/JerrettDavis/headroom/commit/2b331e297a02b0a685b9f3fdf2432eefc3cdcccd))
* add Kompress backend and thread controls ([a2ea964](https://github.com/JerrettDavis/headroom/commit/a2ea9648a4380fee97a4ad2b98ed0197d8ea03f0))
* align docs and prune dead compatibility surfaces ([359e8c9](https://github.com/JerrettDavis/headroom/commit/359e8c9a5b6f4dfc0c4126affd3308893da4c421))
* **anthropic:** CCR exception must re-raise, not silently swallow ([#838](https://github.com/JerrettDavis/headroom/issues/838)) ([8db5efc](https://github.com/JerrettDavis/headroom/commit/8db5efc6f9f6de59e9d55cbcd63b75c37a81a26e))
* B1 — retire ICM, RollingWindow, scoring, relevance + dependents ([967b0db](https://github.com/JerrettDavis/headroom/commit/967b0db4392d358a68003cb3e0ea1af88e1e54b8))
* B2 — live-zone block dispatcher skeleton ([e190544](https://github.com/JerrettDavis/headroom/commit/e190544c77573e4441fd50815b7e67319df8fa44))
* B3 — wire type-aware compressors into live-zone dispatcher ([2b55050](https://github.com/JerrettDavis/headroom/commit/2b55050a654b78d43b78609fc5b17b27e732f6b0))
* B4 — token validation gate + per-content-type byte thresholds ([b3b3fef](https://github.com/JerrettDavis/headroom/commit/b3b3feff6f29b9c3eeba86c66ef2a43e5591e932))
* B5 — TOIN observation-only refactor + per-tenant aggregation key ([6819b7e](https://github.com/JerrettDavis/headroom/commit/6819b7e5e5e820a342fce7afc448a6ac953cf82e))
* B6 — memory injection moves to live-zone user-tail ([2ee0577](https://github.com/JerrettDavis/headroom/commit/2ee05774b92ddd759459bfa0aab684e41f143f5a))
* B7 — CCR hardening: persistent backends + always-on tool ([00902b8](https://github.com/JerrettDavis/headroom/commit/00902b8feaf4de1f173e051fcbb265a9ac9b56c1))
* batch and guard Kompress inference ([b946a44](https://github.com/JerrettDavis/headroom/commit/b946a44c44a74ec601fd9fb4b97a4199b6c9b792))
* **bedrock:** apply rustfmt to header_value_preview tests ([42ff8af](https://github.com/JerrettDavis/headroom/commit/42ff8afa90c29a11079256bbe2d4a96878d708c6))
* **bedrock:** use floor_char_boundary to avoid UTF-8 slice panic in header preview ([399cb32](https://github.com/JerrettDavis/headroom/commit/399cb32fa475ebbc0e6790024f77675d6b72f4fd))
* **bedrock:** use floor_char_boundary to avoid UTF-8 slice panic in header preview ([82468e4](https://github.com/JerrettDavis/headroom/commit/82468e4e99e6013ab9a44df2e2f12c9352bbc18f)), closes [#415](https://github.com/JerrettDavis/headroom/issues/415)
* **bedrock:** use is_char_boundary loop instead of floor_char_boundary (MSRV 1.80) ([b784f40](https://github.com/JerrettDavis/headroom/commit/b784f400c109b51352825d68341a2801a29123d6))
* **build:** shrink Rust extension wheels — strip + thin-LTO + single codegen unit ([d73cbd6](https://github.com/JerrettDavis/headroom/commit/d73cbd6b0aea20a0e99d29e6c775f26d27f59eb5))
* **build:** shrink Rust extension wheels (strip + thin-LTO + single codegen unit) ([ec8e208](https://github.com/JerrettDavis/headroom/commit/ec8e2080f3231353f11589016a191dd02e100b98))
* C1 — byte-level SSE parser + state machines ([f34e0c2](https://github.com/JerrettDavis/headroom/commit/f34e0c2e8be92d8b7bc6bce2d072bf8280efd7f4))
* C1 — byte-level SSE parser + state machines ([ddc6f6c](https://github.com/JerrettDavis/headroom/commit/ddc6f6ceb0359a7a89b470dc6330543f0cd93f31))
* C2 — /v1/chat/completions Rust handler + OpenAI live-zone ([f9b4491](https://github.com/JerrettDavis/headroom/commit/f9b449196c61ecf1a7947585bd43b6b75daf3e37))
* C2 — /v1/chat/completions Rust handler + OpenAI live-zone ([fe00dc0](https://github.com/JerrettDavis/headroom/commit/fe00dc006aec32f65e1aa6bcf8b2776f37230f4e))
* C3 — /v1/responses Rust HTTP handler + per-item-type passthrough ([70cad6c](https://github.com/JerrettDavis/headroom/commit/70cad6c1c6cc13e6b0794a16ad694ea98979fe9a))
* C3 — /v1/responses Rust HTTP handler + per-item-type passthrough ([57c3e38](https://github.com/JerrettDavis/headroom/commit/57c3e38cb3ccb65744c48927a72238315d154e61))
* C4 — /v1/responses streaming + Conversations API in Rust ([1352f62](https://github.com/JerrettDavis/headroom/commit/1352f621fce223fbcc8cfac0b796d1577a028407))
* C4 — /v1/responses streaming + Conversations API in Rust ([866d346](https://github.com/JerrettDavis/headroom/commit/866d346bd041a581fe39a68e6570b74e2d538fd1))
* **ccr:** key Rust search/diff/log markers with explicit_hash ([#852](https://github.com/JerrettDavis/headroom/issues/852)) ([bfcb07d](https://github.com/JerrettDavis/headroom/commit/bfcb07d78ea7eba539a65b11e100ec23b336d8d1))
* **ccr:** make retrieval TTL configurable ([#715](https://github.com/JerrettDavis/headroom/issues/715)) ([2533f77](https://github.com/JerrettDavis/headroom/commit/2533f7703ee261dc35767b11e46b8eab6e0c454d))
* **ccr:** scope proactive expansion by workspace (cross-project leak) ([197601b](https://github.com/JerrettDavis/headroom/commit/197601bc64ee72e786bf6b94cd90efcac4269bcf))
* **ccr:** scope proactive expansion by workspace (cross-project leak) ([1bc163f](https://github.com/JerrettDavis/headroom/commit/1bc163f5bc1a8422f9ad659061e1fdd8cfeb077b))
* **ccr:** skip CCR when model calls headroom_retrieve alongside user tools ([#839](https://github.com/JerrettDavis/headroom/issues/839)) ([30078f8](https://github.com/JerrettDavis/headroom/commit/30078f8465fb6bb78a5a9c394b75e60cd3c4eeec))
* **ccr:** use shared compression store ([#875](https://github.com/JerrettDavis/headroom/issues/875)) ([249af6c](https://github.com/JerrettDavis/headroom/commit/249af6cc7b379678e60da3e98e552368632fd4f4))
* **ci:** A0 verify must run outside /build cwd ([6b15802](https://github.com/JerrettDavis/headroom/commit/6b15802c0b7f946f5b62d4505c731da3a452821f))
* **ci:** add minimal-privilege permissions blocks to release.yml jobs ([d289c0d](https://github.com/JerrettDavis/headroom/commit/d289c0d4334fc6cb5331d3d2e9e16799e3429193))
* **ci:** correct comments, timeouts, and pip reliability in native e2e workflows ([#878](https://github.com/JerrettDavis/headroom/issues/878)) ([b716c8c](https://github.com/JerrettDavis/headroom/commit/b716c8c2ee7ccc68dd1b9294760db1af866843f2))
* **ci:** docker per-arch bake needs explicit image name in output ([c81755c](https://github.com/JerrettDavis/headroom/commit/c81755c965bd5d57533d75939e1017ad0ff60435))
* **ci:** docker per-arch bake needs explicit image name in output ([8f6bc58](https://github.com/JerrettDavis/headroom/commit/8f6bc5865c33e5bf7e9b5f59986f24d7d98096df))
* **ci:** force-link glibc shim with -Wl,-u for aarch64 ([39cb7c4](https://github.com/JerrettDavis/headroom/commit/39cb7c4dc4102a9e14ecaf9c35deb6bd3186830b))
* **ci:** force-link glibc shim with -Wl,-u so aarch64 wheel includes it ([820e66c](https://github.com/JerrettDavis/headroom/commit/820e66cae65723a7691f7b79707a6f077ac266f0))
* **ci:** glibc shim — drop alias attribute, forward-declare strtol ([af0a960](https://github.com/JerrettDavis/headroom/commit/af0a9604ee5503d1e7b89e8cf41224a306568fce))
* **ci:** glibc shim — drop alias attribute, forward-declare strtol ([6b15acc](https://github.com/JerrettDavis/headroom/commit/6b15acc3b572381260e9c3f068bc5c43bb9eda18))
* **ci:** include NOTICE in sdist + assert License-File metadata matches tarball ([fc6b0a0](https://github.com/JerrettDavis/headroom/commit/fc6b0a0917063df89058d4d01292f43667656c8f))
* **ci:** include NOTICE in sdist + assert License-File metadata matches tarball ([183d51c](https://github.com/JerrettDavis/headroom/commit/183d51c8a860a5ca8d00c091df89c398ffb0425e))
* **ci:** install patchelf for maturin wheel-link repair ([4bf559d](https://github.com/JerrettDavis/headroom/commit/4bf559d5b524e618f3585845bf72c58f8b5f49fd))
* **ci:** multi-stage manylinux build for e2e dockerfiles + release workflow test ([b31a34b](https://github.com/JerrettDavis/headroom/commit/b31a34b4ac00f905c9e319428261be9316ef1c51))
* **ci:** pin cosign-installer to v3 (v4 does not exist) ([#774](https://github.com/JerrettDavis/headroom/issues/774)) ([199d693](https://github.com/JerrettDavis/headroom/commit/199d693f98ecd72d80181c8fee8422b6b64651a2))
* **ci:** pin public PyPI in pyproject.toml + scrub Netflix URLs from uv.lock ([e325d1b](https://github.com/JerrettDavis/headroom/commit/e325d1b86618273abb4177cce672645e0b6a8bd8))
* **ci:** pre-install rustfmt+clippy components in all Dockerfiles ([2ae5772](https://github.com/JerrettDavis/headroom/commit/2ae57725e75ccc5eb57d95e2611bee11f3b0d515))
* **ci:** pypi publish skip-existing to unblock idempotent re-runs ([f94a9d2](https://github.com/JerrettDavis/headroom/commit/f94a9d22bace56af115bb48c38f4539a9bb4a5f4))
* **ci:** pypi publish skip-existing to unblock idempotent re-runs ([6a191b4](https://github.com/JerrettDavis/headroom/commit/6a191b405b5c9bdbcdab4eb1250bea9fa295f18f))
* **ci:** rebuild sdist on the renamed wheel-matrix host ([31bd9f7](https://github.com/JerrettDavis/headroom/commit/31bd9f78e23a6de20cdc318dc26cd774a62d073b))
* **ci:** rebuild sdist on the renamed wheel-matrix host ([75576da](https://github.com/JerrettDavis/headroom/commit/75576dae2bca18610b870b2706c6b54e007dda13))
* **ci:** regenerate uv.lock against public PyPI (was Netflix-internal) ([ce3b2f0](https://github.com/JerrettDavis/headroom/commit/ce3b2f0b0ba07bb43012f162e75ccbdc2944a95d))
* **ci:** reorder Dockerfile so wheel installs before headroom-ai ([f1aa12c](https://github.com/JerrettDavis/headroom/commit/f1aa12cebfbd5ae4657133ec11efecf7ad6f433c))
* **ci:** replace ls with find in smoke-import wheel discovery ([3e78421](https://github.com/JerrettDavis/headroom/commit/3e784212673a6ffc1b82bcb3f478998c6ff8433b))
* **ci:** restore green lint gate on main ([fe50f9d](https://github.com/JerrettDavis/headroom/commit/fe50f9daed35151134f79b767733d4be8093e325))
* **ci:** rustls-everywhere — eliminate openssl-sys from build tree ([6f2c0a8](https://github.com/JerrettDavis/headroom/commit/6f2c0a84003a0d4cee0408bee7a94b8cbb6502ae))
* **ci:** rustls-everywhere — eliminate openssl-sys from build tree (kills the wheel-build cascade) ([e731f87](https://github.com/JerrettDavis/headroom/commit/e731f87c5d6d790b1d94298a36de7658b25f6b34))
* **ci:** skip smoke OpenAI eval cleanly when OPENAI_API_KEY secret unset ([a281de6](https://github.com/JerrettDavis/headroom/commit/a281de6bb0279c05a2b44cc868bde9a0428ba82f))
* **ci:** smoke-import wheels on customer-representative envs before publish (X1) ([596212b](https://github.com/JerrettDavis/headroom/commit/596212b428ba2019885b49381ffdf9cb309ec8e3))
* **ci:** stage smoke-import script as host file (broke main post-[#387](https://github.com/JerrettDavis/headroom/issues/387)) ([7fe2d1e](https://github.com/JerrettDavis/headroom/commit/7fe2d1e5b632c3e5424f5364baef0ef8af6510a4))
* **ci:** stage smoke-import script as host file (broke main) ([e536d9b](https://github.com/JerrettDavis/headroom/commit/e536d9b1801652ff6d4b9bce9683f155029b23c7))
* **ci:** switch e2e runtime to python:3.11-slim (trixie, glibc 2.41) ([73a4782](https://github.com/JerrettDavis/headroom/commit/73a4782917cf393bfb71e240cc66e6629ef31702))
* **ci:** unblock A0 Docker e2e — install pkg-config + opt out wrap-e2e ([55dfc19](https://github.com/JerrettDavis/headroom/commit/55dfc19e1debd9a5f4bb3ec2a632d606452e2bd7))
* **ci:** unbreak release pipeline — wheel openssl + dead npm artifact downloads ([75fa4a4](https://github.com/JerrettDavis/headroom/commit/75fa4a42f5cec62c969a8ef242116e38474189eb))
* **ci:** unbreak release pipeline — wheel openssl + dead npm artifact downloads ([7ba47e2](https://github.com/JerrettDavis/headroom/commit/7ba47e257b599856e011a1dc7abcd3c0dfb69d77))
* **ci:** update sdist license-packaging invariant test to match new shape ([cd89a82](https://github.com/JerrettDavis/headroom/commit/cd89a8297ae55fe26343fc101b695953f07969de))
* **ci:** update sdist license-packaging invariant test to match new shape ([37c0df2](https://github.com/JerrettDavis/headroom/commit/37c0df23b6cc7bf3e1fbc4f7ad67499a61eb539e))
* **ci:** update tests to assert absence of requires_openai_auth (bug 3, [#406](https://github.com/JerrettDavis/headroom/issues/406)) ([4071d57](https://github.com/JerrettDavis/headroom/commit/4071d571345c5a88df56eac5ad975200831e9b7d))
* **ci:** use ghcr devcontainer rust feature instead of manual rustup install ([9ce61af](https://github.com/JerrettDavis/headroom/commit/9ce61afcd4d8b48a5e3dac32cd0d89ee599532db))
* **ci:** vendor OpenSSL via cargo + drop x86_64 macOS from wheel matrix ([20e482d](https://github.com/JerrettDavis/headroom/commit/20e482da8fc16cc6b6ae022fbdd709a98ce70e4a))
* **ci:** vendor OpenSSL via cargo + drop x86_64 macOS from wheel matrix ([1314842](https://github.com/JerrettDavis/headroom/commit/1314842b1901b7465bb8d2057a3123402c208fc0))
* **ci:** vendored OpenSSL must live in headroom-py, not headroom-proxy ([f34f95a](https://github.com/JerrettDavis/headroom/commit/f34f95afe9170adcab3701e861d740ec141fd624))
* **ci:** vendored OpenSSL must live in headroom-py, not headroom-proxy ([a5c7f6f](https://github.com/JerrettDavis/headroom/commit/a5c7f6fed9af0f96ba8f02dbf161d3d969169d32))
* **ci:** wheel before-script must work on Debian aarch64-cross container ([1323830](https://github.com/JerrettDavis/headroom/commit/1323830f70acdc754079d6c11fbb18146de40d55))
* **ci:** wheel build before-script-linux must work on Debian aarch64-cross ([cf4ea02](https://github.com/JerrettDavis/headroom/commit/cf4ea0243295f2703093e0b72d3f81eb75c4e81a))
* **ci:** X1 — smoke-import wheels on customer-representative envs before publish ([4745219](https://github.com/JerrettDavis/headroom/commit/4745219901003be6607df2219e4ce672187b8f27))
* **ci:** yarnpkg GPG + YAML colon syntax in refactor ([86177da](https://github.com/JerrettDavis/headroom/commit/86177da871cc9e39999353cdcee67c1a8d2ab887))
* clear CI mypy + rust test failures introduced in eaf5980 ([17ffae0](https://github.com/JerrettDavis/headroom/commit/17ffae0cd8e4498bf60f4fa5a36521905f9fb151))
* **cli:** G1 remediation — non-string clobber, per-model systemMessage, openhands gate ([ea1976e](https://github.com/JerrettDavis/headroom/commit/ea1976e37a5147ecf37dbf5ffe4af5c2f2d1be6a))
* **cli:** proxy/perf/wrap UX cleanup + perf --hours correctness ([9ff28e9](https://github.com/JerrettDavis/headroom/commit/9ff28e9803eaba3c1be9c20924d7b34d98f507b0))
* **cli:** proxy/perf/wrap UX cleanup + perf --hours correctness ([265554d](https://github.com/JerrettDavis/headroom/commit/265554d4adee7fbf03234a6de832293dec95f811))
* **cli:** resolve duplicate --code-aware flag breaking proxy import ([5dd2ac5](https://github.com/JerrettDavis/headroom/commit/5dd2ac5e516b9f429648ca5a48be9bb1a6bacb0b))
* **cli:** wrap CLI breadth — cline, continue, goose, openhands ([8625f80](https://github.com/JerrettDavis/headroom/commit/8625f8075ed75d2a002f6ba357697de0fa1ec434))
* **cli:** wrap subcommands for cline, continue, goose, openhands ([c375fa1](https://github.com/JerrettDavis/headroom/commit/c375fa156dd0434256805f274c07be4f45db9814))
* Codex ws cancel logging ([73b0e56](https://github.com/JerrettDavis/headroom/commit/73b0e56e970c6f805a1d174f50b886000372cdb2))
* **codex:** auto-enable fail-open on compression timeout in headroom wrap codex ([#531](https://github.com/JerrettDavis/headroom/issues/531)) ([5f5f261](https://github.com/JerrettDavis/headroom/commit/5f5f261a035d12d069eb212eb75c472e2c9edeff))
* **codex:** bug 3 — strip requires_openai_auth, restore consistent openai_base_url injection ([6c4ddc8](https://github.com/JerrettDavis/headroom/commit/6c4ddc824e074245ffd6ea8002a068cb108f27fb))
* **codex:** drop env_key from provider blocks to preserve subscription auth ([32f499c](https://github.com/JerrettDavis/headroom/commit/32f499cbba4c370865105a55748a365294ec5815))
* **codex:** fail open for proxy compression timeout ([e8ecd08](https://github.com/JerrettDavis/headroom/commit/e8ecd0882990e464f9b4a7d2041ce86863931e83))
* **codex:** inject openai_base_url in init and persistent-install paths ([bf1e31b](https://github.com/JerrettDavis/headroom/commit/bf1e31b27c64ad9400d2997ce1f70692ebb408d0))
* **codex:** keep init model_provider at config root ([#260](https://github.com/JerrettDavis/headroom/issues/260)) ([304dcc7](https://github.com/JerrettDavis/headroom/commit/304dcc78047bc744fc2f7656b484ec54dc271354))
* **codex:** keep init model_provider at config root ([#260](https://github.com/JerrettDavis/headroom/issues/260)) ([849b46d](https://github.com/JerrettDavis/headroom/commit/849b46de5934a88369af2fd7f7d52e9af0536a7e))
* **codex:** respect CODEX_HOME for wrap config ([#731](https://github.com/JerrettDavis/headroom/issues/731)) ([96abf38](https://github.com/JerrettDavis/headroom/commit/96abf38b0972adf5e5c66f9a49aa9d9f951b1aa0))
* **codex:** restore openai_base_url top-level injection for subscription routing ([d54c5b6](https://github.com/JerrettDavis/headroom/commit/d54c5b6a5840fabf0d28079fe67fb0347f016fbb))
* **codex:** strip requires_openai_auth and openai_base_url injection (bug 3, [#406](https://github.com/JerrettDavis/headroom/issues/406)) ([09b8510](https://github.com/JerrettDavis/headroom/commit/09b851001bf6a2e6ccc34478e42f54c87ddaf5d3))
* **codex:** write canonical hooks feature flag and migrate deprecated codex_hooks ([#743](https://github.com/JerrettDavis/headroom/issues/743)) ([dff6a19](https://github.com/JerrettDavis/headroom/commit/dff6a19946b8f96bb8b16fa945b69a1ed09709af))
* complete /v1/responses compression telemetry, multi-frame WS, frozen-count cap ([89f7b6c](https://github.com/JerrettDavis/headroom/commit/89f7b6c2dd0f3503312490b2052f534e76c01ac3))
* Compress Codex Responses payloads ([d90d2ca](https://github.com/JerrettDavis/headroom/commit/d90d2caed3b1c3199dd89714a7f0a330b23db59f))
* **content_router:** guard against empty compression output causing Anthropic 400 ([#771](https://github.com/JerrettDavis/headroom/issues/771)) ([2f9ff07](https://github.com/JerrettDavis/headroom/commit/2f9ff07e6caef0fe32d00ece6266a476eecff5a3))
* **content-router,proxy:** cache-safe text-block compression and online streaming usage ([ecb3e00](https://github.com/JerrettDavis/headroom/commit/ecb3e00451f5fd62a2be9d924996c2f075a3de2d))
* **content-router,proxy:** cache-safe text-block compression and online streaming usage ([79baee0](https://github.com/JerrettDavis/headroom/commit/79baee082f894cf0c467468eed1ae2fed04f37e6))
* **content-router,proxy:** compress text blocks + close DeepSeek metrics gaps ([d322e6d](https://github.com/JerrettDavis/headroom/commit/d322e6df1fc0f949469242488c5c0080beb206a1))
* **content-router,proxy:** compress text blocks + close DeepSeek metrics gaps ([d955abb](https://github.com/JerrettDavis/headroom/commit/d955abb1d65a14e07b5c6a915cafc12ddb12ace7))
* **copilot:** deterministic subscription token handoff to the proxy ([72da461](https://github.com/JerrettDavis/headroom/commit/72da46121726074515e0c1eb9745498457a1a8d5))
* **copilot:** restore generic endpoint for non-subscription OAuth ([#610](https://github.com/JerrettDavis/headroom/issues/610)) ([#612](https://github.com/JerrettDavis/headroom/issues/612)) ([18925b8](https://github.com/JerrettDavis/headroom/commit/18925b8c6e343c9d593891cd29ac27fee1cb9836))
* **copilot:** support subscription auth through Headroom ([ff4a0c6](https://github.com/JerrettDavis/headroom/commit/ff4a0c6bc64e5e68ab76c38047a36a3c7a6aaacf))
* **copilot:** use responses API for subscription reasoning models ([#647](https://github.com/JerrettDavis/headroom/issues/647)) ([84ac332](https://github.com/JerrettDavis/headroom/commit/84ac332d14dafacedc2f0b46f5ac6b3977b098d0))
* **core:** expose compress_openai_responses_live_zone via PyO3 (hot-fix c1/2) ([c48735d](https://github.com/JerrettDavis/headroom/commit/c48735d0298b94d25d40397ea64a2d66d1e557d0))
* **core:** F2.2 c1/3 — extend CompressionPolicy with three per-mode tuning fields ([797dc63](https://github.com/JerrettDavis/headroom/commit/797dc63da747dd8594958709271abd658de4a5c0))
* **core:** introduce CompressionPolicy struct + auth_mode mapping (F2.1 c1/6) ([8376630](https://github.com/JerrettDavis/headroom/commit/837663006254cde327e9c7f71b2ea5682f9b902e))
* correct preserved-entry index mapping in Gemini content round-trip ([#836](https://github.com/JerrettDavis/headroom/issues/836)) ([0ffe2b6](https://github.com/JerrettDavis/headroom/commit/0ffe2b6ea49e5c8d3bff5fe2c90873c71a95c457))
* correct tiktoken encoding for unknown gpt-4 model snapshots ([#552](https://github.com/JerrettDavis/headroom/issues/552)) ([0e551de](https://github.com/JerrettDavis/headroom/commit/0e551de9d81021bb7f0dde1857a2341408606969))
* **crusher:** bridge SmartCrusher row-drop hash to Python compression_store ([#389](https://github.com/JerrettDavis/headroom/issues/389)) ([e664f11](https://github.com/JerrettDavis/headroom/commit/e664f11260e997c9e643feec8b2cff27f127660f))
* **crusher:** bridge SmartCrusher row-drop hash to Python compression_store ([#389](https://github.com/JerrettDavis/headroom/issues/389)) ([a40d4a5](https://github.com/JerrettDavis/headroom/commit/a40d4a5f6d3183c94941a7ffe8c28bcb5d769792))
* **crusher:** shim __libc_single_threaded for glibc &lt; 2.32 + extend audit ([19a43f0](https://github.com/JerrettDavis/headroom/commit/19a43f0d52a0b406b1e0cae2804fe0e204b4dcba))
* **crusher:** shim __libc_single_threaded for glibc &lt; 2.32 + extend audit ([17d6207](https://github.com/JerrettDavis/headroom/commit/17d6207bf544ddb670236be67d4530c8913e735f))
* **crusher:** switch compression_store cache key from MD5 to SHA-256 (CodeQL [#395](https://github.com/JerrettDavis/headroom/issues/395)) ([af9e628](https://github.com/JerrettDavis/headroom/commit/af9e6282f89ff4fb2e0052fd58314d407997b7e6))
* **dashboard:** rebind hero tile to file-backed savings, add cross-process lock ([a8b62a7](https://github.com/JerrettDavis/headroom/commit/a8b62a77d9068372c55f70d74721f2a73774caf3))
* **dashboard:** stable 'Proxy $ Saved' hero tile under --workers &gt; 1 ([#481](https://github.com/JerrettDavis/headroom/issues/481)) ([fd73b88](https://github.com/JerrettDavis/headroom/commit/fd73b88368b22beeb586b8e1aa37fcd2afb12532))
* decode/encode owned config, state and template assets as UTF-8 ([2f1538a](https://github.com/JerrettDavis/headroom/commit/2f1538a641dd0e60a7be3de85646a70c4bf7e287))
* decode/encode owned config, state and template assets as UTF-8 (fixes [#533](https://github.com/JerrettDavis/headroom/issues/533)) ([92075b9](https://github.com/JerrettDavis/headroom/commit/92075b95af799951c90a305a08ec4e958473967a))
* **deps:** move gunicorn to [proxy-prod] extra, add Windows guard ([#537](https://github.com/JerrettDavis/headroom/issues/537)) ([fa558c5](https://github.com/JerrettDavis/headroom/commit/fa558c5647a91562f4a8fba0271d27b02c8ae01f))
* **docker:** upgrade base images to Python 3.13 / debian13 ([e6bf7a0](https://github.com/JerrettDavis/headroom/commit/e6bf7a03fef8a9f2e4802d63afdafb40627c7ad9))
* **docker:** upgrade base images to Python 3.13 / debian13, drop digest pinning ([08a2197](https://github.com/JerrettDavis/headroom/commit/08a219708c97dcdc678483a0e6891306624a1fad))
* **docs:** bump next.js to 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([a6a09e6](https://github.com/JerrettDavis/headroom/commit/a6a09e6cfbe6962a70a6fb2e4bebeee80756e304))
* **docs:** mkdocs configuration to build with correct folder ([#543](https://github.com/JerrettDavis/headroom/issues/543)) ([5557944](https://github.com/JerrettDavis/headroom/commit/55579445f84c363219f45dc5358599a04d4263ed))
* **docs:** update brace-expansion to 5.0.6 to remediate GHSA-jxxr-4gwj-5jf2 (CVE-2026-45149) ([6eb6fb5](https://github.com/JerrettDavis/headroom/commit/6eb6fb5941adfbd056daa1689c3fa0c3755fd298))
* **docs:** update bun.lock to next 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([91e0937](https://github.com/JerrettDavis/headroom/commit/91e0937243c801fa5f1021b4c47debef2444650c))
* don't inject empty tools:[] when client omitted the tools field ([#772](https://github.com/JerrettDavis/headroom/issues/772)) ([574bbae](https://github.com/JerrettDavis/headroom/commit/574bbae2cbe2f20b3f0e12b421c25ac256712f0a))
* **e2e:** pin marketplace source via env var in init Dockerfile ([9ae696e](https://github.com/JerrettDavis/headroom/commit/9ae696eb752f1fc58b3465c4cc8a3d4dd6add2a0))
* expose code-aware flag ([7d99a71](https://github.com/JerrettDavis/headroom/commit/7d99a71285b3b3ec5c475c0676aa1a4a0887fbe7))
* expose compression latency bottlenecks ([4e146b0](https://github.com/JerrettDavis/headroom/commit/4e146b0b5a12d1040bb8dddd88b3445fc6b2af87))
* expose compression latency bottlenecks ([e9cae01](https://github.com/JerrettDavis/headroom/commit/e9cae0131bfe0c3b6db9096b2c5604259bda673d))
* format Codex compression changes ([03d12fc](https://github.com/JerrettDavis/headroom/commit/03d12fc14039d64fc6b53507a28308177661ab41))
* format Kompress tests for ruff ([fc0cba7](https://github.com/JerrettDavis/headroom/commit/fc0cba7b48cf7cefbef1968f57527bdf5c2fe652))
* harden Copilot API auth token handling ([#557](https://github.com/JerrettDavis/headroom/issues/557)) ([6b0c09f](https://github.com/JerrettDavis/headroom/commit/6b0c09ffd5f2ce18c4d2cfa6233feaf37d487ead))
* harden learn path handling across platforms ([9577ef8](https://github.com/JerrettDavis/headroom/commit/9577ef811b3f078959fe821b39304977877f6f09))
* harden learn path handling across platforms ([5ceca13](https://github.com/JerrettDavis/headroom/commit/5ceca13c656165209dacc35d1dc2cc0aebd1afe1))
* harden release gating and clarify pipx compatibility ([9492386](https://github.com/JerrettDavis/headroom/commit/9492386398408ae6652aab58966fb16f4f26c2c8))
* harden release gating and clarify pipx Python compatibility ([a7e1f93](https://github.com/JerrettDavis/headroom/commit/a7e1f931a5511294c1e3daa6992f72d195373677))
* **health:** readyz verifies upstream connectivity, not just process liveness ([#744](https://github.com/JerrettDavis/headroom/issues/744)) ([5dfb446](https://github.com/JerrettDavis/headroom/commit/5dfb446da1fb65002e0dea18a90210a2a026f0b3))
* ignore brackets inside JSON strings when splitting mixed content ([#553](https://github.com/JerrettDavis/headroom/issues/553)) ([bdcfc32](https://github.com/JerrettDavis/headroom/commit/bdcfc322da0c4cde69931d641cfa18c76ddb138b))
* **init:** guard persistent task startup ([#616](https://github.com/JerrettDavis/headroom/issues/616)) ([9252d85](https://github.com/JerrettDavis/headroom/commit/9252d852c5a4c716eb5438b8f438d50e59a55fef))
* **init:** normalize Windows hook paths to forward slashes ([#788](https://github.com/JerrettDavis/headroom/issues/788)) ([6ea6e31](https://github.com/JerrettDavis/headroom/commit/6ea6e31f09845b2ad5c8bae73bcf353f3b629188))
* **init:** suppress hook recovery output ([#760](https://github.com/JerrettDavis/headroom/issues/760)) ([b439599](https://github.com/JerrettDavis/headroom/commit/b4395993aecbb65b85a5b2479dfdb35ea243bf54))
* inject openai_base_url so Codex subscription (ChatGPT plan) routes through proxy ([f0b40f1](https://github.com/JerrettDavis/headroom/commit/f0b40f11d5dfdfa41debb069478f9ad283d753d6))
* inject openai_base_url so Codex subscription (ChatGPT plan) routes through proxy ([5c7a5b4](https://github.com/JerrettDavis/headroom/commit/5c7a5b4857e4259ff16d272b1ba0308f3ac45282))
* integrate B6+B7 — fix cross-test contamination + injector mock parity ([2fb905f](https://github.com/JerrettDavis/headroom/commit/2fb905fdb016d354fa4938bd9fcb8e2654865998))
* **learn:** claude-cli streams output with idle timeout ([#373](https://github.com/JerrettDavis/headroom/issues/373)) ([9bff575](https://github.com/JerrettDavis/headroom/commit/9bff5752bbd769902f249cdfde42bc53539afd02))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([211daae](https://github.com/JerrettDavis/headroom/commit/211daae25687901d1f893714d877b25606d0ef69))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([491a8b3](https://github.com/JerrettDavis/headroom/commit/491a8b3a1b260f42f503b3553a04c578c18e1cc0))
* **learn:** finish gemini-flash-latest default model sweep ([982d01b](https://github.com/JerrettDavis/headroom/commit/982d01b9c996fd5fe26154dc2f94d567192f6ff6))
* **learn:** finish gemini-flash-latest default model sweep ([#532](https://github.com/JerrettDavis/headroom/issues/532)) ([d797366](https://github.com/JerrettDavis/headroom/commit/d7973665f4e2f40f2b3acadd0ec584609fb33c6c))
* log Codex ws cancellations safely ([62a1f23](https://github.com/JerrettDavis/headroom/commit/62a1f23b8864f9b0a82dbab6225514f73b7ada9d))
* make headroom wrap readiness probe timeout configurable for slow ML imports ([#581](https://github.com/JerrettDavis/headroom/issues/581)) ([163677b](https://github.com/JerrettDavis/headroom/commit/163677b405d7ca8a54d6d7c798bf6ead90da7880))
* make proxy upgrades version-aware ([2732ae7](https://github.com/JerrettDavis/headroom/commit/2732ae7d820451bafdefceb76513c4eba109b29b))
* make proxy upgrades version-aware ([ea1f608](https://github.com/JerrettDavis/headroom/commit/ea1f608e79d8809ff4d7d2428972fdb633a01918))
* **mcp:** auto-register headroom MCP server in wrap claude/codex and init -g ([d9d8972](https://github.com/JerrettDavis/headroom/commit/d9d8972ac4f51c0f018c470b8dba337cb959074d))
* **memory:** drop regex-based pref extraction; filter system-reminder noise (refs [#464](https://github.com/JerrettDavis/headroom/issues/464)) ([1139b21](https://github.com/JerrettDavis/headroom/commit/1139b21f2a9a39803bcc11b981c73890980508ee))
* **memory:** expose memory IDs in auto-tail + memory_list tool + ID-usage guidance ([f844f64](https://github.com/JerrettDavis/headroom/commit/f844f64840491d9a838be4b00a4ca6d9ff97adba))
* **memory:** expose memory IDs in auto-tail + memory_list tool + ID-usage guidance ([c62d45e](https://github.com/JerrettDavis/headroom/commit/c62d45eea826c661aa8ebf1f2c8aba8408ea6109))
* **memory:** per-project storage so projects can no longer bleed memories (refs [#462](https://github.com/JerrettDavis/headroom/issues/462)) ([f8ffdbf](https://github.com/JerrettDavis/headroom/commit/f8ffdbfe7369d49c23c67ca9a3c4668238aab508))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([a178249](https://github.com/JerrettDavis/headroom/commit/a178249fc0af4a1b6f212decb4f6d2793d57fae8))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([482f80e](https://github.com/JerrettDavis/headroom/commit/482f80e735f124ee6860f6854255c77170b862e7))
* **memory:** traffic_learner indexes system-reminder fragments as user preferences (refs [#464](https://github.com/JerrettDavis/headroom/issues/464)) ([0be0eed](https://github.com/JerrettDavis/headroom/commit/0be0eede9e09466e4ec0c0ee10b1d0aa317e0e73))
* narrow compressed type for mypy 1.14 in ContentRouter.compress ([05fe6c0](https://github.com/JerrettDavis/headroom/commit/05fe6c010222921103651ebc3f9f4715a0418780))
* **observability:** G3 remediation — bound cardinality + wire dead metrics ([2a717a9](https://github.com/JerrettDavis/headroom/commit/2a717a993ee99f9401f5cdf78a23dcecd7cb1a51))
* **observability:** RTK metrics + Rust observability (Phase H blocker) ([b36ad9f](https://github.com/JerrettDavis/headroom/commit/b36ad9fe1c6a488eb9ffbf0e8b38d989278cf8ef))
* **observability:** wire Phase G PR-G3 RTK + proxy metrics (H-blocker) ([5f264a5](https://github.com/JerrettDavis/headroom/commit/5f264a53292e292c9c56b837c2750d1a415b1ea9))
* **parser:** detect waste signals in Anthropic tool_result content blocks ([#815](https://github.com/JerrettDavis/headroom/issues/815)) ([929698a](https://github.com/JerrettDavis/headroom/commit/929698af1030e5926f3766d7d6ac292d6e38437b))
* per-project memory storage so projects can no longer bleed memories (GH [#462](https://github.com/JerrettDavis/headroom/issues/462)) ([7694f05](https://github.com/JerrettDavis/headroom/commit/7694f050fe3b4d80c82903f17f38a93b4b3d670e))
* Phase A+B realignment — cache safety + live-zone-only compression + production hotfixes ([9266ea7](https://github.com/JerrettDavis/headroom/commit/9266ea711153c2bb7487b0c73dd9f7bfe1fb22c3))
* populate Codex WS dashboard performance metrics ([4279a7f](https://github.com/JerrettDavis/headroom/commit/4279a7f35349bde2b0361e9fe6e9f3a75adbf5de))
* PR [#281](https://github.com/JerrettDavis/headroom/issues/281) — synthesize 5h subscription window after Anthropic reset ([27adb1d](https://github.com/JerrettDavis/headroom/commit/27adb1da57c4cd45804aaea3942362e3a9d09289))
* PR [#281](https://github.com/JerrettDavis/headroom/issues/281) — synthesize 5h subscription window after Anthropic reset (no extra polling) ([0b77955](https://github.com/JerrettDavis/headroom/commit/0b77955ecb5b64ab4f8224f003dc532cf71d5909))
* PR [#372](https://github.com/JerrettDavis/headroom/issues/372) — restore [image] extra on Python 3.13 via rapidocr 3.x adapter ([486372b](https://github.com/JerrettDavis/headroom/commit/486372b6445bd217a6306159bb7210173d0578ec))
* PR [#372](https://github.com/JerrettDavis/headroom/issues/372) — restore [image] extra on Python 3.13 via rapidocr 3.x adapter ([b154e17](https://github.com/JerrettDavis/headroom/commit/b154e178533e51cf95fb4410812ee01fbe3975df))
* PR-C5 retire responses_converter.py — Rust owns /v1/responses ([76c113d](https://github.com/JerrettDavis/headroom/commit/76c113dd4a2ed210cac301b5bd46961dd34f1d0d))
* PR-C5 retire responses_converter.py — Rust owns /v1/responses ([221109d](https://github.com/JerrettDavis/headroom/commit/221109d95e6f18041fe664144710ae32f00401ad))
* PR-D1 native Bedrock InvokeModel route + SigV4 ([dd1dadf](https://github.com/JerrettDavis/headroom/commit/dd1dadfe8a4b8041cd9c6588a5b075fc4d24b4f0))
* PR-D2 Bedrock streaming via binary EventStream ([32a1fd4](https://github.com/JerrettDavis/headroom/commit/32a1fd4fe0dc5003ee479229890b52547b455ba8))
* PR-D3 Bedrock observability + auth-mode integration (Phase D close) ([0c46c7e](https://github.com/JerrettDavis/headroom/commit/0c46c7e01070a3efa2fcfb420665fa0d420eeb58))
* PR-D4 native Vertex publisher path + ADC ([d3c70ac](https://github.com/JerrettDavis/headroom/commit/d3c70acea5d2d893e6490b293a6b2f931a57a1d3))
* PR-E1 + PR-E2 tool array sort + schema-key sort (Phase E) ([c8e0b36](https://github.com/JerrettDavis/headroom/commit/c8e0b367572d6c38dc4807955599e32b90248001))
* PR-E1 tool array deterministic sort (Phase E) ([4a3b76b](https://github.com/JerrettDavis/headroom/commit/4a3b76bcc879b3ba4a6a29096b5b50cb87d34f40))
* PR-E2 recursive JSON Schema key sort (Phase E) ([9112fed](https://github.com/JerrettDavis/headroom/commit/9112fed9379cfe43d31e935c9ac855116abd60f9))
* PR-E3 Anthropic cache_control auto-placement (Phase E) ([397c805](https://github.com/JerrettDavis/headroom/commit/397c805698d01da20256d651a92a17429327cf6a))
* PR-E3 Anthropic cache_control auto-placement (Phase E) ([8672d5c](https://github.com/JerrettDavis/headroom/commit/8672d5c32680c790bfb6c431c2d1a854fac6566b))
* PR-E4 OpenAI prompt_cache_key auto-injection (Phase E) ([2fc73d0](https://github.com/JerrettDavis/headroom/commit/2fc73d0f73241a937304a8b8d4e11c550945e6df))
* PR-E4 OpenAI prompt_cache_key auto-injection (Phase E) ([573543f](https://github.com/JerrettDavis/headroom/commit/573543fce7dcf461d8a65bbfde4a6f9a93cee351))
* PR-E5 volatile-content detector + customer warning (Phase E) ([d1b836d](https://github.com/JerrettDavis/headroom/commit/d1b836d78c5b92ec12b565e58e47aaaa3e7bc209))
* PR-E5 volatile-content detector + customer warning (Phase E) ([d8aae38](https://github.com/JerrettDavis/headroom/commit/d8aae382b00c0d340dd79ef89963126156f2f805))
* PR-E6 cache-bust drift detector telemetry (Phase E) ([41ca054](https://github.com/JerrettDavis/headroom/commit/41ca0544c793731522a26a916bc351cc18a460bf))
* PR-E6 cache-bust drift detector telemetry (Phase E) ([ce37940](https://github.com/JerrettDavis/headroom/commit/ce37940d17e149bf6961742472a61885de5f2e73))
* PR-F1 classify_auth_mode helper (Phase F kickoff) ([fb25a26](https://github.com/JerrettDavis/headroom/commit/fb25a26180a4fb508e9e090f716f340af4a1ed03))
* PR-F1 classify_auth_mode helper (Phase F kickoff) ([ca9de93](https://github.com/JerrettDavis/headroom/commit/ca9de93cfc4c5e8dd4e017615aedad3249ea8c1d))
* preserve Claude Code tool-search deferral through the proxy ([#746](https://github.com/JerrettDavis/headroom/issues/746)) ([#753](https://github.com/JerrettDavis/headroom/issues/753)) ([1c8c538](https://github.com/JerrettDavis/headroom/commit/1c8c538780d5c9a43781f1fdd9eeda2657280446))
* preserve Codex OAuth provider config ([7e29a60](https://github.com/JerrettDavis/headroom/commit/7e29a60b6feca5848ce1a07119294d45a472c04e))
* preserve Codex OAuth proxy delivery ([06428d2](https://github.com/JerrettDavis/headroom/commit/06428d20fd0514375024a6784b30b873a0f8d82d))
* **providers:** register claude-opus-4-7 + [1m] tier with 1M context ([2e9f52f](https://github.com/JerrettDavis/headroom/commit/2e9f52fda2e81f3c3bafc3e2562e2e2ef5b321f2))
* **proxy,dashboard:** correct savings reporting ([71348f4](https://github.com/JerrettDavis/headroom/commit/71348f407f5d3463165a44665bc48e40f7b29a69))
* **proxy,dashboard:** correct savings reporting for backend-routed traffic and drop broken Compression Quality widget ([6643266](https://github.com/JerrettDavis/headroom/commit/6643266ec02abbdb54cd2910b3a23845c144805f))
* **proxy:** add auth_mode_policy_enforcement feature flag (F2.1 c3/6) ([027b203](https://github.com/JerrettDavis/headroom/commit/027b203c2352693b53a54d9e7763cc61c19e2f20))
* **proxy:** bound Codex Responses compression work ([160989c](https://github.com/JerrettDavis/headroom/commit/160989c43ecbe86d36c1c69681c578276c756285))
* **proxy:** cache concurrency lock, multi-worker docs, bounded compre… ([51eeaf6](https://github.com/JerrettDavis/headroom/commit/51eeaf6662816375244c8a09009b5b77dee92d7d))
* **proxy:** F2.1 — per-auth-mode CompressionPolicy gates ([3b1ee2e](https://github.com/JerrettDavis/headroom/commit/3b1ee2e4af9eba4318e748d65be88bb04daef81c))
* **proxy:** F2.2 — per-mode CompressionPolicy tuning fields ([294df2b](https://github.com/JerrettDavis/headroom/commit/294df2b894a985c8501981de35865aa3245fbdc2))
* **proxy:** F4 — trust X-Forwarded-* only behind allow-listed gateway ([d10bd5f](https://github.com/JerrettDavis/headroom/commit/d10bd5f59c5a36e14f6c5f0480b821532521b753))
* **proxy:** F4 — trust X-Forwarded-* only behind allow-listed gateway ([665c082](https://github.com/JerrettDavis/headroom/commit/665c0823d96b461f6127debe2d19df4e91c7c918))
* **proxy:** fail-open on corrupt golden bytes instead of RuntimeError ([#603](https://github.com/JerrettDavis/headroom/issues/603)) ([2170a1b](https://github.com/JerrettDavis/headroom/commit/2170a1b4a00e9c46e845993c9b0f6cb2ef0c0684))
* **proxy:** hoist compression_policy outside is_token_mode branch (F2.1 c5 followup) ([708b87a](https://github.com/JerrettDavis/headroom/commit/708b87a19fbe0c71ab5945ef8fda154f0abd2eef))
* **proxy:** hot-fix Codex /v1/responses compression via PyO3 inline call ([1a3098a](https://github.com/JerrettDavis/headroom/commit/1a3098a48f8021e00c1fb118b1a5dab0b8add559))
* **proxy:** inline WebSocket /v1/responses compression via PyO3 ([1050e00](https://github.com/JerrettDavis/headroom/commit/1050e00241960e475e63badd59c175abc5b3e7cf))
* **proxy:** inline WebSocket /v1/responses compression via PyO3 ([4a50313](https://github.com/JerrettDavis/headroom/commit/4a503135487182523a9db559cff23cd42427ca64))
* **proxy:** lazy-import server to avoid fastapi crash ([#442](https://github.com/JerrettDavis/headroom/issues/442)) ([93c6937](https://github.com/JerrettDavis/headroom/commit/93c69372e614f2b04873bed75602a88d2256a7fc))
* **proxy:** make CCR multi-worker warning conditional on backend ([#770](https://github.com/JerrettDavis/headroom/issues/770)) ([d76a729](https://github.com/JerrettDavis/headroom/commit/d76a7296df121365d74c415b8c702a3ad80abd30))
* **proxy:** make Kompress eager preload cache-only so a cold cache can't block startup ([#783](https://github.com/JerrettDavis/headroom/issues/783)) ([841663d](https://github.com/JerrettDavis/headroom/commit/841663da16971b1e0d8e204fdf18e4bafedaf9e0))
* **proxy:** MemoryDecision contract + 3 bypass bugs + drop 500-char query cap ([4e1b218](https://github.com/JerrettDavis/headroom/commit/4e1b21854456431253952ec4d32b8464133cc667))
* **proxy:** MemoryDecision contract + 3 bypass bugs + drop 500-char query cap ([71d5a7b](https://github.com/JerrettDavis/headroom/commit/71d5a7b5455f92bd07c1cfc95909738687672307))
* **proxy:** plumb CompressionPolicy through proxy + dispatchers (F2.1 c2/6) ([948c8f2](https://github.com/JerrettDavis/headroom/commit/948c8f2069c6c88eaba40196f2235cd279e2064f))
* **proxy:** PR-D1 native Bedrock InvokeModel route + SigV4 ([f2d4fe3](https://github.com/JerrettDavis/headroom/commit/f2d4fe39cb9f533e5928201c2b17e50a625c1996))
* **proxy:** PR-D2 Bedrock streaming via binary EventStream ([66426e7](https://github.com/JerrettDavis/headroom/commit/66426e7b75a730411c29df6501f49f5d6f5b8ab5))
* **proxy:** PR-D3 Bedrock observability + auth-mode integration ([90ef662](https://github.com/JerrettDavis/headroom/commit/90ef66213d8566925ce0b021d84fd094fab79340))
* **proxy:** PR-D4 native Vertex publisher path + ADC bearer auth ([c10a219](https://github.com/JerrettDavis/headroom/commit/c10a2195afdf86e0ade17fad53a509de577224d4))
* **proxy:** record cache reads/writes on backend-routed streaming ([#327](https://github.com/JerrettDavis/headroom/issues/327)) ([7ddc0ce](https://github.com/JerrettDavis/headroom/commit/7ddc0cef1eb7b58021fcb57f6c2a7f00a6f090c7))
* **proxy:** record cache reads/writes on backend-routed streaming ([#327](https://github.com/JerrettDavis/headroom/issues/327)) ([3e97a0c](https://github.com/JerrettDavis/headroom/commit/3e97a0cb73a5801496dc1ae2dab9e81788d87b2a))
* **proxy:** recover SSE usage from truncated final event ([b166bd3](https://github.com/JerrettDavis/headroom/commit/b166bd3d6a9b50925bd0fee5dedce704666ee35e))
* **proxy:** recover SSE usage from truncated final event ([d91254a](https://github.com/JerrettDavis/headroom/commit/d91254a0f2419dbd77e6b8f3b6718a029b725082))
* **proxy:** remove invalid savings tracker flock ([0d8de25](https://github.com/JerrettDavis/headroom/commit/0d8de25bfc346da572fecabf86580415e3939341))
* **proxy:** restore Codex usage headers on WS and streaming SSE transports ([#577](https://github.com/JerrettDavis/headroom/issues/577)) ([#794](https://github.com/JerrettDavis/headroom/issues/794)) ([0ce68de](https://github.com/JerrettDavis/headroom/commit/0ce68dedd770d5411d16abe30e5ea9dd0b7d8eee))
* **proxy:** route Claude Code model metadata to Anthropic ([#627](https://github.com/JerrettDavis/headroom/issues/627)) ([30c1ac8](https://github.com/JerrettDavis/headroom/commit/30c1ac8656bcc3d11755daef8d1d27cd8770ebc7))
* **proxy:** route Codex subscription /backend-api/* catchall to chatgpt.com ([4073dcd](https://github.com/JerrettDavis/headroom/commit/4073dcd23108b2ed1e37a9acbab29f34789e2bbc))
* **proxy:** route Codex subscription /backend-api/* catchall to chatgpt.com ([341dcf0](https://github.com/JerrettDavis/headroom/commit/341dcf03e9a28683bc762af436347b3798102d24))
* **proxy:** rustfmt drift in live_zone_anthropic imports (F2.1 c2 followup) ([0546795](https://github.com/JerrettDavis/headroom/commit/05467955479d9578934a956c29581997d12a1b16))
* **proxy:** Strands MCP bundle + backend path fixes + Codex fail-closed protection ([20dc1f2](https://github.com/JerrettDavis/headroom/commit/20dc1f28f3ccadbc2d1109d73b5bfe875eb81c47))
* **proxy:** surface CompressionDecision.passthrough_reason in tags ([88983ad](https://github.com/JerrettDavis/headroom/commit/88983ad34efef0556be9baeb1dd57397e4e6d724))
* **proxy:** surface CompressionDecision.passthrough_reason in tags ([e4e28b6](https://github.com/JerrettDavis/headroom/commit/e4e28b65f459df6879b38384842faaeba6357a2a))
* **proxy:** thread tags into 13 outcome sites + synth /v1/models + free-fn _extract_tags ([3ec5492](https://github.com/JerrettDavis/headroom/commit/3ec549288aed46dc46067b6dc55f287a8b19d218))
* **proxy:** thread tags into 13 outcome sites; synthesize /v1/models for ChatGPT auth ([6d62985](https://github.com/JerrettDavis/headroom/commit/6d62985b73b4a9e50d13ee9fc3df4b62bcba1c14))
* **proxy:** unblock Codex WS compression — delete inner-pool + global semaphore ([6cf8f7f](https://github.com/JerrettDavis/headroom/commit/6cf8f7f44abf82b5a34572ead0b6ee0bf523a6ce))
* **proxy:** unblock Codex WS compression — delete inner-pool + global semaphore ([a167f5c](https://github.com/JerrettDavis/headroom/commit/a167f5cc299ae78ce94cc6ddc3694af26c5c408b))
* **proxy:** wire CompressionPolicy through handlers + flip default enabled (F2.1 c5/5) ([0fef428](https://github.com/JerrettDavis/headroom/commit/0fef428f1fecff6a72a355098457bacc6465ef5f))
* **proxy:** wire PyO3 compression into /v1/responses handler (hot-fix c2/2) ([c0baaf0](https://github.com/JerrettDavis/headroom/commit/c0baaf052f18b2561226d20de6b182bb14300416))
* re-enable Codex /v1/responses compression + fix prose-format over-freeze ([7aaa4ac](https://github.com/JerrettDavis/headroom/commit/7aaa4ac48fcada1dc8e44477dc8256153d75093b))
* register serena mcp during wrap ([44231f6](https://github.com/JerrettDavis/headroom/commit/44231f68cd8bbbd010defdba2fb919bf49bafc89))
* register serena mcp during wrap ([a7160f7](https://github.com/JerrettDavis/headroom/commit/a7160f7eab87813f065ab398692231398c221b87))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([4a39ef5](https://github.com/JerrettDavis/headroom/commit/4a39ef54ed6cdaa24d8f9fa49bbd3daf7100658e))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([0f3e3af](https://github.com/JerrettDavis/headroom/commit/0f3e3af6b2a154c5ecaeda3f9770cec97e9a3ba0))
* remove env_key from injected Codex provider — fixes [#393](https://github.com/JerrettDavis/headroom/issues/393) ([7fb6d7f](https://github.com/JerrettDavis/headroom/commit/7fb6d7fefdce6145a39485b0c7d3227bdd573530))
* **rust:** A4 — honor cache_control markers; serde_json arbitrary_precision + raw_value ([3f99128](https://github.com/JerrettDavis/headroom/commit/3f991282364d715e98882749e8fa9a5b02455609))
* **rust:** IntelligentContextManager port (simplified, OSS) — PR-B ([21784f4](https://github.com/JerrettDavis/headroom/commit/21784f4288078518cf28e0d6731764809c082355))
* **rust:** PR-A1 — make /v1/messages compression a passthrough ([a974bb1](https://github.com/JerrettDavis/headroom/commit/a974bb153a25a4b32a75ff34a5f11b8b39f675a6))
* **rust:** wire ICM compressor into Rust proxy on /v1/messages ([4c52766](https://github.com/JerrettDavis/headroom/commit/4c52766860c6fa1b3c7d46c223050d8a0c7d7840))
* **rust:** wire ICM compressor into Rust proxy on /v1/messages ([fa5fbfa](https://github.com/JerrettDavis/headroom/commit/fa5fbfabf4df6bd2dfaebec2ffee6f17b39d4a1c))
* schema compaction must not drop property names that match DROP_KEYS ([#785](https://github.com/JerrettDavis/headroom/issues/785)) ([ae2122f](https://github.com/JerrettDavis/headroom/commit/ae2122fda8ff0efc03d609d27270453fea3a8718))
* **security:** allowlist GitGuardian-flagged test fixtures ([cf5a715](https://github.com/JerrettDavis/headroom/commit/cf5a71571c8566de7d315b362d9f1382ffc9d19e))
* **security:** block DNS-rebinding on /debug/* and /stats/reset via Host-header allowlist ([#605](https://github.com/JerrettDavis/headroom/issues/605)) ([b4b5025](https://github.com/JerrettDavis/headroom/commit/b4b50253f16d0a30f1d17a959753137e997efbac))
* **security:** patch loopback guard, retry None raise, async subprocess, and cache race ([06d7cb9](https://github.com/JerrettDavis/headroom/commit/06d7cb9e6c011711a478864a970f7c87ee853a97))
* **security:** patch loopback guard, retry None raise, blocking subprocess, and cache stats race ([78f3a4d](https://github.com/JerrettDavis/headroom/commit/78f3a4dd3e8e26525822a3c830d576d702dfed8b))
* ship glibc 2.38 compat shim + wheel symbol audit (closes [#355](https://github.com/JerrettDavis/headroom/issues/355)) ([6793adb](https://github.com/JerrettDavis/headroom/commit/6793adb003e1efb8ba6a6446a7829de2602a266f))
* ship glibc 2.38 compat shim + wheel symbol audit (closes [#355](https://github.com/JerrettDavis/headroom/issues/355)) ([e214672](https://github.com/JerrettDavis/headroom/commit/e2146724af09801c8f75dcd357d1a61c91d6bbb0))
* sort anthropic handler imports ([8f6a9cf](https://github.com/JerrettDavis/headroom/commit/8f6a9cfb26b480593a7e42e4ecd076fa74c4fc2a))
* speed up Kompress inference ([98f73e3](https://github.com/JerrettDavis/headroom/commit/98f73e3502c62dbc2c4119dcc68ef27537cc8f58))
* **ssl:** upstream httpx client inherits SSL_CERT_FILE, REQUESTS_CA_BUNDLE, NODE_EXTRA_CA_CERTS ([#745](https://github.com/JerrettDavis/headroom/issues/745)) ([e50fbb3](https://github.com/JerrettDavis/headroom/commit/e50fbb3e0d61d561456d7b0ff9e0a8ee106a2f02))
* stabilize codex compression, stats, and proxy lifecycle ([eaf5980](https://github.com/JerrettDavis/headroom/commit/eaf5980b4ac48c909a1fa2ef1ace460752ec0918))
* stabilize learn write-failure worker test ([155c069](https://github.com/JerrettDavis/headroom/commit/155c0691767d68f7ec5b47de476f8215946901bf))
* **startup:** move HF/httpx log suppression before sentence_transformers init ([#622](https://github.com/JerrettDavis/headroom/issues/622)) ([176d4c7](https://github.com/JerrettDavis/headroom/commit/176d4c772a7ca8c9da58ca2403f890ba85e8bad8))
* **startup:** suppress proxy startup log noise ([#619](https://github.com/JerrettDavis/headroom/issues/619)) ([4555901](https://github.com/JerrettDavis/headroom/commit/45559011b16a2e084dda22c675c819a4789f961d))
* **subscription:** address G2 review findings — phantom delta, multi-worker race, silent fallbacks ([f68090c](https://github.com/JerrettDavis/headroom/commit/f68090c5b4bd9670ee7fc9a0c71e57f05072c18c))
* **subscription:** wire tokens_saved_rtk data plane ([c7d1247](https://github.com/JerrettDavis/headroom/commit/c7d1247a2bd06738c3b6c8e73e15902a7e428467))
* **subscription:** wire tokens_saved_rtk from RTK stats endpoint ([44c605f](https://github.com/JerrettDavis/headroom/commit/44c605fbb0e3ae4e7a92d9693d0da8bc21115b81))
* suppress LiteLLM provider banner before import ([#874](https://github.com/JerrettDavis/headroom/issues/874)) ([f9384ef](https://github.com/JerrettDavis/headroom/commit/f9384ef4b780eaa1d8ca6dcc314ad430b87f524a))
* sync Codex MCP proxy config during wrap ([ac1d11c](https://github.com/JerrettDavis/headroom/commit/ac1d11c9a2a946ad5362db106cf9094262b7329a))
* **telemetry:** honour HEADROOM_TELEMETRY=off in /v1/telemetry collector ([218821e](https://github.com/JerrettDavis/headroom/commit/218821e4897dfcb78382b95330fc69cd08d9ccc1))
* **telemetry:** honour HEADROOM_TELEMETRY=off in /v1/telemetry collector ([#390](https://github.com/JerrettDavis/headroom/issues/390)) ([9ddbdf2](https://github.com/JerrettDavis/headroom/commit/9ddbdf2313eb4a77ad6cff3a0ae5eb0264cf7c57))
* **tests:** drive RTK subprocess failure with real exec, not monkeypatched run ([9b6d637](https://github.com/JerrettDavis/headroom/commit/9b6d6374f13a88842a1944688005649ad3680acd))
* **tests:** make codex scheduler stress test machine-independent ([f82d61b](https://github.com/JerrettDavis/headroom/commit/f82d61b1636cfa7d75aeca01f32df070cff3318d))
* **tests:** make stress test CI-robust — uniform frame sizes, tighter ratio ([c22342e](https://github.com/JerrettDavis/headroom/commit/c22342e1c7e909b3270d19fbaa5583b91a166952))
* **tests:** mock logger.warning directly instead of relying on caplog ([c38dac3](https://github.com/JerrettDavis/headroom/commit/c38dac301e6bc702979ab11357a9c27a180ae060))
* **tests:** patch headroom.rtk.get_rtk_path, not the helpers alias ([317dffe](https://github.com/JerrettDavis/headroom/commit/317dffe58fb0c6233210bbc9e42ebf16b9288391))
* **tests:** ship scripts/replay_codex_ws_load.py so CI can import it ([e532763](https://github.com/JerrettDavis/headroom/commit/e53276302ef9585be37519b16eb94513242372e7))
* **tests:** tomllib fallback to tomli on python 3.10 ([74843d1](https://github.com/JerrettDavis/headroom/commit/74843d1d626de70158a359661a540c615ef1a6c5))
* **test:** stub _run_compression_in_executor on _DummyOpenAIHandler ([456a6b3](https://github.com/JerrettDavis/headroom/commit/456a6b33af06f9dfdd07c58cb4a7820c6b890435))
* **tests:** update CacheAligner detector-only tests for F2.2 5-field CompressionPolicy ([be5d9f7](https://github.com/JerrettDavis/headroom/commit/be5d9f7baaf11d706afecf7c0ca0b9da984f38a2))
* **tests:** widen wrap-e2e openclaw startup timeout from 5s to 30s ([2ae88a6](https://github.com/JerrettDavis/headroom/commit/2ae88a6874b19ec9c4173b5a41e4f09918d2a348))
* **tests:** widen wrap-e2e openclaw startup timeout to 30s ([2febf1b](https://github.com/JerrettDavis/headroom/commit/2febf1bbbfd0e2d7fde438afff93b9e2e9c9d093))
* **transforms:** F2.2 c2/3 — wire toin_read_only gate + extend policy_selected log ([5b38cbf](https://github.com/JerrettDavis/headroom/commit/5b38cbf8a79d146adb6ca2ee25974bc35e271b05))
* **transforms:** Python parity port of CompressionPolicy + cache_aligner gate (F2.1 c4/5) ([de8e245](https://github.com/JerrettDavis/headroom/commit/de8e245990399442145a2aa5a8fe61eb4ddb3a76))
* **transforms:** use thread-local tree-sitter parsers to prevent pyo3 Unsendable panic ([#604](https://github.com/JerrettDavis/headroom/issues/604)) ([2ad300a](https://github.com/JerrettDavis/headroom/commit/2ad300aff801838efe5649b00a0396523a401a2a))
* update dashboard doc link ([#544](https://github.com/JerrettDavis/headroom/issues/544)) ([378d77e](https://github.com/JerrettDavis/headroom/commit/378d77e79d0020ca7fba3de8df7aaf910056ad2a))
* Update Next.js to 16.2.4 in docs/bun.lock to address GHSA-gx5p-jg67-6x7h (CVE-2026-44580) ([0b9f11a](https://github.com/JerrettDavis/headroom/commit/0b9f11a223bb6e6a6c1660ff1dfc1df6d67dfa84))
* Update Next.js to 16.2.6 in docs/package.json and package-lock.json to address GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([db5d15f](https://github.com/JerrettDavis/headroom/commit/db5d15f99e71b69a369eb9c161e04dbffb9b5d4a))
* Upgrade litellm to 1.86.2 to remediate CVE-2026-42271 ([07581b9](https://github.com/JerrettDavis/headroom/commit/07581b9e8075b833a6b543149008547260fe9dc0))
* use thread-local tree-sitter parsers to prevent unsendable panic ([38aefc1](https://github.com/JerrettDavis/headroom/commit/38aefc1d34e65beec46f0b8cb0ff43028de38cbb))
* **vertex,bedrock:** remove dead handle_raw_predict, honour X-Forwarded-Proto, cap Bedrock body size ([c6ecdc4](https://github.com/JerrettDavis/headroom/commit/c6ecdc429923872a9821e2e191da4990d41e8ad4))
* **vertex,bedrock:** remove dead handle_raw_predict, honour X-Forwarded-Proto, cap Bedrock body size ([28b2bac](https://github.com/JerrettDavis/headroom/commit/28b2bacdf6ffebb1c2bfa81ea80a3862f8ef0e24))
* Wave 3 — multi-turn live integration tests for A+B realignment ([dcbc921](https://github.com/JerrettDavis/headroom/commit/dcbc921d63e0d647457e04bd449fcae56fdd336b))
* Windows ORT builds and Docker signing retries ([5cc3fd4](https://github.com/JerrettDavis/headroom/commit/5cc3fd4852337b64e635b33d9a490e0931e46cef))
* **wrap:** report unbindable proxy ports ([#602](https://github.com/JerrettDavis/headroom/issues/602)) ([6dfcaa8](https://github.com/JerrettDavis/headroom/commit/6dfcaa839f1175518e378963c79cc7bd3ceb7946))
* **wrap:** track shared proxy clients with markers ([#877](https://github.com/JerrettDavis/headroom/issues/877)) ([05bd56b](https://github.com/JerrettDavis/headroom/commit/05bd56bcb6b103fab5522da2b14295cf7bd8dbc1))


### Code Refactoring

* **cli:** factor shared wrap-subcommand scaffolding ([8eeb926](https://github.com/JerrettDavis/headroom/commit/8eeb9261680dd071654a87204521ccd3703ef77d))
* **cli:** factor shared wrap-subcommand scaffolding ([c74ad11](https://github.com/JerrettDavis/headroom/commit/c74ad113a4ced9968e45cad1077e6a020dc6a401))
* extract litellm model resolution to shared utility ([ec7d006](https://github.com/JerrettDavis/headroom/commit/ec7d0065cc5055e504e79cf24f3951e404fe4cb9))
* extract litellm model resolution to shared utility ([896a093](https://github.com/JerrettDavis/headroom/commit/896a0933998c6dda4fcae44cc194ab2770e6f9d6))
* **proxy:** collapse 3 stream finalizers onto RequestOutcome.from_stream ([af50390](https://github.com/JerrettDavis/headroom/commit/af503905b584e61eac37d14e290b3033074ffc24))
* **proxy:** collapse 3 stream finalizers onto RequestOutcome.from_stream ([694589f](https://github.com/JerrettDavis/headroom/commit/694589fec4997ae60118aa3609672a9ed3262a8f))
* **proxy:** CompressionDecision contract + 4 missing-gate bug fixes ([1699435](https://github.com/JerrettDavis/headroom/commit/1699435ddfd10f3918fcd18e1fdcfc6b232ebdad))
* **proxy:** introduce RequestOutcome funnel; collapse 3 streaming finalizers ([3762f63](https://github.com/JerrettDavis/headroom/commit/3762f6375ca7a93301f6c3b946f6b1cefb9ca87d))
* **proxy:** introduce RequestOutcome funnel; collapse 3 streaming finalizers ([e898f68](https://github.com/JerrettDavis/headroom/commit/e898f68b89fcf3dd6cd5d7ee199e9f032d4fb20a))
* **proxy:** MemoryRanker + ImageCompressionDecision + branch-aware version-sync ([c79aa22](https://github.com/JerrettDavis/headroom/commit/c79aa22be798a40e19c24cfc3e33427cdd84dc29))
* **proxy:** MemoryRanker + ImageCompressionDecision + branch-aware version-sync ([a7b197c](https://github.com/JerrettDavis/headroom/commit/a7b197c6eca08e0080cb04bc19859026233c5603))
* **proxy:** migrate Codex WS + OpenAI HTTP + batch handlers; delete Databricks ([993c907](https://github.com/JerrettDavis/headroom/commit/993c9076f92558402c5f925a685fde4001f96daf))
* **proxy:** migrate Gemini + Anthropic non-streaming onto outcome funnel ([7aca004](https://github.com/JerrettDavis/headroom/commit/7aca00445e23f421d1c8c278bf720cab046da20f))
* single-wheel maturin build backend (fixes [#355](https://github.com/JerrettDavis/headroom/issues/355)) ([aa80ec2](https://github.com/JerrettDavis/headroom/commit/aa80ec2cc46684e27c74f8d4f306b12059f30388))
* single-wheel maturin build backend (fixes [#355](https://github.com/JerrettDavis/headroom/issues/355)) ([2a91cbb](https://github.com/JerrettDavis/headroom/commit/2a91cbb4b41bc5805b34c0445dda6eebaba28186))

## [0.25.0](https://github.com/chopratejas/headroom/compare/v0.24.0...v0.25.0) (2026-06-12)


### Features

* add differential network capture harness ([#761](https://github.com/chopratejas/headroom/issues/761)) ([11ab5f8](https://github.com/chopratejas/headroom/commit/11ab5f83a1ccd617a2608349a42feff7f7e72b98))
* add light mode for dashboard ([#834](https://github.com/chopratejas/headroom/issues/834)) ([c425893](https://github.com/chopratejas/headroom/commit/c425893d123e67c62ee20ff64ae350eb4ea56477))
* add OAuth2 client-credentials upstream-auth proxy extension ([#778](https://github.com/chopratejas/headroom/issues/778)) ([#784](https://github.com/chopratejas/headroom/issues/784)) ([eb2e50f](https://github.com/chopratejas/headroom/commit/eb2e50feb26bacadf8812d6e608a458a990096b9))
* add Vertex AI proxy routing ([#793](https://github.com/chopratejas/headroom/issues/793)) ([3c77e52](https://github.com/chopratejas/headroom/commit/3c77e52ce431210e6045671cf5f7c66c79f90a32))
* **cli:** comprehensive help text, validation, and exception handling improvements ([#640](https://github.com/chopratejas/headroom/issues/640)) ([028efab](https://github.com/chopratejas/headroom/commit/028efabb4e611d77118baefb8ffdd13b0edc4fc5))
* compression safety rails — error-output protection, pipeline circuit breaker, library inflation guard ([#851](https://github.com/chopratejas/headroom/issues/851)) ([c0cadcc](https://github.com/chopratejas/headroom/commit/c0cadccff98e572f126185f371e4de9e241b12e0))
* **dashboard:** per-model savings breakdown and expected-vs-actual cost on historical charts ([#807](https://github.com/chopratejas/headroom/issues/807)) ([34dafe6](https://github.com/chopratejas/headroom/commit/34dafe69d907c9a2971abc0d801ff9bfa498b3a8))
* detect re-served tool results as over-compression waste signal ([#854](https://github.com/chopratejas/headroom/issues/854)) ([5f1d88a](https://github.com/chopratejas/headroom/commit/5f1d88ad2701ed186df93d8e2a3980f0329d9dbb))
* **evals:** add zero-cost tool schema compaction integrity eval ([#817](https://github.com/chopratejas/headroom/issues/817)) ([53a08c6](https://github.com/chopratejas/headroom/commit/53a08c63bf56a76d4fb7b649e37c8e62b0b4cebf))
* gated Markdown-KV compaction formatter (serialization-aware output) ([#859](https://github.com/chopratejas/headroom/issues/859)) ([06b2625](https://github.com/chopratejas/headroom/commit/06b2625b17b0b032f688d321c6aa30ae3f2b7d96))
* **kompress:** warn on unrecognized HEADROOM_KOMPRESS_BACKEND + document backend selection ([#204](https://github.com/chopratejas/headroom/issues/204)) ([6367d0b](https://github.com/chopratejas/headroom/commit/6367d0b7228f53b29bbd20f55c1729476ba5ea68))
* **memory:** add opt-in Apple-GPU (MPS) embedding runtime ([#766](https://github.com/chopratejas/headroom/issues/766)) ([c71592d](https://github.com/chopratejas/headroom/commit/c71592d4214adf1022e4c608518ae0c3ac4aa5e9))
* net-cost cache mutation formula on CompressionPolicy ([#856](https://github.com/chopratejas/headroom/issues/856) P1) ([#857](https://github.com/chopratejas/headroom/issues/857)) ([d5f5802](https://github.com/chopratejas/headroom/commit/d5f58026e2a882bc508acfbddfc9d472100d6e16))
* **plugins:** Hermes agent headroom_retrieve plugin ([#824](https://github.com/chopratejas/headroom/issues/824)) ([058bced](https://github.com/chopratejas/headroom/commit/058bcedab838f3b34ac8e38853e1924329efd820))
* probe-based retention scoring of recorded compression events ([#862](https://github.com/chopratejas/headroom/issues/862)) ([c2106cb](https://github.com/chopratejas/headroom/commit/c2106cbdabb905e1980c6694000c220a5042171c))
* **proxy:** add CLI opt-outs for CCR injection (compression-only mode) ([#823](https://github.com/chopratejas/headroom/issues/823)) ([693d9d2](https://github.com/chopratejas/headroom/commit/693d9d20e2b2d9bfce3a0c48314850ee77ff8af3))
* **proxy:** attribute savings history rollups per provider ([#791](https://github.com/chopratejas/headroom/issues/791)) ([0b8b8d9](https://github.com/chopratejas/headroom/commit/0b8b8d92de3bd5e0301eadedacfb4b1d20a8de7f))
* **proxy:** log compressed messages alongside original request ([#261](https://github.com/chopratejas/headroom/issues/261)) ([2269e40](https://github.com/chopratejas/headroom/commit/2269e40bde7e1b9fb0620bd2cec9e33a92834080))
* **proxy:** per-project savings breakdown on the dashboard (claude, codex, aider, copilot, cursor) ([#803](https://github.com/chopratejas/headroom/issues/803)) ([914a60a](https://github.com/chopratejas/headroom/commit/914a60a2b07caad8488c1e19a5465726b95f83d3))
* support Python 3.14+ via pyo3 abi3 stable ABI ([#516](https://github.com/chopratejas/headroom/issues/516)) ([19eac8e](https://github.com/chopratejas/headroom/commit/19eac8e00dc9e3911f3afe8e8e5dcc9e00346baa))
* switch Kompress default to kompress-v2-base with weight-only int8 ONNX ([#799](https://github.com/chopratejas/headroom/issues/799)) ([74392b2](https://github.com/chopratejas/headroom/commit/74392b238e4f76fa061e673d1415fc7fa2830011))
* **transforms:** attribute read_lifecycle + smart_crush tags ([#249](https://github.com/chopratejas/headroom/issues/249)) ([8f37426](https://github.com/chopratejas/headroom/commit/8f374263d3971c072b5c977375c873864fb05763))


### Bug Fixes

* **anthropic:** CCR exception must re-raise, not silently swallow ([#838](https://github.com/chopratejas/headroom/issues/838)) ([8db5efc](https://github.com/chopratejas/headroom/commit/8db5efc6f9f6de59e9d55cbcd63b75c37a81a26e))
* **ccr:** key Rust search/diff/log markers with explicit_hash ([#852](https://github.com/chopratejas/headroom/issues/852)) ([bfcb07d](https://github.com/chopratejas/headroom/commit/bfcb07d78ea7eba539a65b11e100ec23b336d8d1))
* **ccr:** make retrieval TTL configurable ([#715](https://github.com/chopratejas/headroom/issues/715)) ([2533f77](https://github.com/chopratejas/headroom/commit/2533f7703ee261dc35767b11e46b8eab6e0c454d))
* **ccr:** skip CCR when model calls headroom_retrieve alongside user tools ([#839](https://github.com/chopratejas/headroom/issues/839)) ([30078f8](https://github.com/chopratejas/headroom/commit/30078f8465fb6bb78a5a9c394b75e60cd3c4eeec))
* **ccr:** use shared compression store ([#875](https://github.com/chopratejas/headroom/issues/875)) ([249af6c](https://github.com/chopratejas/headroom/commit/249af6cc7b379678e60da3e98e552368632fd4f4))
* **ci:** correct comments, timeouts, and pip reliability in native e2e workflows ([#878](https://github.com/chopratejas/headroom/issues/878)) ([b716c8c](https://github.com/chopratejas/headroom/commit/b716c8c2ee7ccc68dd1b9294760db1af866843f2))
* **ci:** pin cosign-installer to v3 (v4 does not exist) ([#774](https://github.com/chopratejas/headroom/issues/774)) ([199d693](https://github.com/chopratejas/headroom/commit/199d693f98ecd72d80181c8fee8422b6b64651a2))
* **codex:** respect CODEX_HOME for wrap config ([#731](https://github.com/chopratejas/headroom/issues/731)) ([96abf38](https://github.com/chopratejas/headroom/commit/96abf38b0972adf5e5c66f9a49aa9d9f951b1aa0))
* **content_router:** guard against empty compression output causing Anthropic 400 ([#771](https://github.com/chopratejas/headroom/issues/771)) ([2f9ff07](https://github.com/chopratejas/headroom/commit/2f9ff07e6caef0fe32d00ece6266a476eecff5a3))
* **copilot:** use responses API for subscription reasoning models ([#647](https://github.com/chopratejas/headroom/issues/647)) ([84ac332](https://github.com/chopratejas/headroom/commit/84ac332d14dafacedc2f0b46f5ac6b3977b098d0))
* correct preserved-entry index mapping in Gemini content round-trip ([#836](https://github.com/chopratejas/headroom/issues/836)) ([0ffe2b6](https://github.com/chopratejas/headroom/commit/0ffe2b6ea49e5c8d3bff5fe2c90873c71a95c457))
* **dashboard:** stable 'Proxy $ Saved' hero tile under --workers &gt; 1 ([#481](https://github.com/chopratejas/headroom/issues/481)) ([fd73b88](https://github.com/chopratejas/headroom/commit/fd73b88368b22beeb586b8e1aa37fcd2afb12532))
* don't inject empty tools:[] when client omitted the tools field ([#772](https://github.com/chopratejas/headroom/issues/772)) ([574bbae](https://github.com/chopratejas/headroom/commit/574bbae2cbe2f20b3f0e12b421c25ac256712f0a))
* harden Copilot API auth token handling ([#557](https://github.com/chopratejas/headroom/issues/557)) ([6b0c09f](https://github.com/chopratejas/headroom/commit/6b0c09ffd5f2ce18c4d2cfa6233feaf37d487ead))
* **health:** readyz verifies upstream connectivity, not just process liveness ([#744](https://github.com/chopratejas/headroom/issues/744)) ([5dfb446](https://github.com/chopratejas/headroom/commit/5dfb446da1fb65002e0dea18a90210a2a026f0b3))
* **init:** guard persistent task startup ([#616](https://github.com/chopratejas/headroom/issues/616)) ([9252d85](https://github.com/chopratejas/headroom/commit/9252d852c5a4c716eb5438b8f438d50e59a55fef))
* **init:** normalize Windows hook paths to forward slashes ([#788](https://github.com/chopratejas/headroom/issues/788)) ([6ea6e31](https://github.com/chopratejas/headroom/commit/6ea6e31f09845b2ad5c8bae73bcf353f3b629188))
* **init:** suppress hook recovery output ([#760](https://github.com/chopratejas/headroom/issues/760)) ([b439599](https://github.com/chopratejas/headroom/commit/b4395993aecbb65b85a5b2479dfdb35ea243bf54))
* **learn:** claude-cli streams output with idle timeout ([#373](https://github.com/chopratejas/headroom/issues/373)) ([9bff575](https://github.com/chopratejas/headroom/commit/9bff5752bbd769902f249cdfde42bc53539afd02))
* make headroom wrap readiness probe timeout configurable for slow ML imports ([#581](https://github.com/chopratejas/headroom/issues/581)) ([163677b](https://github.com/chopratejas/headroom/commit/163677b405d7ca8a54d6d7c798bf6ead90da7880))
* **parser:** detect waste signals in Anthropic tool_result content blocks ([#815](https://github.com/chopratejas/headroom/issues/815)) ([929698a](https://github.com/chopratejas/headroom/commit/929698af1030e5926f3766d7d6ac292d6e38437b))
* **proxy:** F4 — trust X-Forwarded-* only behind allow-listed gateway ([d10bd5f](https://github.com/chopratejas/headroom/commit/d10bd5f59c5a36e14f6c5f0480b821532521b753))
* **proxy:** lazy-import server to avoid fastapi crash ([#442](https://github.com/chopratejas/headroom/issues/442)) ([93c6937](https://github.com/chopratejas/headroom/commit/93c69372e614f2b04873bed75602a88d2256a7fc))
* **proxy:** make CCR multi-worker warning conditional on backend ([#770](https://github.com/chopratejas/headroom/issues/770)) ([d76a729](https://github.com/chopratejas/headroom/commit/d76a7296df121365d74c415b8c702a3ad80abd30))
* **proxy:** make Kompress eager preload cache-only so a cold cache can't block startup ([#783](https://github.com/chopratejas/headroom/issues/783)) ([841663d](https://github.com/chopratejas/headroom/commit/841663da16971b1e0d8e204fdf18e4bafedaf9e0))
* **proxy:** restore Codex usage headers on WS and streaming SSE transports ([#577](https://github.com/chopratejas/headroom/issues/577)) ([#794](https://github.com/chopratejas/headroom/issues/794)) ([0ce68de](https://github.com/chopratejas/headroom/commit/0ce68dedd770d5411d16abe30e5ea9dd0b7d8eee))
* schema compaction must not drop property names that match DROP_KEYS ([#785](https://github.com/chopratejas/headroom/issues/785)) ([ae2122f](https://github.com/chopratejas/headroom/commit/ae2122fda8ff0efc03d609d27270453fea3a8718))
* **security:** block DNS-rebinding on /debug/* and /stats/reset via Host-header allowlist ([#605](https://github.com/chopratejas/headroom/issues/605)) ([b4b5025](https://github.com/chopratejas/headroom/commit/b4b50253f16d0a30f1d17a959753137e997efbac))
* **ssl:** upstream httpx client inherits SSL_CERT_FILE, REQUESTS_CA_BUNDLE, NODE_EXTRA_CA_CERTS ([#745](https://github.com/chopratejas/headroom/issues/745)) ([e50fbb3](https://github.com/chopratejas/headroom/commit/e50fbb3e0d61d561456d7b0ff9e0a8ee106a2f02))
* suppress LiteLLM provider banner before import ([#874](https://github.com/chopratejas/headroom/issues/874)) ([f9384ef](https://github.com/chopratejas/headroom/commit/f9384ef4b780eaa1d8ca6dcc314ad430b87f524a))
* **transforms:** use thread-local tree-sitter parsers to prevent pyo3 Unsendable panic ([#604](https://github.com/chopratejas/headroom/issues/604)) ([2ad300a](https://github.com/chopratejas/headroom/commit/2ad300aff801838efe5649b00a0396523a401a2a))
* **wrap:** track shared proxy clients with markers ([#877](https://github.com/chopratejas/headroom/issues/877)) ([05bd56b](https://github.com/chopratejas/headroom/commit/05bd56bcb6b103fab5522da2b14295cf7bd8dbc1))


### Code Refactoring

* extract litellm model resolution to shared utility ([ec7d006](https://github.com/chopratejas/headroom/commit/ec7d0065cc5055e504e79cf24f3951e404fe4cb9))

## [0.24.0](https://github.com/chopratejas/headroom/compare/v0.23.0...v0.24.0) (2026-06-08)


### Features

* **perf:** add --format {text,json,csv} to `headroom perf` ([#648](https://github.com/chopratejas/headroom/issues/648)) ([9fe4886](https://github.com/chopratejas/headroom/commit/9fe4886cf6b612452f7271d3204872f804074c1f))
* **proxy:** show resolved upstream API targets in startup banner ([#586](https://github.com/chopratejas/headroom/issues/586)) ([8dbe7ad](https://github.com/chopratejas/headroom/commit/8dbe7ad41b3a1d33c01874be5c1cbc68a5e68111)), closes [#583](https://github.com/chopratejas/headroom/issues/583)
* **relevance:** weight BM25 score_batch by corpus IDF ([#646](https://github.com/chopratejas/headroom/issues/646)) ([88177bd](https://github.com/chopratejas/headroom/commit/88177bd7a680490ac85d244c5fff90f21a3be27c))
* support CLAUDE_CODE_USE_FOUNDRY and custom upstream gateways ([#726](https://github.com/chopratejas/headroom/issues/726)) ([d90cdce](https://github.com/chopratejas/headroom/commit/d90cdce3b69bbf27e0f5feea461766a9d797cf7e))


### Bug Fixes

* **ci:** restore green lint gate on main ([fe50f9d](https://github.com/chopratejas/headroom/commit/fe50f9daed35151134f79b767733d4be8093e325))
* **codex:** auto-enable fail-open on compression timeout in headroom wrap codex ([#531](https://github.com/chopratejas/headroom/issues/531)) ([5f5f261](https://github.com/chopratejas/headroom/commit/5f5f261a035d12d069eb212eb75c472e2c9edeff))
* **copilot:** restore generic endpoint for non-subscription OAuth ([#610](https://github.com/chopratejas/headroom/issues/610)) ([#612](https://github.com/chopratejas/headroom/issues/612)) ([18925b8](https://github.com/chopratejas/headroom/commit/18925b8c6e343c9d593891cd29ac27fee1cb9836))
* **deps:** move gunicorn to [proxy-prod] extra, add Windows guard ([#537](https://github.com/chopratejas/headroom/issues/537)) ([fa558c5](https://github.com/chopratejas/headroom/commit/fa558c5647a91562f4a8fba0271d27b02c8ae01f))
* **proxy:** fail-open on corrupt golden bytes instead of RuntimeError ([#603](https://github.com/chopratejas/headroom/issues/603)) ([2170a1b](https://github.com/chopratejas/headroom/commit/2170a1b4a00e9c46e845993c9b0f6cb2ef0c0684))
* **proxy:** route Claude Code model metadata to Anthropic ([#627](https://github.com/chopratejas/headroom/issues/627)) ([30c1ac8](https://github.com/chopratejas/headroom/commit/30c1ac8656bcc3d11755daef8d1d27cd8770ebc7))
* **security:** patch loopback guard, retry None raise, async subprocess, and cache race ([06d7cb9](https://github.com/chopratejas/headroom/commit/06d7cb9e6c011711a478864a970f7c87ee853a97))
* **security:** patch loopback guard, retry None raise, blocking subprocess, and cache stats race ([78f3a4d](https://github.com/chopratejas/headroom/commit/78f3a4dd3e8e26525822a3c830d576d702dfed8b))
* **startup:** move HF/httpx log suppression before sentence_transformers init ([#622](https://github.com/chopratejas/headroom/issues/622)) ([176d4c7](https://github.com/chopratejas/headroom/commit/176d4c772a7ca8c9da58ca2403f890ba85e8bad8))
* **startup:** suppress proxy startup log noise ([#619](https://github.com/chopratejas/headroom/issues/619)) ([4555901](https://github.com/chopratejas/headroom/commit/45559011b16a2e084dda22c675c819a4789f961d))
* **wrap:** report unbindable proxy ports ([#602](https://github.com/chopratejas/headroom/issues/602)) ([6dfcaa8](https://github.com/chopratejas/headroom/commit/6dfcaa839f1175518e378963c79cc7bd3ceb7946))

## [Unreleased]

### Added

* **kompress:** warn when `HEADROOM_KOMPRESS_BACKEND` is set to an unrecognized
  value instead of silently falling back to `auto`, and document the backend
  selection env var (`auto` / `onnx` / `onnx_cpu` / `onnx_coreml` / `pytorch` /
  `pytorch_mps` plus shorthand aliases) in `wiki/configuration.md` (issue
  [#202](https://github.com/chopratejas/headroom/issues/202), PR
  [#204](https://github.com/chopratejas/headroom/pull/204)).
* **proxy:** per-provider attribution in the savings history rollups. Each `/stats-history` bucket (hourly/daily/weekly/monthly) now carries a `by_provider` map breaking down `tokens_saved`, `compression_savings_usd_delta`, `total_input_tokens_delta`, and `total_input_cost_usd_delta` per provider, so consumers can show how savings and spend are distributed across providers within a time period. Providers only appear in a bucket where they moved a counter; legacy history checkpoints with no provider collapse into `"unknown"`. Affected files: `headroom/proxy/savings_tracker.py`, `headroom/proxy/prometheus_metrics.py`.
* **cli:** startup banner now includes a `Performance Tuning` section that surfaces active `HEADROOM_COMPRESSION_STABLE_AFTER_TURN`, `HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS`, and embedding-server socket values when set; shows a hint to set them when all defaults are in use.

### Changed

* **deps:** loosen over-pinned constraints and add upper bounds
  - `litellm==1.82.3` -> `>=1.86.2,<2.0` (exact pin blocked security patches; floor stays above the CVE-2026-42271 fix)
  - `transformers>=4.30.0` -> `>=4.30.0,<6.0` (add upper bound; library already crossed a major version silently)
  - `sentence-transformers>=2.2.0` -> `>=2.2.0,<6.0` (same; applied in `memory`, `evals`, and `dev` extras)
  - `neo4j>=5.20.0` -> `>=5.20.0,<7.0` (client had already crossed the 5.x/6.x boundary)
  - `mem0ai>=0.1.100` -> `>=1.0.0,<2.0` (floor was pre-1.0; locked package is already 1.0.11)
  - `langchain-core>=0.2.0` -> `>=1.3.3,<4.0` (floor stays above current high-severity advisory fixes)
  - `langchain-openai>=0.1.0` -> `>=1.1.14,<2.0` (floor stays above current advisory fixes)
  - `qdrant-client>=1.9.0` -> `>=1.9.0,<2.0`
  - `uvicorn>=0.23.0` -> `>=0.23.0,<1.0` (applied in `proxy` and `dev` extras)
  - Same `transformers` and `litellm` bounds applied consistently across `ml`, `voice`, and `dev` extras
* **docker:** bump `neo4j` image in `docker-compose.yml` from `5.15.0` to `5.26` (latest 5.x LTS)
* **docker:** bump `UV_VERSION` in `Dockerfile` from `0.11.16` to `0.11.18`

### Bug Fixes

* **codex:** respect `CODEX_HOME` when `headroom wrap codex` writes provider, MCP, memory, backup, and global `AGENTS.md` config, and warn when `unwrap codex` may be looking at the default Codex home because `CODEX_HOME` is unset.
* **proxy:** multi-worker CCR warning is now conditional on backend — when `HEADROOM_CCR_BACKEND` is unset (default `InMemoryBackend`, per-process), the startup warning includes CCR retrieval failures and suggests `HEADROOM_CCR_BACKEND=sqlite`; when a cross-worker backend is already configured, the warning covers only the remaining per-worker stores (compression cache, prefix tracker, TOIN, CostTracker). Updated `RUST_DEV.md` to accurately document Python `CompressionStore` as per-process by default.
* **deps:** move `gunicorn` to `[proxy-prod]` extra with `sys_platform != 'win32'` guard; removed from `[proxy]` to avoid forcing a Unix-only package on dev, CI, and Windows users ([#537](https://github.com/chopratejas/headroom/pull/537))
* **startup:** suppress proxy startup log noise -- litellm banner, trafilatura parse errors, HuggingFace Hub unauthenticated warnings, tiktoken fallback warning, and httpx INFO lines from sentence_transformers HEAD checks. Affected files: `headroom/providers/litellm.py`, `headroom/transforms/html_extractor.py`, `headroom/memory/adapters/embedders.py`, `headroom/providers/anthropic.py`, `headroom/providers/registry.py`, `headroom/image/onnx_router.py`, `headroom/transforms/kompress_compressor.py`.

## [0.23.0](https://github.com/chopratejas/headroom/compare/v0.22.4...v0.23.0) (2026-06-04)

### Features

* **copilot:** GitHub Copilot subscription mode through Headroom ([f4dff9b](https://github.com/chopratejas/headroom/commit/f4dff9b4885b5c62d79396bbb0847ae3e39a9bd9))


### Bug Fixes

* **ccr:** scope proactive expansion by workspace (cross-project leak) ([197601b](https://github.com/chopratejas/headroom/commit/197601bc64ee72e786bf6b94cd90efcac4269bcf))
* **ccr:** scope proactive expansion by workspace (cross-project leak) ([1bc163f](https://github.com/chopratejas/headroom/commit/1bc163f5bc1a8422f9ad659061e1fdd8cfeb077b))
* **codex:** keep init model_provider at config root ([#260](https://github.com/chopratejas/headroom/issues/260)) ([304dcc7](https://github.com/chopratejas/headroom/commit/304dcc78047bc744fc2f7656b484ec54dc271354))
* **codex:** keep init model_provider at config root ([#260](https://github.com/chopratejas/headroom/issues/260)) ([849b46d](https://github.com/chopratejas/headroom/commit/849b46de5934a88369af2fd7f7d52e9af0536a7e))
* **copilot:** deterministic subscription token handoff to the proxy ([72da461](https://github.com/chopratejas/headroom/commit/72da46121726074515e0c1eb9745498457a1a8d5))
* **copilot:** support subscription auth through Headroom ([ff4a0c6](https://github.com/chopratejas/headroom/commit/ff4a0c6bc64e5e68ab76c38047a36a3c7a6aaacf))
* correct tiktoken encoding for unknown gpt-4 model snapshots ([#552](https://github.com/chopratejas/headroom/issues/552)) ([0e551de](https://github.com/chopratejas/headroom/commit/0e551de9d81021bb7f0dde1857a2341408606969))
* decode/encode owned config, state and template assets as UTF-8 ([2f1538a](https://github.com/chopratejas/headroom/commit/2f1538a641dd0e60a7be3de85646a70c4bf7e287))
* decode/encode owned config, state and template assets as UTF-8 (fixes [#533](https://github.com/chopratejas/headroom/issues/533)) ([92075b9](https://github.com/chopratejas/headroom/commit/92075b95af799951c90a305a08ec4e958473967a))
* **docker:** upgrade base images to Python 3.13 / debian13 ([e6bf7a0](https://github.com/chopratejas/headroom/commit/e6bf7a03fef8a9f2e4802d63afdafb40627c7ad9))
* **docker:** upgrade base images to Python 3.13 / debian13, drop digest pinning ([08a2197](https://github.com/chopratejas/headroom/commit/08a219708c97dcdc678483a0e6891306624a1fad))
* **docs:** bump next.js to 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([a6a09e6](https://github.com/chopratejas/headroom/commit/a6a09e6cfbe6962a70a6fb2e4bebeee80756e304))
* **docs:** mkdocs configuration to build with correct folder ([#543](https://github.com/chopratejas/headroom/issues/543)) ([5557944](https://github.com/chopratejas/headroom/commit/55579445f84c363219f45dc5358599a04d4263ed))
* **docs:** update brace-expansion to 5.0.6 to remediate GHSA-jxxr-4gwj-5jf2 (CVE-2026-45149) ([6eb6fb5](https://github.com/chopratejas/headroom/commit/6eb6fb5941adfbd056daa1689c3fa0c3755fd298))
* **docs:** update bun.lock to next 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([91e0937](https://github.com/chopratejas/headroom/commit/91e0937243c801fa5f1021b4c47debef2444650c))
* ignore brackets inside JSON strings when splitting mixed content ([#553](https://github.com/chopratejas/headroom/issues/553)) ([bdcfc32](https://github.com/chopratejas/headroom/commit/bdcfc322da0c4cde69931d641cfa18c76ddb138b))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([211daae](https://github.com/chopratejas/headroom/commit/211daae25687901d1f893714d877b25606d0ef69))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([491a8b3](https://github.com/chopratejas/headroom/commit/491a8b3a1b260f42f503b3553a04c578c18e1cc0))
* **learn:** finish gemini-flash-latest default model sweep ([982d01b](https://github.com/chopratejas/headroom/commit/982d01b9c996fd5fe26154dc2f94d567192f6ff6))
* **learn:** finish gemini-flash-latest default model sweep ([#532](https://github.com/chopratejas/headroom/issues/532)) ([d797366](https://github.com/chopratejas/headroom/commit/d7973665f4e2f40f2b3acadd0ec584609fb33c6c))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([a178249](https://github.com/chopratejas/headroom/commit/a178249fc0af4a1b6f212decb4f6d2793d57fae8))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([482f80e](https://github.com/chopratejas/headroom/commit/482f80e735f124ee6860f6854255c77170b862e7))
* update dashboard doc link ([#544](https://github.com/chopratejas/headroom/issues/544)) ([378d77e](https://github.com/chopratejas/headroom/commit/378d77e79d0020ca7fba3de8df7aaf910056ad2a))
* Update Next.js to 16.2.4 in docs/bun.lock to address GHSA-gx5p-jg67-6x7h (CVE-2026-44580) ([0b9f11a](https://github.com/chopratejas/headroom/commit/0b9f11a223bb6e6a6c1660ff1dfc1df6d67dfa84))
* Update Next.js to 16.2.6 in docs/package.json and package-lock.json to address GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([db5d15f](https://github.com/chopratejas/headroom/commit/db5d15f99e71b69a369eb9c161e04dbffb9b5d4a))
* Upgrade litellm to 1.86.2 to remediate CVE-2026-42271 ([07581b9](https://github.com/chopratejas/headroom/commit/07581b9e8075b833a6b543149008547260fe9dc0))


### Code Refactoring

* **cli:** factor shared wrap-subcommand scaffolding ([8eeb926](https://github.com/chopratejas/headroom/commit/8eeb9261680dd071654a87204521ccd3703ef77d))
* **cli:** factor shared wrap-subcommand scaffolding ([c74ad11](https://github.com/chopratejas/headroom/commit/c74ad113a4ced9968e45cad1077e6a020dc6a401))

## [0.22.4](https://github.com/chopratejas/headroom/compare/v0.22.3...v0.22.4) (2026-05-26)


### Bug Fixes

* **cli:** G1 remediation — non-string clobber, per-model systemMessage, openhands gate ([ea1976e](https://github.com/chopratejas/headroom/commit/ea1976e37a5147ecf37dbf5ffe4af5c2f2d1be6a))
* **cli:** wrap CLI breadth — cline, continue, goose, openhands ([8625f80](https://github.com/chopratejas/headroom/commit/8625f8075ed75d2a002f6ba357697de0fa1ec434))
* **cli:** wrap subcommands for cline, continue, goose, openhands ([c375fa1](https://github.com/chopratejas/headroom/commit/c375fa156dd0434256805f274c07be4f45db9814))
* **observability:** G3 remediation — bound cardinality + wire dead metrics ([2a717a9](https://github.com/chopratejas/headroom/commit/2a717a993ee99f9401f5cdf78a23dcecd7cb1a51))
* **observability:** RTK metrics + Rust observability (Phase H blocker) ([b36ad9f](https://github.com/chopratejas/headroom/commit/b36ad9fe1c6a488eb9ffbf0e8b38d989278cf8ef))
* **observability:** wire Phase G PR-G3 RTK + proxy metrics (H-blocker) ([5f264a5](https://github.com/chopratejas/headroom/commit/5f264a53292e292c9c56b837c2750d1a415b1ea9))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([4a39ef5](https://github.com/chopratejas/headroom/commit/4a39ef54ed6cdaa24d8f9fa49bbd3daf7100658e))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([0f3e3af](https://github.com/chopratejas/headroom/commit/0f3e3af6b2a154c5ecaeda3f9770cec97e9a3ba0))
* **subscription:** address G2 review findings — phantom delta, multi-worker race, silent fallbacks ([f68090c](https://github.com/chopratejas/headroom/commit/f68090c5b4bd9670ee7fc9a0c71e57f05072c18c))
* **subscription:** wire tokens_saved_rtk data plane ([c7d1247](https://github.com/chopratejas/headroom/commit/c7d1247a2bd06738c3b6c8e73e15902a7e428467))
* **subscription:** wire tokens_saved_rtk from RTK stats endpoint ([44c605f](https://github.com/chopratejas/headroom/commit/44c605fbb0e3ae4e7a92d9693d0da8bc21115b81))
* **tests:** drive RTK subprocess failure with real exec, not monkeypatched run ([9b6d637](https://github.com/chopratejas/headroom/commit/9b6d6374f13a88842a1944688005649ad3680acd))
* **tests:** mock logger.warning directly instead of relying on caplog ([c38dac3](https://github.com/chopratejas/headroom/commit/c38dac301e6bc702979ab11357a9c27a180ae060))
* **tests:** patch headroom.rtk.get_rtk_path, not the helpers alias ([317dffe](https://github.com/chopratejas/headroom/commit/317dffe58fb0c6233210bbc9e42ebf16b9288391))
* **tests:** tomllib fallback to tomli on python 3.10 ([74843d1](https://github.com/chopratejas/headroom/commit/74843d1d626de70158a359661a540c615ef1a6c5))

## [Unreleased]

### Security
- **`/debug/memory` loopback guard.** The endpoint was missing the
  `Depends(_require_loopback)` guard that all other `/debug/*` endpoints carry.
  External callers can no longer reach it.
- **`retry_max_attempts` zero guard.** When `retry_enabled=True` and
  `retry_max_attempts=0` the retry loop exited without setting `last_error`,
  causing `raise last_error` to raise `TypeError: exceptions must derive from
  BaseException`. A `RuntimeError` with an actionable message is now raised
  instead, and `ProxyConfig.__post_init__` rejects `retry_max_attempts < 1`
  at construction time.
- **Blocking subprocess on async event loop.** `_read_rtk_lifetime_stats` and
  `_read_lean_ctx_lifetime_stats` called `subprocess.run` directly on the
  asyncio thread. The `initialize_context_tool_session_baseline` function is
  now `async` and offloads the subprocess via `asyncio.to_thread`; the stats
  endpoint uses `await asyncio.to_thread(_get_context_tool_stats)`.
- **Hardcoded Neo4j credential in `docker-compose.yml`.** `NEO4J_AUTH` now
  defaults to `${NEO4J_AUTH:-neo4j/devpassword}` and is documented in
  `.env.example` (excluded from `.gitignore` via `!.env.example`).
- **`SemanticCache.get_memory_stats()` concurrent iteration.** The method
  iterates `self._cache.values()` without holding the async lock. A snapshot
  is now taken via `list(self._cache.values())` before iterating to avoid
  `RuntimeError: dictionary changed size during iteration` under async load.
- **Default Neo4j password in `ProxyConfig`.** `memory_neo4j_password` default
  changed from `"password"` to `""`. The proxy startup path now emits a
  `logger.warning` when `memory_backend == "qdrant-neo4j"` and the password
  is empty, prompting operators to set a real credential.

### Fixed
- **PyPI install clarity and release gating.** Documented `pipx --python python3.13`
  for environments where unsupported Python wheel tags cause older-version
  resolution, made PyPI publish failures block GitHub Releases unless
  `PYPI_SKIP=true`, and added an sdist `LICENSE` invariant.

- **`headroom learn` with claude-cli no longer fails silently on slow
  networks or large digests.** The CLI backend timeout was a hard 120s
  wall-clock cap with no liveness signal: a successful long analysis and
  a hung connection looked identical, and exit 0 with "no recommendations"
  was the only user-visible signal. Two changes:
  (1) **Streaming + idle timeout for claude-cli**: the command now uses
  `--output-format stream-json --verbose` and a watchdog thread reads
  events as they arrive. The process is killed only after
  `HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS` (default 60s) of zero output, or
  after `HEADROOM_LEARN_CLI_TIMEOUT_SECS` (default 300s, was 120s) total.
  Long-but-active analyses run to completion; genuine hangs are caught
  fast. The final `type:"result"` event carries the assistant response.
  Drains stdout/stderr via reader threads so the watchdog works on
  Windows too. (2) **Env-var overrides for all CLI backends**:
  `HEADROOM_LEARN_CLI_TIMEOUT_SECS` is honored by gemini-cli and
  codex-cli as the wall-clock timeout; idle override applies only to the
  streaming claude-cli path.
- **`Learned: error recovery` section in MEMORY.md no longer bloats with
  stale, one-shot, or contradictory entries.** The matchers paired up
  unrelated tool calls (e.g. `state.rs` and `lib.rs` in the same dir
  becoming `File state.rs does not exist. The correct path is lib.rs.`),
  the dedup key was the literal rendered bullet text so near-duplicates
  each created their own row, the shutdown flush dropped the evidence
  gate to 1 so every singleton landed at session end, and there was no
  TTL or re-validation. Fixed at every layer:
  (1) **Emission**: Read recoveries require the failed/successful
  basenames to be identical or close in edit distance; Bash recoveries
  require a shared binary (allowing `python`↔`python3` and
  `ruff`↔`.venv/bin/ruff` variants) plus low-edit-distance OR a shared
  substantive non-flag token. Unrelated pairs are rejected at the source.
  (2) **Dedup**: error-recovery rows are hashed on recovery intent —
  Read on `(basename(error_path), basename(success_path))`, Bash on the
  primary command stripped of volatile suffixes (`| tail -N`, `2>&1`,
  etc.). Near-duplicates collapse into one row.
  (3) **Evidence gating**: default `min_evidence` raised from 2 to 5;
  shutdown-relaxation removed; new `--min-evidence` flag and
  `HEADROOM_MIN_EVIDENCE` envvar so embedded clients can tighten the
  threshold further.
  (4) **Render-time refinement**: drop rows not re-observed in 21 days,
  re-validate Read success paths against the filesystem, collapse
  same-error_path-with-multiple-targets into one "use Glob/Grep first"
  bullet, rank by `evidence_count * 0.5 ** (days/5)`, cap the section
  at 15. A→B / B→A contradiction pairs are also dropped at flush time.
  Patterns now stamp `first_seen_at` / `last_seen_at` on every save;
  `_bump_persisted_evidence` updates them via `json_set`. Other
  `Learned: …` categories (environment, preference, architecture) are
  untouched.
- **`headroom unwrap codex` now actually undoes `headroom wrap codex`** —
  previously there was no `unwrap codex` subcommand at all, so the injected
  `model_provider = "headroom"` / `[model_providers.headroom]` block stayed
  in `~/.codex/config.toml` forever and Codex continued routing through the
  (potentially stopped) proxy, surfacing as `Missing environment variable:
  OPENAI_API_KEY`. `wrap codex` now snapshots the pre-wrap
  `config.toml` to `config.toml.headroom-backup` before its first injection,
  and `unwrap codex` restores that snapshot byte-for-byte (or, if the
  backup is missing, strips only the Headroom-managed block and leaves
  surrounding user content intact). Safe no-op when run without a prior
  wrap. Reported by @raenaryl in Discord.
- **Image compressors now release shared router models after use and proxy shutdown** —
  the proxy/image compression path no longer keeps global `technique-router`
  and `SigLIP` model instances pinned in memory after one-off image
  optimization work. The `get_compressor()` helper now returns a fresh,
  caller-owned compressor instead of a process-lifetime singleton.
- **`headroom learn` no longer clobbers prior recommendations on re-run** —
  the marker block in `CLAUDE.md` / `MEMORY.md` is now merged with the
  prior block instead of wholesale-replaced. Sections re-surfaced by the
  new run win; sections not re-surfaced are carried forward so learnings
  accumulate across runs instead of disappearing. To fully rebuild the
  block, delete it manually and re-run. (#231)
- **`headroom learn` no longer emits dangling cross-references when a
  section is re-surfaced** — the analyzer now includes the project's
  current `<!-- headroom:learn -->` block (from `CLAUDE.md` and
  `MEMORY.md`) in the LLM digest as a "Prior Learned Patterns" section,
  and the system prompt instructs the LLM that re-emitting a section
  replaces the prior one wholesale. Prevents bullets like "`X` is *also*
  large — same rule as `Y`, `Z`" from appearing after `Y` and `Z` got
  dropped during per-section replacement. The writer's section-level
  carry-forward from #231 remains in place as a safety net for sections
  the LLM omits entirely. New helper `extract_marker_block` added to
  `headroom.learn.writer`.

### Added
- **`turn_id` linking agent-loop API calls to a single user prompt** — a new
  `compute_turn_id(model, system, messages)` helper in
  `headroom/proxy/helpers.py` hashes the message prefix up to and including
  the last user-text message, yielding an id that is stable across every
  agent-loop iteration of one prompt but rolls over when the user sends a
  new prompt (or runs `/compact`, `/clear`). `RequestLog` gained a
  `turn_id: str | None` field, which is stamped at every log site
  (anthropic handler bedrock + direct branches, and the streaming handler)
  and surfaced as `turn_id` in `/transformations/feed`. Lets downstream
  consumers (e.g. the Headroom Desktop Activity tab) aggregate savings per
  user prompt rather than per API call.
- **Live flush of traffic-learned patterns to CLAUDE.md / MEMORY.md** — the
  `TrafficLearner` now writes to agent-native context files continuously
  during proxy operation, not just at shutdown. A new dirty-flag debounced
  `_flush_worker` (10s window, `FLUSH_DEBOUNCE_SECONDS`) calls
  `flush_to_file()` whenever `_accumulate()` marks the learner dirty, so
  patterns surface in `CLAUDE.md` / `MEMORY.md` near real-time. Flushes
  read both persisted rows (via `_load_persisted_patterns_from_sqlite`)
  and the in-memory accumulator, bucket patterns by project via the learn
  plugin registry (`plugin.discover_projects()` + longest-path anchoring
  in `_project_for_pattern`), and route by `PatternCategory` to the
  correct file (`_patterns_to_recommendations` +
  `_CATEGORY_TO_TARGET`). Live flushes require `evidence_count >= 2`;
  the shutdown flush accepts single-evidence rows.

### Fixed
- **Traffic-learner evidence count stuck at 1; duplicate DB rows across
  restarts.** `_accumulate` queued patterns with the default
  `ExtractedPattern.evidence_count = 1` regardless of how many times the
  pattern was actually seen, so every persisted row landed at `1` and
  never crossed the live-flush gate (`evidence_count >= 2`). Worse, once
  a pattern was in `_saved_hashes` it was early-returned on every
  re-sighting, and `_saved_hashes` reset on process restart — so a second
  sighting in a later session inserted a duplicate row rather than
  bumping the existing one. Now: `_accumulate` writes the real
  accumulated count at save time, `start()` hydrates `_saved_hashes` +
  a new `_persisted_ids` map from the DB, and re-sightings bump the
  persisted row's `metadata.evidence_count` via an atomic `json_set`
  `UPDATE` (`_bump_persisted_evidence`). `_load_persisted_patterns_from_sqlite`
  now filters via `json_extract(metadata, '$.source')` instead of a
  LIKE on the raw JSON string, so rows survive metadata rewrites.

### Added
- **`HEADROOM_QDRANT_*` environment variables for memory Qdrant configuration**
  (#31) — `Memory(backend="qdrant-neo4j")`, `Mem0Config`, `MemoryConfig`, and
  `ProxyConfig` now resolve their Qdrant connection from
  `HEADROOM_QDRANT_URL`, `HEADROOM_QDRANT_HOST`, `HEADROOM_QDRANT_PORT`,
  `HEADROOM_QDRANT_API_KEY`, `HEADROOM_QDRANT_HTTPS`,
  `HEADROOM_QDRANT_PREFER_GRPC`, and `HEADROOM_QDRANT_GRPC_PORT`. Explicit
  constructor arguments still win; unset env keeps the existing
  `localhost:6333` defaults. Adds matching `--memory-qdrant-{url,host,port,api-key}`
  CLI flags. Enables hosted Qdrant (Qdrant Cloud) and shared/remote Qdrant
  stacks without code changes. New helper:
  [`headroom/memory/qdrant_env.py`](headroom/memory/qdrant_env.py).
- **Telemetry stack & install-mode identity fields** — anonymous beacon now
  reports `headroom_stack` (how Headroom is invoked: `proxy`, `wrap_claude`,
  `adapter_ts_openai`, ...) and `install_mode` (`wrapped` / `persistent` /
  `on_demand`), plus `requests_by_stack` for proxies that serve multiple
  integrations. Proxy exposes a `by_stack` bucket alongside `by_provider` /
  `by_model` on `/stats`, a matching `headroom_requests_by_stack` Prometheus
  counter, and an `X-Headroom-Stack` header honored by the FastAPI middleware.
  `headroom wrap <tool>` sets `HEADROOM_STACK=wrap_<agent>`; the TS SDK and
  all four adapters (`openai`, `anthropic`, `gemini`, `vercel-ai`) tag their
  compress calls. Schema migration:
  [`sql/upgrade_telemetry_stack_context.sql`](sql/upgrade_telemetry_stack_context.sql).
- **Canonical filesystem contract** (issue #175) — new `HEADROOM_CONFIG_DIR`
  (default `~/.headroom/config`, read-mostly) and `HEADROOM_WORKSPACE_DIR`
  (default `~/.headroom`, read-write state) env vars recognized by the Python
  proxy/CLI and the npm SDK. Additive; all existing per-resource env vars
  (`HEADROOM_SAVINGS_PATH`, `HEADROOM_TOIN_PATH`,
  `HEADROOM_SUBSCRIPTION_STATE_PATH`, `HEADROOM_MODEL_LIMITS`) continue to
  work with identical semantics. Docker install scripts and
  `docker-compose.native.yml` forward the new vars into containers so
  savings, logs, and telemetry resolve to the bind-mounted `.headroom` path.
  See [`wiki/filesystem-contract.md`](wiki/filesystem-contract.md).

### Changed
- **`/stats-history` now returns compact checkpoint history by default** — the
  JSON response keeps recent checkpoints dense while evenly sampling older
  checkpoints so long-running installs do not return ever-growing payloads.
  Add `history_mode=full` to fetch the full retained checkpoint list, or
  `history_mode=none` to skip it entirely while still receiving the derived
  hourly/daily/weekly/monthly rollups. Responses now include a
  `history_summary` block describing stored versus returned points.

### Fixed
- **Streaming Anthropic requests are now visible to `/stats.recent_requests`
  and `/transformations/feed`** — `_finalize_stream_response` did not call
  `self.logger.log(...)`, so the entire streaming Anthropic code path (the
  one Claude Code uses) silently bypassed the request logger. Only the
  non-streaming Anthropic path and the Bedrock streaming path were logged.
  As a consequence, `--log-messages` had no observable effect on the live
  transformations feed for typical traffic. The streaming finalizer now
  emits the same `RequestLog` shape the other paths do, including
  `request_messages` when `log_full_messages` is enabled.

## [0.5.22] - 2026-04-11

### Added
- **Cross-agent memory** — Claude saves a fact, Codex reads it back. All agents sharing one proxy share one memory store. Project-scoped DB at `.headroom/memory.db`, auto user_id from `$USER`.
- **Agent provenance tracking** — every memory records which agent saved it (`source_agent`, `source_provider`, `created_via`), with edit history on updates.
- **LLM-mediated dedup** — on `memory_save`, enriched response hints similar existing memories to the LLM. Background async dedup auto-removes >92% cosine duplicates. Zero extra LLM calls.
- **Memory for OpenAI and Gemini handlers** — context injection + tool handling wired into all three provider handlers (Anthropic, OpenAI, Gemini).
- **Plugin architecture for `headroom learn`** — each agent (Claude, Codex, Gemini) is a self-contained plugin. External plugins register via `headroom.learn_plugin` entry points. `--agent` flag for CLI.
- **GeminiScanner** for `headroom learn` — reads `~/.gemini/tmp/*/chats/session-*.json` and `.jsonl`.
- **Code graph integration** — `headroom wrap claude --code-graph` auto-indexes the project via [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) for call-chain traversal, impact analysis, and architectural queries. Opt-in, ~200 token overhead with Claude Code's MCP Tool Search.
- **OpenAI embedder auto-detection** — memory backend uses OpenAI embeddings when `sentence-transformers` is unavailable (no torch/2GB dependency needed).
- **Live traffic learning flush** — `headroom wrap <agent> --learn` flushes learned patterns to the correct agent-native file (MEMORY.md / AGENTS.md / GEMINI.md) at proxy shutdown.

### Changed
- **CodeCompressor disabled by default** — AST-based code compression produced invalid syntax on 40% of real files. Code now passes through uncompressed. Use `--code-graph` for code intelligence instead, or re-enable with `--code-aware`.
- **Shared tool name map** — consolidated tool normalization across all learn plugins into `_shared.py`.
- **Dynamic CLI agent detection** — `headroom learn` discovers agents via plugin registry, no hardcoded choices.

### Fixed
- **CodeCompressor statement-based truncation** — body truncation now walks AST statements (not lines), never cuts mid-expression. Fixes syntax errors on multi-line dict literals and function calls.
- **Docstring FIRST_LINE mode** — uses source lines directly instead of reconstructing from byte offsets. Properly handles all quote styles.
- **Memory shutdown queue drain** — patterns in the save queue were lost on proxy shutdown. Now drained before exit.

## [Unreleased]

### Added
- **Codex-proxy resilience hardening** — reduces event-loop starvation under cold-start reconnect storms
  - **Stage-timing instrumentation** — per-stage durations for both Codex WS accept and Anthropic `/v1/messages` pre-upstream phases emitted as a single `STAGE_TIMINGS` structured log line per request plus Prometheus histograms
  - **Per-pipeline shared warmup** — Anthropic + OpenAI pipelines eagerly load compressors/parsers once at startup; status merged into `WarmupRegistry` for `/debug/warmup` and `/readyz`
  - **WS session registry** — first-class tracking of active Codex WS sessions with deterministic relay-task cancellation and termination-cause classification (`client_disconnect`, `upstream_error`, `client_timeout`, etc.)
  - **Bounded pre-upstream Anthropic concurrency** — `--anthropic-pre-upstream-concurrency` / `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY` caps simultaneous `/v1/messages` pre-upstream work (body read, deep copy, first compression stage, memory-context lookup, upstream connect) so replay storms cannot starve `/livez`, `/readyz`, and new Codex WS opens. Default: auto `max(2, min(8, cpu_count))`; `0` or negative disables (unbounded)
  - **Loopback-only debug endpoints** — `/debug/tasks`, `/debug/ws-sessions`, `/debug/warmup` return `404` (not `403`) to non-loopback callers so external scanners cannot enumerate them
  - **Reconnect-storm repro harness** — `scripts/repro_codex_replay.py` drives concurrent WS + HTTP replay traffic against a local proxy and asserts `/livez` p99 under threshold; `--json` output routes JSON to stdout and the human summary to stderr
- **Proxy liveness and readiness health checks**
  - Adds `GET /livez` for process liveness and `GET /readyz` for traffic readiness
  - Keeps `GET /health` backward compatible while expanding it with readiness details and subsystem checks
  - Eagerly initializes configured memory backends during proxy startup so readiness reflects real serving capability
  - Wires `/readyz` into the Docker image `HEALTHCHECK` and the example `docker-compose.yml`
- **Durable proxy savings history**
  - Persists proxy compression savings history locally at `~/.headroom/proxy_savings.json`
  - Supports `HEADROOM_SAVINGS_PATH` to override the storage location
  - Adds `/stats-history` with lifetime totals plus hourly/daily/weekly/monthly rollups
  - Supports JSON and CSV export from `/stats-history`
  - Extends `/stats` with a `persistent_savings` block while keeping `savings_history` backward compatible
  - Adds a historical mode to `/dashboard` backed by `/stats-history`, including export actions
- **Proxy telemetry SDK override** via `HEADROOM_SDK`
  - Downstream apps can override the anonymous telemetry `sdk` field without patching installed files
  - Blank values fall back to the default `proxy` label
- **`headroom learn`** — Offline failure learning for coding agents
  - Analyzes past conversation history (Claude Code, extensible to Cursor/Codex)
  - **Success correlation**: for each failure, finds what succeeded after and extracts the specific correction
  - 5 analyzers: Environment, Structure, Command Patterns, Retry Prevention, Cross-Session
  - Writes specific learnings to CLAUDE.md (stable project facts) and MEMORY.md (session patterns)
  - Generic architecture: tool-agnostic `ToolCall` model, pluggable Scanner/Writer adapters
  - Dry-run by default, `--apply` to write, `--all` for all projects
  - Example output: "FirstClassEntity.java is not at axion-formats/ — actually at axion-scala-common/"
- **Read Lifecycle Management** — Event-driven compression of stale/superseded Read outputs
  - Detects when a Read output becomes stale (file was edited after) or superseded (file was re-read)
  - Replaces stale/superseded content with compact CCR markers, stores originals for retrieval
  - 75% of Read output bytes are provably stale or redundant (from real-world analysis of 66K tool calls)
  - Fresh Reads (latest read, no subsequent edit) are never touched — Edit safety preserved
  - Opt-in via `ReadLifecycleConfig(enabled=True)`, disabled by default
  - Handles both OpenAI and Anthropic message formats
- **any-llm backend** - Route requests through 38+ LLM providers (OpenAI, Mistral, Groq, Ollama, etc.) via [any-llm](https://mozilla-ai.github.io/any-llm/providers/)
  - Enable with `--backend anyllm --anyllm-provider <provider>`
  - Install with: `pip install 'headroom-ai[anyllm]'`
- Production-ready proxy server with caching, rate limiting, and metrics
- CLI command `headroom proxy` to start the proxy server
- **IntelligentContextManager** (semantic-aware context management)
  - Multi-factor importance scoring: recency, semantic similarity, TOIN importance, error indicators, forward references, token density
  - No hardcoded patterns - all importance signals learned from TOIN or computed from metrics
  - TOIN integration for retrieval_rate and field_semantics-based scoring
  - Strategy selection: NONE, COMPRESS_FIRST, DROP_BY_SCORE based on budget overage
  - Atomic tool unit handling (call + response dropped together)
  - Configurable scoring weights via `ScoringWeights` dataclass
  - `IntelligentContextConfig` for full configuration control
  - Backwards compatible with `RollingWindowConfig`
- **LLMLingua-2 Integration** (opt-in ML-based compression)
  - `LLMLinguaCompressor` transform using Microsoft's LLMLingua-2 model
  - Content-aware compression rates (code: 0.4, JSON: 0.35, text: 0.3)
  - Memory management utilities: `unload_llmlingua_model()`, `is_llmlingua_model_loaded()`
  - Proxy integration via `--llmlingua` flag
  - Device selection: `--llmlingua-device` (auto/cuda/cpu/mps)
  - Custom compression rate: `--llmlingua-rate`
  - Helpful startup hints when llmlingua is available but not enabled
  - ~~Install with: `pip install headroom-ai[llmlingua]`~~ (the `[llmlingua]` extra was removed in 0.9.x)
- **Code-Aware Compression** (AST-based, syntax-preserving)
  - `CodeAwareCompressor` transform using tree-sitter for AST parsing
  - Supports Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
  - Preserves imports, function signatures, type annotations, error handlers
  - Compresses function bodies while maintaining structural integrity
  - Guarantees syntactically valid output (no broken code)
  - Automatic language detection from code patterns
  - Memory management: `is_tree_sitter_available()`, `unload_tree_sitter()`
  - Uses `tree-sitter-language-pack` for broad language support
  - Install with: `pip install headroom-ai[code]`
- **ContentRouter** (intelligent compression orchestrator)
  - Auto-routes content to optimal compressor based on type detection
  - Source hint support for high-confidence routing (file paths, tool names)
  - Handles mixed content (e.g., markdown with code blocks)
  - Strategies: CODE_AWARE, SMART_CRUSHER, SEARCH, LOG, TEXT, LLMLINGUA
  - Configurable strategy preferences and fallbacks
  - Routing decision log for transparency and debugging
- **Custom Model Configuration**
  - Support for new models: Claude 4.5 (Opus), Claude 4 (Sonnet, Haiku), o3, o3-mini
  - Pattern-based inference for unknown models (opus/sonnet/haiku tiers)
  - Custom model config via `HEADROOM_MODEL_LIMITS` environment variable
  - Config file support: `~/.headroom/models.json`
  - Graceful fallback for unknown models (no crashes)
  - Updated pricing data for all current models

### Fixed
- **Event.wait task leak in subscription trackers** — `asyncio.shield` pattern prevents cancellation of the outer `wait_for` from leaking the inner `Event.wait` task
- **Python 3.10 compatibility for memory-context fail-open** — catches `asyncio.TimeoutError` (the 3.10-compatible alias) rather than `TimeoutError` to preserve behaviour on older runtimes
- **uvicorn `proxy_headers=False`** — refuses `Forwarded` / `X-Forwarded-For` rewrites so the loopback guard on `/debug/*` cannot be spoofed by a misconfigured reverse proxy
- **First-frame timeout for Codex WS accepts** — guards against a client that opens a handshake and never sends the first frame; relays cancel deterministically with `client_timeout`
- **Semaphore leak on unexpected exception in Anthropic pre-upstream path** — the finalizer now releases the pre-upstream semaphore on every exit path (early 4xx, cache hit, upstream error, streaming handoff)
- **`active_relay_tasks` gauge double-decrement** — `deregister_and_count` returns `(handle, released_task_count)` atomically so the handler decrements the Prometheus gauge by the exact number it registered, eliminating drift

### Internal
- **IPv6-mapped loopback recognition** — the loopback guard parses `::ffff:127.0.0.1` and other dual-stack literals through `ipaddress.ip_address(...).is_loopback`
- **Lock-free stage-timing accumulators** — `record_stage_timings` writes to per-path counters that do not contend with `/metrics` export or `record_request`
- **Narrow `contextlib.suppress` in relay classification** — only `CancelledError` is suppressed where we reclassify it; other exceptions propagate so termination cause stays truthful
- **`jitter_delay_ms` helper** — shared exponential-backoff + 50-150% jitter formula in `headroom/proxy/helpers.py`; used by three proxy retry sites and mirrored inline in the repro harness

## [0.2.0] - 2025-01-07

### Added
- **SmartCrusher**: Statistical compression for tool outputs
  - Keeps first/last K items, errors, anomalies, and relevance matches
  - Variance-based change point detection
  - Pattern detection (time series, logs, search results)
- **Relevance Scoring Engine**: ML-powered item relevance
  - `BM25Scorer`: Fast keyword matching (zero dependencies)
  - `EmbeddingScorer`: Semantic similarity with sentence-transformers
  - `HybridScorer`: Adaptive combination of both methods
- **CacheAligner**: Prefix stabilization for better cache hits
  - Dynamic date extraction
  - Whitespace normalization
  - Stable prefix hashing
- **RollingWindow**: Context management within token limits
  - Drops oldest tool units first
  - Never orphans tool results
  - Preserves recent turns
- **Multi-Provider Support**:
  - Anthropic with official `count_tokens` API
  - Google with official `countTokens` API
  - Cohere with official `tokenize` API
  - Mistral with official tokenizer
  - LiteLLM for unified interface
- **Integrations**:
  - LangChain callback handler (`HeadroomOptimizer`)
  - MCP (Model Context Protocol) utilities
- **Proxy Server** (`headroom.proxy`):
  - Semantic caching with LRU eviction
  - Token bucket rate limiting
  - Retry with exponential backoff
  - Cost tracking with budget enforcement
  - Prometheus metrics endpoint
  - Request logging (JSONL)
- **Pricing Registry**: Centralized model pricing with staleness tracking
- **Benchmarks**: Performance benchmarks for transforms and relevance scoring

### Changed
- Improved token counting accuracy across all providers
- Enhanced tool output compression with relevance-aware selection

### Fixed
- Mistral tokenizer API compatibility
- Google token counting for multi-turn conversations

## [0.1.0] - 2025-01-05

### Added
- Initial release
- `HeadroomClient`: OpenAI-compatible client wrapper
- `ToolCrusher`: Basic tool output compression
- Audit mode for observation without modification
- Optimize mode for applying transforms
- Simulate mode for previewing changes
- SQLite and JSONL storage backends
- HTML report generation
- Streaming support

### Safety Guarantees
- Never removes human content
- Never breaks tool ordering
- Parse failures are no-ops
- Preserves recency (last N turns)

---

## Migration Guide

### From 0.1.x to 0.2.x

The 0.2.0 release is backward compatible. New features are opt-in:

```python
# Old code still works
from headroom import HeadroomClient, OpenAIProvider

# New SmartCrusher (replaces ToolCrusher for better compression)
from headroom import SmartCrusher, SmartCrusherConfig

config = SmartCrusherConfig(
    min_tokens_to_crush=200,
    max_items_after_crush=50,
)
crusher = SmartCrusher(config)

# New relevance scoring
from headroom import create_scorer

scorer = create_scorer("hybrid")  # or "bm25" for zero deps
```

### Using the Proxy

New in 0.2.0 - run Headroom as a proxy server:

```bash
# Start the proxy
headroom proxy --port 8787

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

[Unreleased]: https://github.com/chopratejas/headroom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/chopratejas/headroom/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chopratejas/headroom/releases/tag/v0.1.0

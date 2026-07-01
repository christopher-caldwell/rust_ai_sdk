# Independent Audit: reviewer-03

## Scope Reviewed
Reviewed the repository at HEAD `ece26117a971286911cd7bb6e459b3e19ca9f14a`. I used `.understand-anything/knowledge-graph.json` only for orientation, then verified conclusions against `Cargo.toml`, `README.md`, `CHANGELOG.md`, `Justfile`, `release-plz.toml`, `scripts/publish-crate.sh`, `src/**`, and `examples/**`. I independently ran short build checks: `cargo check --all-targets`, core/provider feature combinations, `cargo test`, `cargo test --all-features`, example `cargo check` commands, `cargo doc --all-features --no-deps`, and `cargo package --list --allow-dirty`.

## Executive Summary
The crate is compact, has a reasonable provider-neutral shape, and the checked feature combinations/tests pass. The strongest source-level concern is that `generate_text` can send tool-capable requests but silently discard returned tool calls because it maps only text into `TextResult`. The strongest release concern is that the current package archive includes local audit artifacts because `audits/**` is not excluded. I also found a practical reliability risk: provider clients use default `reqwest::Client` instances with no public timeout/client configuration.

## Findings

### Finding 1: Published crate archive includes local audit artifacts

**Severity:** medium
**Confidence:** 9
**Category:** other
**Location:** `Cargo.toml` lines 11-18; `scripts/publish-crate.sh` lines 191-193; independently generated `cargo package --list --allow-dirty` output
**Status:** issue

**Issue:**
The package exclude list omits `audits/**`, so local audit files are included in the crate archive that would be uploaded to crates.io.

**Evidence:**
`Cargo.toml` excludes `.understand-anything/**`, `.github/**`, `examples/**`, `issues*.md`, `release-plz.toml`, and `target/**`, but not `audits/**`. The publish script's package listing step runs `cargo package --list` / `cargo package --list --allow-dirty`. My independent package listing showed audit paths in the package file list, including the audit config and reviewer markdown files.

**Impact:**
This can leak internal review material into an immutable published crate archive and makes the package look unprofessional. It also means release validation currently reports the problem but does not fail on it.

**Recommended Fix:**
Add `audits/**` to `Cargo.toml` `exclude`, and consider excluding local agent/instruction files that are not intended for consumers. Add a release preflight assertion that fails if packaged paths match known local-only directories.

### Finding 2: `generate_text` silently discards tool calls from tool-capable requests

**Severity:** medium
**Confidence:** 8
**Category:** correctness
**Location:** `src/runtime/generate.rs` lines 8-12; `src/providers/openai/types.rs` lines 389-449; `src/providers/anthropic/types.rs` lines 233-353; `src/providers/gemini/types.rs` lines 219-282 and 453-548
**Status:** issue

**Issue:**
The public `generate_text` path accepts a full `TextRequest`, including tools and tool choice, but provider text-result mappers keep only text and drop tool-call data. If a model returns a tool call, callers can receive an empty or partial `TextResult` instead of a structured tool-call result or an error.

**Evidence:**
`generate_text` delegates directly to `model.generate(request)`. Provider `generate` methods validate and send the request as-is. OpenAI request translation includes tools, while `chat_response_to_text_result` reads only `choice.message.content`; the separate chat-result mapper handles `choice.message.tool_calls`. Anthropic request translation includes tools, while `anthropic_response_to_text_result` concatenates only `"text"` blocks and ignores `"tool_use"` blocks; the chat-result mapper handles those. Gemini request translation includes function declarations/tool config, while `gemini_response_to_text_result` collects only `part.text`; the chat-result mapper handles `functionCall`.

**Impact:**
Application code using `generate_text` with tool definitions can lose the model's requested action. That is a real correctness problem for tool-enabled agents: an apparently successful call may return no usable answer and no executable tool call.

**Recommended Fix:**
Either reject tool-capable requests on `generate_text`/provider `generate` with a clear validation error, or make the text-result path detect returned tool calls and return an error directing callers to `generate_chat`, `stream_text`, or `run_turn`. Add provider tests covering non-streaming tool-call responses through `generate`.

### Finding 3: Provider HTTP clients have no public timeout or client configuration

**Severity:** low
**Confidence:** 8
**Category:** reliability
**Location:** `src/providers/openai/client.rs` lines 43-59 and 201-208; `src/providers/anthropic/client.rs` lines 46-62 and 189-196; `src/providers/gemini/client.rs` lines 38-54 and 194-201
**Status:** risk

**Issue:**
Each provider constructs a default `reqwest::Client::new()` internally and exposes only `new(api_key, model)` at the public model layer. There is no public way to configure request timeout, connect timeout, proxy, retry policy, or a custom HTTP client.

**Evidence:**
OpenAI, Anthropic, and Gemini clients all store a `reqwest::Client`, initialize it with `reqwest::Client::new()`, and await `.send()` directly. The only base URL override is `pub(crate)` and used for tests. The public provider model constructors only call these default clients.

**Impact:**
In production services, provider calls can hang longer than the application expects unless every caller wraps SDK futures externally. That makes cancellation and operational behavior harder to reason about and is below the reliability bar expected from a server-oriented SDK.

**Recommended Fix:**
Expose builder-style configuration for timeout/base URL/custom `reqwest::Client`, and set a conservative default timeout if the crate wants safe defaults. At minimum, document that callers must wrap calls in `tokio::time::timeout`.

## Non-Issues / Things Checked
- Core, individual provider, all-provider streaming, and message-stream feature combinations compiled successfully.
- `cargo test` passed with 96 tests; `cargo test --all-features` passed with 107 tests.
- Standalone and chatbot example manifests compiled with `cargo check`.
- `cargo doc --all-features --no-deps` built successfully.
- Runtime stream accumulation stops after `Finished`, preserves malformed tool JSON metadata, and has tests for parallel tool calls and provider metadata.
- Error mapping keeps provider, status, body snippet, code, and error type where the current provider adapters parse them.
- Gemini tool-call continuation preserves provider call IDs and thought signatures through provider metadata.

## Assumptions
- External provider API documentation and live APIs were out of scope, so model ID freshness and wire-format compatibility were not independently verified against external docs.
- I treated generated command output from the commands I ran as allowed verification material.
- I did not treat style-only concerns as findings unless they affected correctness, reliability, packaging, or user trust.

## Open Questions
- Should `generate_text` intentionally be usable with tool definitions, or should tool-capable requests be reserved for `generate_chat`, `stream_text`, and `run_turn`?
- Does the project intend to publish local agent instructions such as `AGENTS.md` in the crate archive, or should those be excluded with the audit artifacts?
- Should provider clients support OpenAI-compatible/custom endpoints publicly, or is this crate intentionally limited to first-party endpoints?

# Independent Audit: reviewer-04

## Scope Reviewed
Reviewed HEAD `ece26117a971286911cd7bb6e459b3e19ca9f14a` for the `another-ai-sdk` Rust crate. I used `.understand-anything/knowledge-graph.json` first for orientation, then verified against current source files under `src/**`, examples, package metadata, README/CHANGELOG/Justfile/release-plz config, and `scripts/publish-crate.sh`.

Independently ran:

- `cargo test`
- `cargo test --all-features`
- `cargo check --all-targets`
- `cargo check --no-default-features`
- `cargo check --no-default-features --features streaming`
- `cargo check --no-default-features --features message-stream`
- `cargo check --no-default-features --features openai`
- `cargo check --no-default-features --features anthropic`
- `cargo check --no-default-features --features gemini`
- `cargo check --manifest-path examples/standalone/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml`

All commands completed successfully. I did not run `just publish-dry` or release automation.

## Executive Summary
Overall, the crate is in better shape than I expected for a small provider-neutral SDK. Core request validation, provider translation tests, streaming accumulation, malformed tool JSON handling, feature-gated compilation, and examples all passed the checks I ran. The main risks I found are package reliability/usability gaps rather than immediate compile failures: provider clients have no public timeout/client configuration, tool definitions are not validated before provider calls, and some release/package documentation is inconsistent with the crate being publishable.

## Findings

### Finding 1: Provider HTTP clients have no timeout or public client configuration

**Severity:** medium
**Confidence:** 8
**Category:** reliability
**Location:** `src/providers/openai/client.rs` lines 44-58, `src/providers/anthropic/client.rs` lines 47-61, `src/providers/gemini/client.rs` lines 39-53; model constructors in `src/providers/*/model.rs`
**Status:** risk

**Issue:**
All provider clients construct a default `reqwest::Client` internally, and the public model constructors only accept an API key and model id. There is no public builder or constructor that lets applications set request timeouts, inject a configured HTTP client, configure proxies, or select alternate base URLs.

**Evidence:**
`OpenAiClient::new`, `AnthropicClient::new`, and `GeminiClient::new` each store `http: reqwest::Client::new()`. Their `with_base_url` helpers are `pub(crate)` and are only usable by tests. Public `OpenAiChatModel::new`, `AnthropicChatModel::new`, and `GeminiChatModel::new` call those default clients directly.

**Impact:**
Server users can have model calls wait indefinitely on stalled network connections unless they wrap every SDK call in their own timeout. They also cannot use OpenAI-compatible endpoints, custom transports, corporate proxy settings, or tuned connection pools without forking or adding ad hoc wrappers. For a Rust SDK intended for reliable server use, HTTP configuration is part of practical stability.

**Recommended Fix:**
Add public configuration APIs, for example provider model/client builders with `timeout(Duration)`, `http_client(reqwest::Client)`, and `base_url(String)` options. Consider setting a conservative default timeout even when callers do not provide one.

### Finding 2: Tool definitions are not validated before provider requests

**Severity:** low
**Confidence:** 8
**Category:** correctness
**Location:** `src/core/tool.rs` lines 3-23; `src/core/request.rs` lines 75-80 and 184-221; provider serializers in `src/providers/*/types.rs`
**Status:** issue

**Issue:**
`TextRequest::validate()` validates message shape, generation options, and `tool_choice`, but it does not validate the tool definitions themselves. Empty tool names/descriptions, duplicate tool names, and non-object or provider-invalid schemas can be sent directly to providers.

**Evidence:**
`ToolDefinition::new` accepts any `name`, `description`, and `serde_json::Value` schema without checks. `TextRequest::validate()` calls `validate_request_has_messages`, `validate_generation_options`, `validate_tool_choice`, and `validate_message_sequence`, but no tool-definition validator. Provider serializers then pass `tool.name`, `tool.description`, and `tool.input_schema` through to OpenAI, Anthropic, or Gemini request bodies. `validate_required_tool_choice` only checks whether at least one tool has the required name; it does not detect duplicate definitions.

**Impact:**
Bad tool definitions fail later as provider API errors instead of as local validation errors with clear diagnostics. Duplicate names can also make application behavior ambiguous because providers receive duplicate declarations while `ToolRegistry` stores handlers by name and silently replaces earlier registrations.

**Recommended Fix:**
Extend request validation to check tool definitions before provider calls. At minimum reject blank names, blank descriptions, duplicate names, and schemas that are not JSON objects. If provider-specific name/schema constraints differ, keep provider-neutral baseline validation in core and add stricter adapter validation where necessary.

### Finding 3: Published-package installation docs still instruct path dependencies

**Severity:** low
**Confidence:** 9
**Category:** documentation
**Location:** `README.md` lines 29-51; `Cargo.toml` lines 1-10; `scripts/publish-crate.sh` lines 7-9 and 253-318
**Status:** issue

**Issue:**
The README installation section says the crate is "currently used from this repository" and tells users to add a path dependency. At the same time, the crate has normal package metadata and release tooling for publishing to crates.io.

**Evidence:**
`Cargo.toml` declares package name `another-ai-sdk`, version `0.0.4`, repository, readme, keywords, categories, and license. `scripts/publish-crate.sh` runs `cargo publish --dry-run` and supports `--publish`. However, README lines 31-51 only show path dependencies such as `another-ai-sdk = { path = "/path/to/rust_ai_sdk" }` and the `message-stream` example also uses a path dependency.

**Impact:**
If the crate is published, crates.io users see installation instructions that do not tell them how to depend on the published package. That hurts package usability and trust, especially for Rust users evaluating whether this is a serious crate.

**Recommended Fix:**
Add a normal crates.io dependency snippet, for example `another-ai-sdk = "0.0.4"` or a version-range recommendation, and keep path dependency instructions in an "Using from this repository" subsection.

### Finding 4: Release script output gives conflicting tag instructions

**Severity:** low
**Confidence:** 8
**Category:** maintainability
**Location:** `scripts/publish-crate.sh` lines 24-29, 256-290, 294-318
**Status:** issue

**Issue:**
The release script both automatically tags/pushes during `--publish` and tells users after prepare/dry-run to tag manually after publishing.

**Evidence:**
The usage text says `scripts/publish-crate.sh --publish` publishes, tags the release commit, and pushes the tag. The publish path calls `ensure_tag_is_available`, then `cargo_publish_release`, then `tag_and_push_release`. But the prepare and dry-run messages tell users: "After publishing, tag the released commit: git tag $tag; git push origin $tag".

**Impact:**
The stale instructions can lead users to attempt a duplicate manual tag after `--publish`, or to misunderstand whether tagging is handled by automation. Release scripts should be unambiguous because mistakes there are hard to recover from once a crate version is uploaded.

**Recommended Fix:**
Update the prepare/dry-run messages to say that `scripts/publish-crate.sh --publish` will tag and push automatically, or remove automatic tagging if the intended workflow is manual.

## Non-Issues / Things Checked
- The checked-out HEAD matched the requested commit.
- Default and all-feature tests passed, including provider streaming, malformed tool JSON preservation, and runtime tool-loop tests.
- `cargo check` passed for no-default, provider-specific, `streaming`-only, and `message-stream`-only feature combinations.
- Standalone and chatbot Rust examples compiled successfully.
- Core message sequence validation rejects empty requests, invalid tool-result ordering, duplicate pending tool call IDs, and invalid `tool_choice` references.
- OpenAI streaming carries forward response metadata and trailing usage in the covered tests.
- Anthropic streaming preserves tool-use deltas and malformed JSON metadata in the covered tests.
- Gemini tool-call metadata for function call IDs and thought signatures is intentionally preserved for continuations.
- The publish script performs substantial preflight checks before upload; I did not run publish dry-run automation because the audit instructions disallowed it.

## Assumptions
- I treated `.understand-anything/knowledge-graph.json` as orientation only because it was generated from an older commit than the audit target.
- I treated examples as trust/usability material, not as published crate contents, because `Cargo.toml` excludes `examples/**`.
- I did not use external provider documentation; provider-specific comments are based on source behavior and general SDK reliability expectations.

## Open Questions
- Should the crate expose provider builders for advanced users now, or intentionally keep the public API minimal until a later version?
- Is the crate intended to be consumed from crates.io today, or is publishing infrastructure being prepared ahead of public installation docs?
- Should core validation enforce provider-neutral tool schema requirements, or should each provider adapter own stricter tool validation?

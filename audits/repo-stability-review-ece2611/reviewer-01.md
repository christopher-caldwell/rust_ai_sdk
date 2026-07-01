# Independent Audit: reviewer-01

## Scope Reviewed

I reviewed the repository at HEAD `ece26117a971286911cd7bb6e459b3e19ca9f14a`. I used `.understand-anything/knowledge-graph.json` only for initial orientation, then verified findings against current allowed files.

Reviewed areas:

- Top-level package metadata and release files: `Cargo.toml`, `Cargo.lock`, `README.md`, `CHANGELOG.md`, `Justfile`, `release-plz.toml`, and `scripts/publish-crate.sh`.
- Core SDK types and validation under `src/core/**`.
- Runtime generation, streaming, tool registry, turn accumulation, continuation, and message-stream helpers under `src/runtime/**`.
- OpenAI, Anthropic, and Gemini provider request/response/stream adapters under `src/providers/**`.
- Standalone and chatbot examples under `examples/**`.

Independent verification commands run:

- `cargo check --no-default-features --features streaming`
- `cargo check --no-default-features --features message-stream`
- `cargo check --no-default-features --features providers-all`
- `cargo check --all-targets`
- `cargo test`
- `cargo test --all-features`
- `cargo check --manifest-path examples/standalone/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml`
- `cargo doc --all-features --no-deps`

All verification commands passed.

## Executive Summary

The crate is small, coherent, and better covered than expected for an early `0.0.x` SDK. The core/provider/runtime layering is clear, provider adapters have focused tests for error mapping and streaming edge cases, and the runtime has useful protections around malformed tool JSON and never-ending streams.

The main correctness gap I found is in tool-call conversation validation. The validator catches unknown, duplicate, and malformed tool result IDs, but it does not reject unresolved pending tool calls or non-tool messages inserted before pending tool calls are answered. That lets callers send provider-invalid continuation requests and receive provider API failures instead of a clear SDK validation error.

The other issues are release/package polish: contradictory tag instructions in the publish script and no explicit `rust-version` despite using edition 2024.

## Findings

### Finding 1: Tool-call sequence validation allows unresolved pending tool calls

**Severity:** medium
**Confidence:** 9
**Category:** correctness
**Location:** `src/core/request.rs`, `validate_message_sequence`, lines 223-233; `validate_tool_results`, lines 321-343; `collect_assistant_tool_calls`, lines 355-386
**Status:** issue

**Issue:**
`TextRequest::validate()` tracks pending assistant tool calls and removes them when matching tool results appear, but it never checks that all pending tool calls have been answered by the end of the request. It also does not reject a new non-tool message while tool calls are still pending.

**Evidence:**
In `validate_message_sequence`, a `HashSet` of pending tool-call IDs is created, passed through every message, and then ignored:

```rust
fn validate_message_sequence(request: &TextRequest) -> Result<(), SdkError> {
    let mut pending_tool_calls = HashSet::new();

    for (message_index, message) in request.messages.iter().enumerate() {
        validate_message_content_storage(message_index, message)?;
        validate_message_shape(message_index, message)?;
        validate_tool_results(message_index, message, &mut pending_tool_calls)?;
        collect_assistant_tool_calls(message_index, message, &mut pending_tool_calls)?;
    }

    Ok(())
}
```

`validate_tool_results` removes matching IDs and rejects unknown/already-used IDs, while `collect_assistant_tool_calls` inserts IDs and rejects empty/duplicate pending IDs. There is no final `pending_tool_calls.is_empty()` check and no check that only tool-result messages may appear while the set is non-empty.

Provider serializers then emit the invalid conversation shape directly:

- OpenAI serializes assistant `tool_calls` and separate `tool` messages in order in `src/providers/openai/types.rs`.
- Anthropic serializes assistant `tool_use` and user-role `tool_result` parts in `src/providers/anthropic/types.rs`.
- Gemini serializes `functionCall` parts and expects later `functionResponse` parts via a running call-ref map in `src/providers/gemini/types.rs`.

**Impact:**
A request can pass SDK validation even though it ends with an assistant tool call and no corresponding tool result, or contains another user/assistant message before answering a pending tool call. That moves a predictable local validation failure into provider-specific API errors. It also weakens the reliability of `ContinuationBuilder`, because partially built or incorrectly completed continuations are not caught before network calls.

**Recommended Fix:**
Make `validate_message_sequence` enforce provider-neutral tool-call closure:

- If `pending_tool_calls` is non-empty and the next message is not a `Role::Tool` message containing tool results, return `SdkError::Validation`.
- After the loop, return `SdkError::Validation` if any pending tool-call IDs remain.
- Add tests for an assistant tool call with no result, a user message before a pending tool result, and multiple tool calls where only one result is supplied.

### Finding 2: Publish script tells operators to manually tag after a path that already tags

**Severity:** low
**Confidence:** 10
**Category:** documentation
**Location:** `scripts/publish-crate.sh`, usage lines 24-29, prepare/check-only messages lines 267-289, publish path lines 315-323
**Status:** issue

**Issue:**
The release script gives contradictory tag instructions. Its usage text says `scripts/publish-crate.sh --publish` publishes, tags, and pushes the tag. The actual publish path does call `tag_and_push_release`. But the prepare and dry-run success messages still instruct the operator to manually run `git tag $tag` and `git push origin $tag` after publishing.

**Evidence:**
The usage flow says:

```text
4. Run: scripts/publish-crate.sh --publish
5. The script publishes the crate, tags the release commit, and pushes the tag.
```

The publish path does:

```bash
echo "==> Publishing"
cargo_publish_release

tag_and_push_release

cat <<MSG

Published $name $version.
Tagged and pushed $tag.
MSG
```

But non-publish success text says:

```text
After publishing, tag the released commit:
  git tag $tag
  git push origin $tag
```

**Impact:**
An operator following the script output after a successful `--publish` run will try to create and push a tag that already exists. At best this is noisy and confusing; at worst it undermines confidence during a permanent crates.io publish flow.

**Recommended Fix:**
Remove the manual tag instructions from the prepare and check-only messages, or replace them with a statement that `scripts/publish-crate.sh --publish` will create and push the tag after a successful publish.

### Finding 3: The crate uses edition 2024 without declaring `rust-version`

**Severity:** low
**Confidence:** 8
**Category:** maintainability
**Location:** `Cargo.toml`, package metadata lines 1-10
**Status:** recommendation

**Issue:**
`Cargo.toml` declares `edition = "2024"` but does not declare a `rust-version`. That leaves the crate's MSRV implicit.

**Evidence:**
The package section contains name, version, license, edition, description, repository, readme, keywords, and categories, but no `rust-version` field:

```toml
[package]
name = "another-ai-sdk"
version = "0.0.4"
license = "MIT"
edition = "2024"
description = "Provider-neutral Rust SDK for streaming chat and tool calling."
```

**Impact:**
Rust users and downstream CI systems cannot tell the intended compiler floor from package metadata. Because edition 2024 already implies a recent toolchain, making the MSRV explicit would improve package predictability and release discipline.

**Recommended Fix:**
Add a `rust-version` field matching the minimum supported compiler for edition 2024 and the crate's dependencies, then include that version in CI and release checks.

## Non-Issues / Things Checked

- The repository HEAD matched the requested audit commit.
- Default tests, all-feature tests, all-target checks, rustdoc, and the Rust example crate checks all passed.
- Feature combinations checked directly (`streaming` alone, `message-stream` alone, and `providers-all` without default features) compiled successfully.
- Provider HTTP error mapping preserves status/body snippets for setup failures; provider JSON errors are mapped into `SdkError::Api`.
- OpenAI streaming carries response metadata forward, defers finish until trailing usage can arrive, handles malformed tool JSON, and has tests for parallel tool calls.
- Anthropic streaming handles `message_stop`, fallback finish on stream end, tool-use start/delta/ready, mixed text/tool output, malformed tool JSON, and unknown event types.
- Gemini tool calls preserve provider function-call IDs and thought signatures through provider metadata, including continuation serialization.
- `run_turn` and message-stream collection stop after `StreamEvent::Finished`, so provider streams that keep yielding after finish do not hang those runtime helpers.
- `ToolRegistry::execute` rejects malformed JSON tool inputs before invoking application handlers.
- The release script performs meaningful preflight checks before publish: clean worktree, metadata, formatting, tests, all-feature tests, example checks, rustdoc, package listing, and `cargo publish --dry-run`.

## Assumptions

- I treated `.understand-anything/knowledge-graph.json` only as navigation/orientation material and did not use it as evidence for findings.
- I did not use external provider documentation; provider behavior conclusions are based on the SDK's own abstractions, serializers, tests, and documented intent.
- I treated examples as support code rather than part of the published crate, because `Cargo.toml` excludes `examples/**` from the package.
- I assumed a respected Rust package should make its MSRV explicit when using edition 2024, even if Cargo can infer a minimum edition parser requirement.

## Open Questions

- Should `TextRequest` allow incomplete assistant tool-call turns as an intermediate in-memory representation, while provider calls reject them? If yes, validation may need a mode split between "structurally well formed" and "sendable to provider."
- What MSRV does the project intend to support?
- Are model enum values intended to be a tightly verified list or convenience constants that may be updated optimistically between provider releases?

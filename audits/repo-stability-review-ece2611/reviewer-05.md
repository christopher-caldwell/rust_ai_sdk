# Independent Audit: reviewer-05

## Scope Reviewed
Reviewed the repository at HEAD `ece26117a971286911cd7bb6e459b3e19ca9f14a`. I used `.understand-anything/knowledge-graph.json` only for orientation, then verified findings against current source files and allowed project files. Covered the public core API, request validation, provider adapters for OpenAI/Anthropic/Gemini, streaming accumulation, message-stream adapter, examples, manifest, README, Justfile, release config, and publish script.

I independently ran:

- `cargo test`
- `cargo test --all-features`
- `cargo check --all-targets`
- `just check-features`
- `just check-examples`
- `cargo doc --all-features --no-deps`
- `cargo fmt --check`

I did not run `just publish-dry` because the publish script starts with a clean-worktree check that can reveal unrelated `audits/**` paths, which this isolated review is not allowed to inspect.

## Executive Summary
Overall, the package is coherent and substantially better covered than I expected for a small provider-neutral SDK. The provider adapters validate requests before HTTP calls, carry provider metadata through streaming/tool continuations, and have focused tests for streaming, malformed tool JSON, usage propagation, feature combinations, and examples.

I found one practical correctness gap in tool-call request validation and one packaging/documentation usability problem. Neither appears catastrophic, but both matter for a crate that wants to be respected by Rust users: invalid tool histories should be rejected before provider calls, and the README should tell crates.io users how to install the published crate.

## Findings

### Finding 1: Request validation allows unresolved assistant tool calls to be sent as outbound requests

**Severity:** medium
**Confidence:** 8
**Category:** correctness
**Location:** `src/core/request.rs`, `validate_message_sequence`, lines 223-233; tool tracking at lines 321-383; tests at lines 461-490
**Status:** issue

**Issue:**
`TextRequest::validate()` tracks pending assistant tool calls and rejects tool results that do not match a prior call, but it never rejects a request that still has pending tool calls at the end of validation. It also does not reject a non-tool message that appears while a prior assistant tool call remains unresolved.

**Evidence:**
`validate_message_sequence` initializes `pending_tool_calls`, processes each message, and then returns `Ok(())` unconditionally:

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

The helper removes matched results from `pending_tool_calls` and adds assistant calls to it, but there is no final check for remaining IDs. Existing tests cover the valid continuation (`assistant ToolCall` followed by `Tool` result) and reject a tool result without a prior call, but there is no test for an assistant tool call without a following result.

Provider adapters call `request.validate()?` before sending HTTP requests, so this gap affects all providers. OpenAI serializes assistant `tool_calls`; Anthropic serializes `tool_use`; Gemini serializes `functionCall`. With no corresponding tool result/function response in the request history, these are malformed or at least incomplete continuations for practical provider use.

**Impact:**
Applications can accidentally send invalid histories to providers and get provider-specific HTTP/API failures instead of a clear SDK validation error. This weakens the provider-neutral contract and makes tool-loop bugs harder to diagnose, especially when users manually construct `TextRequest` values or resume persisted conversations.

**Recommended Fix:**
After iterating messages, reject any remaining `pending_tool_calls` with a validation error that names the unresolved IDs. Consider also rejecting any `System`, `User`, or `Assistant` message encountered while `pending_tool_calls` is non-empty, unless the message is the assistant message that just introduced those calls. Add tests for:

- request ending immediately after an assistant tool call
- request with assistant tool call, then unrelated user message
- request with multiple tool calls where only one result is supplied

### Finding 2: README installation instructions are path-only despite the crate being configured for publishing

**Severity:** low
**Confidence:** 9
**Category:** documentation
**Location:** `README.md`, Installation section, lines 29-80; `Cargo.toml`, package metadata, lines 1-18
**Status:** issue

**Issue:**
The README tells users the crate is "currently used from this repository" and only shows path dependencies, including the feature examples. At the same time, `Cargo.toml` contains normal crates.io package metadata (`name`, `version`, `license`, `repository`, `readme`, `keywords`, `categories`) and the repo has publish automation.

**Evidence:**
README lines 31-35 instruct:

```toml
another-ai-sdk = { path = "/path/to/rust_ai_sdk" }
```

README lines 49-50 and 69-80 repeat path dependencies for feature examples. `Cargo.toml` declares `name = "another-ai-sdk"`, `version = "0.0.4"`, license, repository, readme, keywords, categories, and package excludes.

**Impact:**
If this crate is published, the README shown on crates.io will direct users away from the normal registry dependency form. That harms package usability and user trust, and it creates unnecessary friction for consumers trying to evaluate or adopt the crate.

**Recommended Fix:**
Change the primary installation examples to crates.io dependency syntax, for example:

```toml
another-ai-sdk = "0.0.4"
```

Then keep path dependencies in a separate "Using the repository examples" or "Local development" subsection. Update the feature examples to show registry syntax first and path syntax only for local examples.

## Non-Issues / Things Checked
- Provider clients call `request.validate()?` before generating, generating structured chat, or streaming.
- `cargo test`, `cargo test --all-features`, `cargo check --all-targets`, the feature matrix, examples, rustdoc, and formatting all passed locally.
- Streaming accumulators stop after `Finished` and have tests for streams that keep yielding after the terminal event.
- OpenAI streaming carries response metadata and usage across chunks and defers `Finished` to include trailing usage.
- Tool JSON malformation is preserved as provider metadata and `ToolRegistry` refuses to execute malformed tool inputs.
- Gemini provider metadata for function-call IDs and thought signatures is round-tripped into continuation requests.
- Feature flags compile for core-only, individual provider, all providers, streaming, and message-stream combinations.
- The publish script does meaningful preflight work: manifest checks, clean worktree enforcement, tests, example checks, rustdoc, package listing, dry-run publish, and tag collision checks.

## Assumptions
- I treated provider behavior using only repo source and tests because external provider/API docs were disallowed.
- I assumed `TextRequest` is intended to represent outbound provider requests, not partially accumulated in-progress assistant turns.
- I assumed the crate is intended to be publishable because package metadata, release-plz config, and publish automation are present.
- I did not inspect any existing files under `./audits/**` other than creating this assigned report path.

## Open Questions
- Are the hardcoded provider model enum variants intended to be authoritative "known model IDs" or only examples? If authoritative, they need periodic verification against provider docs, but external docs were out of scope for this review.
- Should `ToolChoice::Auto` with no tools be invalid for all providers? Current validation rejects it, but Gemini tests exercise direct translator behavior without validation.
- Should `SdkError::Unknown` be used for unknown runtime tools, or should that be a validation/application error for easier caller handling?

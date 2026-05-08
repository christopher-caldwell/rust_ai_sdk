# Final confirmed issues

Audited commit: fc82696c5c419e1020748ec3cd62b8cb2a3962ce
Date: 2026-05-08
Scope: Rust SDK core, providers, runtime, message-stream feature, examples where locally checkable.

## Executive summary

- Total unanimous confirmed issues: 3
- Highest severity: medium
- Commands that passed/failed: baseline `fmt`, `check` under default/all/no-default features, tests, doctests, clippy, and docs all exited successfully; `cargo doc` emitted 3 broken-link warnings; optional `cargo audit` and `cargo deny` were unavailable; temporary repro tests for F-001 and F-002 failed as expected.
- Important caveat: Findings below are limited to issues that all five audit passes verified.

## Issue list

### F-001: Runtime waits for stream EOF after terminal `Finished`

- Severity: medium
- Consensus: unanimous-confirmed
- Evidence: `run_turn` loops until stream exhaustion at `src/runtime/turn.rs:40-42` and only returns at `src/runtime/turn.rs:44`; `message-stream` similarly waits in `src/runtime/message_stream.rs:117-139` before it can emit `finish-step`, `finish`, and `[DONE]` at `src/runtime/message_stream.rs:150-158`. Temporary repros using `once(Ok(Finished)).chain(pending())` failed in `.audit/repro-run-turn-terminal-hang.txt`, `.audit/repro-message-stream-terminal-hang.txt`, and the self-check reruns.
- Why all audits agree: The core runtime, UI adapter, and reliability passes all verified that `StreamEvent::Finished` is accumulated but does not stop the consumer loop. Build/API and provider-mapping passes found no counter-contract requiring EOF after `Finished`.
- Impact: A chatbot request can hang after the model turn has already completed, leaving server tasks and browser clients waiting until an outer timeout cancels the request.
- Root cause: The runtime treats stream closure, not the terminal SDK event, as the model-turn boundary.
- Recommended fix: In `run_turn`, push `Finished` and immediately return `acc.into_outcome()`. In `stream_text_messages`, push `Finished`, skip direct event chunk emission as today, then break out of the per-step SDK stream loop so the existing finalization path emits terminal UI chunks.
- Regression test: Add timeout-based tests for `run_turn` and `stream_text_messages` with a stream built from `once(Ok(StreamEvent::Finished { .. })).chain(pending())`; both should complete and emit terminal results without waiting for EOF.
- Risk of fix: medium, because consumers might currently rely on post-`Finished` malformed events surfacing; terminal-event semantics make that reliance undesirable, but the behavior change is observable.

### F-002: OpenAI non-streaming chat can mark a tool-call turn as `Stop`

- Severity: medium
- Consensus: unanimous-confirmed
- Evidence: `src/providers/openai/types.rs:402-411` adds `MessagePart::ToolCall` values, but `src/providers/openai/types.rs:414-418` maps the provider `finish_reason` directly. The streaming path has a checked-in expectation that tool calls override provider `"stop"` at `src/providers/openai/client.rs:1048-1082`, and Gemini non-streaming does the same override at `src/providers/gemini/types.rs:538-548`. The temporary repro failed in `.audit/repro-openai-nonstream-tool-finish.txt` and the self-check rerun.
- Why all audits agree: Every pass rechecked the code path and found a real provider-neutral inconsistency: the same logical tool-call turn is `ToolUse` in OpenAI streaming and Gemini non-streaming, but can be `Stop` in OpenAI non-streaming.
- Impact: Chatbot code using `generate_chat` and branching on `finish_reason` can skip required tool execution even though the result contains tool calls.
- Root cause: The OpenAI non-streaming mapper does not override the mapped finish reason when tool calls are present.
- Recommended fix: After building `parts`, return `FinishReason::ToolUse` when any part is `MessagePart::ToolCall`; otherwise keep the existing provider finish reason mapping.
- Regression test: Construct a `ChatCompletionResponse` with one `tool_calls` entry and `finish_reason: Some("stop")`; assert `result.has_tool_calls()` and `matches!(result.finish_reason, FinishReason::ToolUse)`.
- Risk of fix: low, because it aligns non-streaming OpenAI behavior with existing streaming behavior and Gemini behavior.

### F-003: Public model enum docs contain broken rustdoc links

- Severity: low
- Consensus: unanimous-confirmed
- Evidence: `cargo doc --all-features --no-deps` emitted unresolved intra-doc link warnings for `src/providers/anthropic/models.rs:4`, `src/providers/gemini/models.rs:4`, and `src/providers/openai/models.rs:4`; the warning output is captured in `.audit/doc-all-features.txt` and `.audit/selfcheck-doc-all-features.txt`.
- Why all audits agree: The cited doc comments still contain unqualified links to sibling module types, and rustdoc consistently warns that those targets are not in scope.
- Impact: Published API docs direct users through broken links when they try to construct custom provider model IDs.
- Root cause: The docs link to `AnthropicChatModel::new`, `GeminiChatModel::new`, and `OpenAiChatModel::new` without paths resolvable from the enum modules.
- Recommended fix: Qualify the links with their module paths, for example `crate::providers::openai::model::OpenAiChatModel::new`, and do the equivalent for Anthropic and Gemini.
- Regression test: Add a docs CI step with `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps`.
- Risk of fix: low, documentation-only.

## Recommended fix order

1. F-001: Fix terminal-event handling in `run_turn` and `message-stream`.
2. F-002: Normalize OpenAI non-streaming tool-call finish reasons.
3. F-003: Fix rustdoc links and add warning-as-error docs coverage.

## Real but not unanimous / needs more verification

- None.

## Rejected false positives

- Feature combinations failing to compile: rejected by successful default/all/no-default `cargo check` runs.
- Unit tests, all-feature tests, doctests, or clippy failing: rejected by successful baseline runs.
- Unsupported UI-message parts being a bug: rejected because checked-in tests assert unsupported parts are ignored and no support contract was found.
- Tool execution lacking built-in authorization: rejected because README explicitly makes tools application-owned.
- Provider API keys leaking in SDK errors: rejected after inspecting error construction; no code path included auth headers in returned errors.
- Missing default HTTP timeout by itself: rejected under the audit rule requiring a concrete SDK-level failure mode.
- Dependency advisory findings: unproven because `cargo audit` and `cargo deny` were unavailable.

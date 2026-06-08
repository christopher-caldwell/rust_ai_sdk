# Consensus Issues

This list consolidates four independent graph-guided audits of the codebase.
Each audit was instructed to ignore `issues.md`; this file is based on the
agreements across those reports plus source verification against the current
tree.

## 1. Core message state is under-modeled

Severity: High
Consensus: 4/4 agents

Evidence:
- `src/core/message.rs:52` exposes `Message { role, content, parts }` as public, independently mutable fields.
- `src/core/message.rs:112` defines `effective_parts()` where structured `parts` supersede `content`.
- `src/providers/openai/types.rs:268`, `src/providers/anthropic/types.rs:195`, and `src/providers/gemini/types.rs:190` each rediscover the same message semantics while translating to provider payloads.

Problem:
The public type allows contradictory states: `content` and `parts` can both be set, a user message can contain assistant tool calls, an assistant message can contain tool results, and empty structured messages are valid. The intended invariant is documented in comments and adapter behavior, not expressed by the type system.

Why this matters:
This is the central SDK abstraction. A senior Rust reader should be able to inspect the type and know which states are legal. A newer Rust user will reasonably assume public fields are safe to construct directly.

Recommended direction:
Introduce a clearer content model, such as:

```rust
pub struct Message {
    role: Role,
    content: MessageContent,
}

pub enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}
```

Make constructors enforce valid role/content combinations. If preserving struct literals is important short term, add `TextRequest::validate()` and start deprecating direct field construction.

## 2. Tool result semantics are role-confusing and stringly typed

Severity: High
Consensus: 4/4 agents, either as its own finding or as part of the message-model issue

Evidence:
- `src/core/message.rs:3` defines `Role::Tool`.
- `src/core/message.rs:45` defines `ToolResult { tool_call_id, content: String }`.
- `src/core/message.rs:100` creates tool-result messages with `role: Role::User`.
- `src/runtime/tools.rs:55` returns tool outputs as `serde_json::Value`, then examples and runtime continuations serialize them with `output.to_string()`.
- `src/providers/gemini/types.rs:358` reparses tool-result strings back into JSON.

Problem:
The SDK has a `Role::Tool`, but tool results are represented as user-role messages with a structured part. Tool output also moves from `Value` to `String` and sometimes back to `Value`. That obscures whether a tool result is provider-neutral structured data, plain text, or a provider-specific wire workaround.

Recommended direction:
Make tool results first-class and structured:

- Either use `Role::Tool` consistently or remove it from the public enum if tool semantics live entirely in `MessagePart::ToolResult`.
- Change `ToolResult.content: String` to `ToolResult.output: Value` or a small `ToolOutput` enum.
- Centralize provider serialization of tool outputs so examples do not teach `output.to_string()` as the normal path.

## 3. Request construction has no validation boundary

Severity: High
Consensus: 3/4 agents

Evidence:
- `src/core/request.rs:4` exposes all `TextRequest` fields publicly.
- `src/core/request.rs:117` has `TextRequestBuilder::build() -> TextRequest`, so construction cannot fail.
- `examples/standalone/src/openai/tool_use.rs:29`, `examples/standalone/src/anthropic/tool_use.rs:30`, and `examples/standalone/src/gemini/tool_use.rs:30` mutate `request.max_output_tokens` after builder construction.
- The same examples later mutate `request.tool_choice` directly.

Problem:
Invalid or ambiguous requests are easy to build: empty message lists, `ToolChoice::Required` without a matching tool, out-of-range temperature, direct post-build mutation, or tool results that do not follow prior tool calls.

Recommended direction:
Add a validation boundary and call it before every provider translation:

- `TextRequest::validate() -> Result<(), SdkError>`
- Or `TextRequestBuilder::build() -> Result<TextRequest, SdkError>`

Update examples to use fluent construction or explicit continuation helpers instead of direct field mutation.

## 4. `stream_text_messages` combines too many responsibilities

Severity: High
Consensus: 3/4 agents

Evidence:
- `src/runtime/message_stream.rs:86` starts the public `stream_text_messages` helper.
- The same function handles model-step looping, SDK stream startup, event translation, turn accumulation, fallback tool IDs, tool execution, continuation building, max-step cutoff, and SSE framing.
- `src/runtime/turn.rs:32` and `src/runtime/turn.rs:127` already provide a separate turn runner and accumulator, so the message-stream adapter partially duplicates the turn lifecycle.

Problem:
The function is understandable only after mentally separating several phases inside an `async_stream::stream!` generator. It is valid Rust, but it is too dense for an SDK-facing orchestration path and makes the intended model/tool/model loop hard to inspect.

Recommended direction:
Split it into named pieces:

- A model-step runner that streams SDK events and returns an accumulated turn.
- An SSE encoder for `StreamEvent -> MessageStreamChunk`.
- A tool execution/continuation helper.
- A small public wrapper that wires the pieces together.

This should make the high-level loop readable without hiding the streaming details.

## 5. UI message stream ingestion silently drops non-text state

Severity: High
Consensus: 2/4 agents, with strong source evidence

Evidence:
- `src/runtime/message_stream.rs:45` defines only `MessageStreamPart::Text` plus `Other`.
- `src/runtime/message_stream.rs:197` converts inbound UI messages by collecting only text parts.
- Unknown roles fall through to user messages at `src/runtime/message_stream.rs:218`.

Problem:
The public API name suggests an AI SDK UI-message adapter, but non-text UI parts are ignored. Tool parts, files, reasoning parts, or future protocol fields can disappear without a typed error or visibility. Conversation state can be truncated in a way that is hard to debug.

Recommended direction:
Choose one explicit contract:

- Rename and document this as a text-only adapter, with strict-mode errors or ignored-part counts.
- Or parse the relevant UI message part variants into SDK `MessagePart` values.

Do not silently discard protocol state in a public adapter.

## 6. Stream event and tool-call lifecycle invariants are implicit

Severity: Medium-High
Consensus: 4/4 agents

Evidence:
- `src/core/stream.rs:10` defines the provider-neutral `StreamEvent` enum.
- Provider-specific stream accumulators live in `src/providers/openai/client.rs:222`, `src/providers/anthropic/client.rs:209`, and `src/providers/gemini/client.rs:195`.
- Runtime accumulation then reassembles neutral turns in `src/runtime/turn.rs:99`.

Problem:
Important invariants are not encoded or documented clearly:

- Can a tool call ID be empty?
- Is `ToolCallDelta.input_delta` partial JSON, a complete JSON object, or provider text?
- Must `ToolCallStarted` always precede `ToolCallDelta` and `ToolCallReady`?
- When is `Finished` authoritative?
- What happens when the stream ends without `Finished`?

The providers answer these questions differently in local state machines.

Recommended direction:
Add rustdoc to every `StreamEvent` variant documenting ordering and field invariants. Add a cross-provider stream contract test suite. Consider introducing small types like `ToolCallId`, `ToolInputDelta`, and `CompletedToolCall` if documentation alone is not enough.

## 7. Invalid provider tool JSON is silently converted into valid-looking data

Severity: Medium-High
Consensus: 3/4 agents

Evidence:
- `src/providers/openai/types.rs:424` falls back to `Value::String` for invalid non-streaming tool arguments.
- `src/providers/openai/client.rs:345` does the same for streamed tool arguments.
- `src/providers/anthropic/client.rs:340` falls back to `Value::String`.
- `src/runtime/turn.rs:180` also reparses accumulated deltas and falls back to `Value::String`.

Problem:
Malformed provider/model output becomes normal tool input. This hides provider contract drift and can route bad inputs into application tools as if they were intentional strings.

Recommended direction:
Introduce a shared parser such as:

```rust
enum ToolInput {
    Json(Value),
    RawMalformedJson(String),
}
```

Alternatively, return a structured SDK error when malformed JSON means continuation would be unsafe. The key is to preserve the distinction between valid JSON input and provider/model failure.

## 8. Provider-neutral policies drift across providers

Severity: Medium-High
Consensus: 3/4 agents

Evidence:
- Tool choice mapping differs in `src/providers/openai/types.rs:219`, `src/providers/anthropic/types.rs:233`, and `src/providers/gemini/types.rs:228`.
- Finish reason policy is scattered in `src/providers/openai/types.rs:434`, `src/providers/openai/client.rs:334`, `src/providers/anthropic/types.rs:371`, `src/providers/gemini/types.rs:538`, and `src/providers/gemini/client.rs:290`.
- Gemini schema conversion recursively removes `additionalProperties` in `src/providers/gemini/types.rs:408`.

Problem:
Provider-specific behavior is expected, but the neutral semantics are not stated in one place. `ToolChoice::None`, `FinishReason::ToolUse`, synthetic IDs, and schema adaptation all have provider-specific interpretations. Readers must compare adapters to infer the SDK contract.

Recommended direction:
Create explicit policy helpers and table tests:

- `tool_choice_policy(provider, request_tools, choice)`
- `finish_reason_from_provider_and_parts(provider_reason, has_tool_calls)`
- `provider_schema_for_tool(provider, schema)`

Keep provider quirks local, but make the neutral contract visible and tested.

## 9. Provider metadata leaks as untyped JSON through the core API

Severity: Medium
Consensus: 3/4 agents

Evidence:
- `src/core/message.rs:21` exposes `ToolCall.provider_metadata: Option<Value>`.
- Gemini uses hidden JSON keys for `thoughtSignature` and function-call IDs in `src/providers/gemini/types.rs:15`, `src/providers/gemini/types.rs:374`, and `src/providers/gemini/types.rs:380`.

Problem:
Gemini metadata preservation appears necessary, but the public core type now exposes arbitrary provider JSON. Users can see and mutate metadata without knowing which keys matter. That weakens the provider-neutral surface and makes continuations feel magical.

Recommended direction:
Wrap metadata in a typed or semi-opaque abstraction:

- `ProviderMetadata` with provider-specific variants.
- Or a private map plus provider helper constructors/accessors.

Keep an escape hatch if needed, but do not make raw `Value` the normal public concept.

## 10. `SdkError` is too stringly for an SDK boundary

Severity: Medium-High
Consensus: 1/4 agents, but strong source evidence and high SDK impact

Evidence:
- `src/core/error.rs:4` has only `Http(String)`, `Api(String)`, `Serialization(String)`, and `Unknown(String)`.
- Provider clients collapse HTTP status, provider name, provider error type/code, retryability, and body snippets into formatted strings in `src/providers/openai/client.rs:75`, `src/providers/anthropic/client.rs:75`, and `src/providers/gemini/client.rs:186`.

Problem:
Downstream applications cannot reliably branch on status code, provider, retryability, or provider error category. The error display is readable, but the structured context is lost.

Recommended direction:
Add structured variants while preserving friendly `Display`, for example:

```rust
pub enum SdkError {
    Http { provider: &'static str, status: StatusCode, body_snippet: String },
    Api { provider: &'static str, code: Option<String>, message: String, status: Option<StatusCode> },
    Serialization { provider: Option<&'static str>, source: String },
    Unknown(String),
}
```

The exact shape can be smaller, but it should preserve machine-readable context.

## 11. Provider clients and stream translators repeat control flow

Severity: Medium
Consensus: 3/4 agents

Evidence:
- OpenAI repeats send/status/body/parse flow in `src/providers/openai/client.rs:49`, `src/providers/openai/client.rs:92`, and `src/providers/openai/client.rs:135`.
- Anthropic repeats the same pattern in `src/providers/anthropic/client.rs:51`, `src/providers/anthropic/client.rs:92`, and `src/providers/anthropic/client.rs:133`.
- Gemini has a local `response_bytes` helper at `src/providers/gemini/client.rs:160`, but the pattern is not shared even within all providers.

Problem:
The duplication is not only a performance or maintenance issue; it obscures intent. The reader has to scan transport boilerplate before getting to provider translation logic. Stream accumulators also use different shapes and composition styles, making cross-provider behavior hard to compare.

Recommended direction:
Extract small helpers, not a large abstraction:

- `post_json`
- `decode_json_response`
- `provider_error_from_bytes`
- Provider-local `streaming.rs` modules with consistent helper names

Preserve provider-specific headers and payloads explicitly.

## 12. Feature design is too coarse for a provider-neutral SDK

Severity: Medium-High
Consensus: 1/4 agents, but directly relevant to SDK clarity

Evidence:
- `Cargo.toml:16` makes `reqwest`, `eventsource-stream`, `tokio`, and every provider compile for all consumers.
- Only `message-stream` is feature-gated at `Cargo.toml:32`.
- `src/providers/mod.rs:1` exposes all provider modules unconditionally.

Problem:
For a provider-neutral SDK, users cannot opt into only core types or only one provider. This makes the dependency story and conceptual boundaries less clear.

Recommended direction:
Introduce feature groups:

- `openai`
- `anthropic`
- `gemini`
- `providers-all`
- `streaming`
- `message-stream`

Document the feature matrix in the README and ensure default features match the intended crate story.

## 13. Examples and public contracts are under-tested from the root project

Severity: High
Consensus: 1/4 agents, but source evidence is clear

Evidence:
- `Cargo.toml:11` excludes `examples/**` from the package.
- `cargo metadata --no-deps --format-version 1` reports only the root crate as a workspace member.
- `Justfile:20` runs only `cargo test`.
- Publish preflight runs `cargo test --all-features` at `scripts/publish-crate.sh:128`, but that is late and still does not compile every example crate or frontend.

Problem:
The README teaches primarily through examples, but examples are not validated by the normal root test path. API drift can break examples without failing the default checks.

Recommended direction:
Either make examples workspace members or add explicit checks:

- `cargo check --manifest-path examples/standalone/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server/Cargo.toml`
- `cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml`
- `npm run build` in `examples/chatbot/web`
- `cargo test --all-features` in `just test`

## 14. Examples teach awkward ownership and mutation patterns

Severity: Medium
Consensus: 2/4 agents

Evidence:
- `examples/standalone/src/openai/tool_use.rs:29`, `examples/standalone/src/anthropic/tool_use.rs:30`, and `examples/standalone/src/gemini/tool_use.rs:30` mutate `request.max_output_tokens`.
- The same files reset `request.tool_choice` directly later in the loop.
- `examples/standalone/src/openai/tool_stream.rs:77` manually implements streamed turn accumulation that overlaps with library runtime concepts.

Problem:
The examples are likely the first place a new user learns the SDK. They currently demonstrate direct public-field mutation, repeated clones of full requests in loops, duplicated weather tools, and manual low-level streaming logic where a canonical high-level path should be shown first.

Recommended direction:
Make examples model the style expected from application users:

- Use `TextRequest::builder().max_output_tokens(...)`.
- Prefer `ToolRegistry` plus `run_turn` for the canonical tool loop.
- Add request-update helpers instead of direct mutation.
- Keep low-level streaming accumulation only in an explicitly advanced event-inspection example.

## 15. Public docs and provider exports are too inconsistent for an SDK

Severity: Medium
Consensus: 3/4 agents for exports, 1/4 for docs

Evidence:
- `src/providers/openai/mod.rs:1` exposes `pub mod model` and only re-exports `OpenAiModel`.
- `src/providers/anthropic/mod.rs:7` re-exports `AnthropicChatModel`.
- `src/providers/gemini/mod.rs:7` re-exports `GeminiChatModel`.
- `README.md:87` imports OpenAI through `providers::openai::model::OpenAiChatModel`.
- `src/lib.rs:1`, `src/core/model.rs:11`, `src/core/stream.rs:10`, and `src/core/error.rs:3` have thin or missing public docs for key concepts.

Problem:
SDK users should learn one provider import shape and repeat it. The current public surface is slightly inconsistent, and docs.rs would not explain the core mental model deeply enough without reading examples.

Recommended direction:
Align provider exports:

```rust
pub use model::OpenAiChatModel;
pub use models::OpenAiModel;
```

Then update docs and examples to use the same provider shape for all providers. Add crate-level `//!` docs and rustdoc for `LanguageModel`, `StreamEvent`, `SdkError`, `TextRequest`, provider wrappers, and feature-gated message-stream APIs.


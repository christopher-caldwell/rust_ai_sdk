# Plan: Tool-Aware SDK Evolution for rust_ai_sdk

## Context

The SDK currently supports text-only `generate` and `stream` against OpenAI and Anthropic. The goal is to evolve it into a provider-neutral, tool-aware chat SDK so a Rust chat server can act as a streaming orchestrator: receive frontend messages, call the model, detect tool requests, execute tools in application code, and continue the loop — the same role `Next.js` + Vercel AI SDK plays, but in Rust.

The biggest missing capability is MCP/tool support: structured message parts, tool definitions on requests, tool-call stream events, a richer result type, and a runtime turn helper.

---

## Design Principles

- Additive wherever possible: `TextRequest` and `TextResult` keep working unchanged.
- `LanguageModel` trait gains no new methods — runtime helpers drive the richer flow.
- Provider translators normalize tool semantics; the core stays neutral.
- Application owns tool execution; SDK only models the contract.

---

## Stage 1 — Core Type Foundation

**Files to create:**

- `src/core/tool.rs` — `ToolDefinition`, `ToolChoice`

**Files to modify:**

### `src/core/message.rs`

- Add `Role::Tool`
- Add `MessagePart` enum: `Text(String)`, `ToolCall(ToolCall)`, `ToolResult(ToolResult)`
- Add `ToolCall { id: String, name: String, input: serde_json::Value }`
- Add `ToolResult { tool_call_id: String, content: String }`
- Add `parts: Vec<MessagePart>` field to `Message` (keep `content: String` for compat)
- Add constructors: `Message::user()`, `Message::assistant()`, `Message::system()`, `Message::assistant_parts()`, `Message::tool_result()`

**Key invariant**: Provider translators use a helper `fn effective_parts(msg)` — returns `msg.parts` if non-empty, else `vec![MessagePart::Text(msg.content.clone())]`. Old-style struct literals and new-style constructors behave identically at the wire level.

### `src/core/request.rs`

- Add `tools: Vec<ToolDefinition>` and `tool_choice: Option<ToolChoice>` to `TextRequest`
- Update `TextRequest::prompt()` to use `Message::user()`
- Add builder methods `.with_tools()`, `.with_tool_choice()`

### `src/core/stream.rs`

- Add `StreamEvent` variants:
    ```rust
    ToolCallStarted { id: String, name: String, index: u32 }
    ToolCallDelta { id: String, index: u32, input_delta: String }
    ToolCallReady { id: String, name: String, index: u32, input: serde_json::Value }
    ```
- Add `pub type EventStream = TextEventStream;` alias

### `src/core/result.rs`

- Add `ChatResult { parts: Vec<MessagePart>, finish_reason, usage, response }`
- Add `ChatResult::text()`, `ChatResult::tool_calls()`, `ChatResult::has_tool_calls()` helpers

### `src/core/types.rs`

- Add `FinishReason::ToolUse` variant

### `src/core/mod.rs`

- Add `pub mod tool;`

**Breakage to fix in existing tests** (mechanical):

- `Message { role: ..., content: ... }` literals → add `parts: vec![]`
- `TextRequest { ... }` literals → add `tools: vec![], tool_choice: None`
- `match StreamEvent` arms → add `_ => {}` for new variants

---

## Stage 2 — Runtime Turn Helper

**File to create:** `src/runtime/turn.rs`

```rust
pub enum TurnOutcome {
    Completed(ChatResult),
    ToolsRequired {
        tool_calls: Vec<ToolCall>,
        assistant_parts: Vec<MessagePart>,
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    },
}

pub async fn run_turn<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TurnOutcome, SdkError>

pub struct ContinuationBuilder {
    fn from_request(request: TextRequest) -> Self
    fn with_assistant_turn(mut self, parts: Vec<MessagePart>) -> Self
    fn with_tool_result(mut self, id: &str, content: &str) -> Self
    fn with_tool_results(mut self, results: impl IntoIterator<Item=(String,String)>) -> Self
    fn build(self) -> TextRequest
}
```

Internal `TurnAccumulator`: buffers `ToolCallBuffer { id, name, arguments: String, index }` keyed by `index`. `into_outcome()` returns `ToolsRequired` if any tool calls present.

**Tests (synthetic streams, no HTTP):**

- `test_turn_text_only` — asserts `Completed` with correct text
- `test_turn_single_tool_call` — asserts `ToolsRequired` with correct id/name/input
- `test_turn_parallel_tool_calls` — two tool calls at index 0 and 1
- `test_continuation_builder_message_order` — original + assistant_parts + tool_result appended correctly

**Modify:** `src/runtime/mod.rs` — add `pub mod turn;`

---

## Stage 3 — OpenAI Tool Translation

### `src/providers/openai/types.rs`

New serde types:

```rust
OaiTool { tool_type: "function", function: OaiFunctionDef { name, description, parameters: Value } }
OaiToolChoice { None | Auto | Function { type: "function", function: { name } } }
OaiChunkToolCallDelta { index: u32, id: Option<String>, call_type: Option<String>, function: Option<OaiFunctionDelta> }
OaiFunctionDelta { name: Option<String>, arguments: Option<String> }
```

Changes to existing types:

- `ChatCompletionRequest`: add `tools: Vec<OaiTool>`, `tool_choice: Option<OaiToolChoice>`
- `AssistantMessage`: add `tool_calls: Option<Vec<OaiToolCall>>`
- `ChunkDelta`: add `tool_calls: Option<Vec<OaiChunkToolCallDelta>>`
- `ChatMessage`: extend role/content to support `role: "tool"` with `tool_call_id`

New functions:

- `text_request_to_openai()` — updated to map tools
- `message_to_openai_message()` — handles all `MessagePart` variants
- `chat_response_to_chat_result()` — returns `ChatResult` (alongside existing `chat_response_to_text_result`)

**New tests:**

- `test_tool_def_serialization` — maps to correct `{type: "function", function: {...}}` shape
- `test_tool_choice_variants` — auto/none/required serialize correctly
- `test_message_tool_call_part_to_openai` — assistant message with ToolCall part
- `test_message_tool_result_part_to_openai` — role: "tool" with tool_call_id
- `test_chat_response_to_chat_result_with_tool_calls` — response with tool_calls → ChatResult

### `src/providers/openai/client.rs`

Add to `StreamAccumulator`:

```rust
tool_call_buffers: HashMap<u32, ToolCallBuffer>
```

In `process_chunk()`:

1. When `delta.tool_calls` contains entry with `id: Some(...)`: insert buffer, emit `ToolCallStarted`
2. On each delta with arguments: append to buffer, emit `ToolCallDelta`
3. On `finish_reason = "tool_calls"`: for each buffer, parse JSON, emit `ToolCallReady`, then emit `Finished`

**New tests:**

- `test_stream_single_tool_call` — SSE fixture with tool call deltas → assert event sequence
- `test_stream_parallel_tool_calls` — two tool calls, different indexes
- `test_stream_tool_call_with_text_prefix` — text deltas then tool call

---

## Stage 4 — Anthropic Tool Translation

### `src/providers/anthropic/types.rs`

New serde types:

```rust
AnthropicTool { name, description, input_schema: Value }
AnthropicToolChoice (tag = "type"): Auto | Any | Tool { name }
ContentBlockStartEvent { index: u32, content_block: ContentBlockStart }
ContentBlockStart: Text { text } | ToolUse { id, name }
AnthropicContentPart (tag = "type"): Text { text } | ToolResult { tool_use_id, content }
AnthropicMessageContent (untagged): Text(String) | Parts(Vec<AnthropicContentPart>)
```

Changes:

- `AnthropicRequest`: add `tools`, `tool_choice`
- `AnthropicMessage.content`: change from `String` to `AnthropicMessageContent`
- `ContentDelta`: add `partial_json: Option<String>` for tool input streaming
- `AnthropicContentBlock`: add `id, name, input` fields for tool_use blocks

Update `text_request_to_anthropic()` to map tools and handle message parts.  
Add `anthropic_response_to_chat_result()`.

**New tests:**

- `test_tool_def_serialization` — `{ name, description, input_schema }` (no "function" wrapper)
- `test_message_tool_result_to_anthropic` — `role: "user", content: [{type: "tool_result", ...}]`
- `test_message_tool_call_to_anthropic` — assistant message with tool_use content block

### `src/providers/anthropic/client.rs`

Add `tool_call_buffers: HashMap<u32, ToolCallBuffer>` to accumulator.

Update `process_event()`:

- `"content_block_start"`: deserialize; if ToolUse → emit `ToolCallStarted`
- `"content_block_delta"` with `partial_json`: emit `ToolCallDelta`
- `"content_block_stop"`: if tool block → parse JSON, emit `ToolCallReady`
- `"message_delta"` with `stop_reason: "tool_use"`: sets `FinishReason::ToolUse`

**New tests:**

- `test_process_event_content_block_start_tool_use`
- `test_process_event_input_json_delta`
- `test_process_event_content_block_stop_tool_use`
- `test_stream_tool_use_single_end_to_end`
- `test_stream_mixed_text_and_tool`

---

## Stage 5 — Examples

**Files to create:**

- `examples/src/openai/tool_use.rs` — weather stub tool, `run_turn()` loop
- `examples/src/anthropic/tool_use.rs` — same pattern for Anthropic

**Pattern:**

```rust
let request = TextRequest::prompt("What is the weather in Paris?")
    .with_tools(vec![weather_tool_def()]);

loop {
    match run_turn(&model, request.clone()).await? {
        TurnOutcome::Completed(result) => { println!("{}", result.text()); break; }
        TurnOutcome::ToolsRequired { tool_calls, assistant_parts, .. } => {
            let mut builder = ContinuationBuilder::from_request(request.clone())
                .with_assistant_turn(assistant_parts);
            for call in &tool_calls {
                builder = builder.with_tool_result(&call.id, execute_tool(&call.name, &call.input));
            }
            request = builder.build();
        }
    }
}
```

**Modify:** `examples/Cargo.toml` — add `[[bin]]` for `openai-tool-use` and `anthropic-tool-use`

---

## Verification

1. `cargo test` in root — all existing + new tests pass
2. `cargo build --examples` in `examples/` — all examples compile
3. `OPENAI_API_KEY=... cargo run --example openai-tool-use` — executes tool loop, prints final answer
4. `ANTHROPIC_API_KEY=... cargo run --example anthropic-tool-use` — same for Anthropic

---

## File Index

| Action | Path                                 |
| ------ | ------------------------------------ |
| CREATE | `src/core/tool.rs`                   |
| CREATE | `src/runtime/turn.rs`                |
| CREATE | `examples/src/openai/tool_use.rs`    |
| CREATE | `examples/src/anthropic/tool_use.rs` |
| MODIFY | `src/core/message.rs`                |
| MODIFY | `src/core/request.rs`                |
| MODIFY | `src/core/stream.rs`                 |
| MODIFY | `src/core/result.rs`                 |
| MODIFY | `src/core/types.rs`                  |
| MODIFY | `src/core/mod.rs`                    |
| MODIFY | `src/runtime/mod.rs`                 |
| MODIFY | `src/providers/openai/types.rs`      |
| MODIFY | `src/providers/openai/client.rs`     |
| MODIFY | `src/providers/anthropic/types.rs`   |
| MODIFY | `src/providers/anthropic/client.rs`  |
| MODIFY | `examples/Cargo.toml`                |

ALL STAGES COMPLETE

# Audit 2: provider request/response mapping

## Commands/evidence collected

- Consulted baseline outputs: `.audit/check-default.txt`, `.audit/check-all-features.txt`, `.audit/test-default.txt`, `.audit/test-all-features.txt`.
- Source inspected with `nl -ba`: `src/providers/openai/types.rs`, `src/providers/openai/client.rs`, `src/providers/anthropic/types.rs`, `src/providers/anthropic/client.rs`, `src/providers/gemini/types.rs`, and `src/providers/gemini/client.rs`.
- Temporary repro command: `cargo test providers::openai::types::tests::audit_repro_tool_calls_force_tool_use_even_when_finish_reason_is_stop 2>&1 | tee .audit/repro-openai-nonstream-tool-finish.txt`.
- Temporary repro test was added to `src/providers/openai/types.rs` during this loop and removed immediately after the command.

## Confirmed issues from this loop

### A2-1: OpenAI non-streaming chat can mark a tool-call turn as `Stop`

- Status: confirmed
- Severity: medium
- Evidence type: minimal-repro
- Files/lines: `src/providers/openai/types.rs:402-418`, `src/providers/openai/client.rs:1048-1082`, `src/providers/gemini/types.rs:538-548`
- Trigger: An OpenAI non-streaming chat response contains `message.tool_calls` but the choice `finish_reason` is `"stop"`.
- Observed behavior: `chat_response_to_chat_result` adds `MessagePart::ToolCall` values at `src/providers/openai/types.rs:402-411`, then maps the provider finish reason directly at `src/providers/openai/types.rs:414-418`, so the result is `ChatResult { parts: [ToolCall(..)], finish_reason: FinishReason::Stop, .. }`. The temporary repro in `.audit/repro-openai-nonstream-tool-finish.txt` failed on `assert!(matches!(result.finish_reason, FinishReason::ToolUse))`.
- Expected behavior: A provider-neutral `ChatResult` containing tool calls should be classified as `FinishReason::ToolUse`, matching the checked-in OpenAI streaming behavior in `test_stream_tool_call_ready_when_provider_finish_is_stop` (`src/providers/openai/client.rs:1048-1082`) and the Gemini non-streaming mapping that forces `ToolUse` when tool calls are present (`src/providers/gemini/types.rs:538-548`).
- User impact: Chatbot code using `generate_chat` and branching on `finish_reason` can skip tool execution even though the result contains tool calls. The same provider event shape handled through `stream`/`run_turn` would request tools, so non-streaming and streaming behavior diverge.
- Root cause: The OpenAI non-streaming mapper derives `finish_reason` only from the provider string and does not apply the tool-call presence override used by other paths.
- Minimal fix: In `chat_response_to_chat_result`, compute whether `parts` contains a `MessagePart::ToolCall` and return `FinishReason::ToolUse` when true; otherwise preserve the mapped provider finish reason.
- Test to add: Add the temporary repro as a permanent unit test: construct a `ChatCompletionResponse` with one `tool_calls` entry and `finish_reason: Some("stop")`, then assert `result.has_tool_calls()` and `matches!(result.finish_reason, FinishReason::ToolUse)`.
- What would prove this false: A rerun of the repro test passing at the audited commit without code changes, or a checked-in contract stating that OpenAI non-streaming `generate_chat` intentionally preserves `"stop"` even when tool calls are present while streaming intentionally overrides it.

## Unproven / rejected candidates

### Candidate: Provider error responses leak API keys

- Why rejected/unproven: OpenAI, Anthropic, and Gemini clients read provider error bodies and return provider messages/status snippets, but inspected request construction does not include API keys in request bodies. Header values are not formatted into SDK error messages on provider HTTP status failures.

### Candidate: Gemini loses provider metadata needed for tool continuation

- Why rejected/unproven: Non-streaming Gemini maps function-call ID/name/thought signature into `provider_metadata` at `src/providers/gemini/types.rs:525-533`, and request mapping uses that metadata for the subsequent function call and response at `src/providers/gemini/types.rs:345-377`. Streaming also attaches Gemini metadata in `src/providers/gemini/client.rs:274-284`.

### Candidate: Anthropic tool result continuation is obviously malformed

- Why rejected/unproven: The local mapping and tests show `Message::tool_result` becomes a user-role `tool_result` content block with `tool_use_id` at `src/providers/anthropic/types.rs:286-289` and `src/providers/anthropic/types.rs:463-478`. No provider-doc-backed contradiction was established in this loop.

### Candidate: OpenAI streaming emits terminal `Finished` before usage can arrive

- Why rejected/unproven: `src/providers/openai/client.rs:301-311` flushes a pending finish after updating usage from the next chunk, and checked-in tests cover trailing usage behavior. No local failing sequence was found for normal usage chunks.

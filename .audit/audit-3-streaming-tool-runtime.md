# Audit 3: streaming and tool runtime

## Commands/evidence collected

- Consulted baseline outputs: `.audit/test-default.txt`, `.audit/test-all-features.txt`, `.audit/check-all-features.txt`.
- Source inspected with `nl -ba`: `src/runtime/turn.rs`, `src/runtime/tools.rs`, `src/core/stream.rs`, `src/providers/openai/client.rs`, `src/providers/anthropic/client.rs`, and `src/providers/gemini/client.rs`.
- Temporary repro command: `cargo test runtime::turn::tests::audit_repro_run_turn_returns_after_finished_even_if_stream_stays_open 2>&1 | tee .audit/repro-run-turn-terminal-hang.txt`.
- Temporary repro test was added to `src/runtime/turn.rs` during this loop and removed immediately after the command.

## Confirmed issues from this loop

### A3-1: `run_turn` waits for stream closure after a terminal `Finished` event

- Status: confirmed
- Severity: medium
- Evidence type: minimal-repro
- Files/lines: `src/runtime/turn.rs:33-44`, `src/core/stream.rs:13-17`
- Trigger: A `LanguageModel::stream` implementation yields `StreamEvent::Finished` and then keeps the stream open, for example because the underlying SSE connection stays alive or a custom model stream does not close immediately.
- Observed behavior: `run_turn` loops until `stream.next().await` returns `None` at `src/runtime/turn.rs:40-42`; it pushes `Finished` into the accumulator but does not stop. The temporary repro in `.audit/repro-run-turn-terminal-hang.txt` yielded `Finished`, then `pending()`, and `tokio::time::timeout(..., run_turn(...))` timed out and failed the assertion.
- Expected behavior: `StreamEvent::Finished` is the terminal event shape (`src/core/stream.rs:13-17`); `run_turn` should return the accumulated outcome once that terminal event is observed instead of depending on transport closure.
- User impact: A chatbot using `run_turn` can hang a request after the model has already emitted a complete terminal event, blocking the tool/model loop and tying up server work until an external timeout cancels it.
- Root cause: `run_turn` treats stream exhaustion, not `StreamEvent::Finished`, as the end-of-turn boundary.
- Minimal fix: In `run_turn`, detect `StreamEvent::Finished`, push it into the accumulator, and immediately return `acc.into_outcome()`; keep existing error propagation for errors before terminal.
- Test to add: Add the temporary repro as a permanent unit test with a mock `LanguageModel` whose stream is `once(Ok(Finished { .. })).chain(pending())`; wrap `run_turn` in a short timeout and assert it completes successfully.
- What would prove this false: A checked-in contract stating `StreamEvent::Finished` is informational and streams must close to terminate, or the temporary repro passing at the audited commit without code changes.

## Unproven / rejected candidates

### Candidate: `TurnAccumulator` drops parallel tool calls in normal index order

- Why rejected/unproven: `parts_order` tracks first occurrence of each tool index and checked-in tests cover parallel tool calls. No failing event sequence was found where distinct indexes are dropped.

### Candidate: Tool argument JSON deltas are always malformed on partial chunks

- Why rejected/unproven: The accumulator appends deltas and parses the final concatenated string in `into_accumulated`; checked-in tests cover partial override and complete JSON assembly. Silent fallback to `Value::String` for malformed final JSON may be a policy choice absent a stricter contract.

### Candidate: `ContinuationBuilder` reorders assistant and tool messages

- Why rejected/unproven: The builder appends the assistant turn before tool results, and `test_continuation_builder_message_order` covers that order.

### Candidate: `ToolRegistry` executes unknown tools silently

- Why rejected/unproven: `ToolRegistry::execute` returns `SdkError::Unknown` for missing names at `src/runtime/tools.rs:55-58`, with a checked-in unit test.

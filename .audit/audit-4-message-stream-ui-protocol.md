# Audit 4: message-stream UI protocol adapter

## Commands/evidence collected

- Consulted baseline outputs: `.audit/check-all-features.txt`, `.audit/test-all-features.txt`, `.audit/clippy-all-features.txt`.
- Source inspected with `nl -ba`: `src/runtime/message_stream.rs`, `src/runtime/turn.rs`, and `src/core/stream.rs`.
- Temporary repro command: `cargo test --all-features runtime::message_stream::tests::audit_repro_message_stream_finishes_after_sdk_finished_even_if_stream_stays_open 2>&1 | tee .audit/repro-message-stream-terminal-hang.txt`.
- Temporary repro test was added to `src/runtime/message_stream.rs` during this loop and removed immediately after the command.

## Confirmed issues from this loop

### A4-1: `message-stream` can withhold `finish-step`, `finish`, and `[DONE]` after SDK `Finished`

- Status: confirmed
- Severity: medium
- Evidence type: minimal-repro
- Files/lines: `src/runtime/message_stream.rs:117-158`, `src/core/stream.rs:13-17`
- Trigger: `stream_text_messages` receives an SDK stream that yields `StreamEvent::Finished` and then remains open.
- Observed behavior: The adapter pushes the terminal event into `TurnAccumulator` at `src/runtime/message_stream.rs:134`, emits no chunk for `StreamEvent::Finished` through `chunks_for_sdk_event`, and stays inside `while let Some(event) = sdk_stream.next().await` at `src/runtime/message_stream.rs:117-139`. It cannot reach the `finish-step`, `finish`, or `[DONE]` emission at `src/runtime/message_stream.rs:150-158`. The temporary repro in `.audit/repro-message-stream-terminal-hang.txt` timed out waiting for the next UI stream chunk after `start` and `start-step`.
- Expected behavior: Once the terminal SDK `Finished` event is observed, the adapter should finish the current step and emit the UI-message terminal chunks without requiring the underlying SDK stream to close.
- User impact: Browser clients consuming the Vercel-style UI-message SSE protocol can remain in a loading state despite the model turn being terminal, because the adapter never sends `finish-step`, `finish`, or `[DONE]`.
- Root cause: `stream_text_messages` treats SDK stream exhaustion, not `StreamEvent::Finished`, as the step boundary.
- Minimal fix: In the SDK-event loop, detect `StreamEvent::Finished`, push it to the accumulator, emit no direct event chunk as today, then break out of the per-step stream loop so the existing post-loop finalization path runs.
- Test to add: Add the temporary repro as a permanent `message-stream` unit test: use a mock `LanguageModel` returning `once(Ok(Finished { .. })).chain(pending())`, consume `start` and `start-step`, then assert the next chunk arrives within a timeout and contains `finish-step`.
- What would prove this false: The temporary repro passing at the audited commit without code changes, or a checked-in adapter contract stating that `[DONE]` must wait for transport closure even after SDK `Finished`.

## Unproven / rejected candidates

### Candidate: Unsupported UI-message parts are a confirmed bug

- Why rejected/unproven: `MessageStreamPart::Other` is explicitly ignored at `src/runtime/message_stream.rs:202-205`, and checked-in tests assert unsupported parts are ignored. No README or public API contract claiming support for those parts was found.

### Candidate: Empty UI messages always corrupt conversation history

- Why rejected/unproven: Empty text parts are skipped at `src/runtime/message_stream.rs:209-210`, with a checked-in test. This may be lossy for some clients but no local contract says whitespace-only messages must be preserved.

### Candidate: Tool execution errors produce invalid JSON chunks

- Why rejected/unproven: Tool errors are converted into a JSON object and serialized through `serde_json::json!` at `src/runtime/message_stream.rs:164-174`, so chunk JSON remains valid.

### Candidate: Error chunks omit terminal UI-message markers

- Why rejected/unproven: `error_chunks` emits `error`, `finish-step`, `finish`, and `[DONE]` at `src/runtime/message_stream.rs:227-233`.

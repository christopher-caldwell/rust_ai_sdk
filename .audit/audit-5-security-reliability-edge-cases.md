# Audit 5: security, reliability, and production edge cases

## Commands/evidence collected

- Consulted baseline outputs: `.audit/test-default.txt`, `.audit/test-all-features.txt`, `.audit/cargo-audit.txt`, `.audit/cargo-deny.txt`.
- Source inspected with `nl -ba` and `rg`: `src/runtime/turn.rs`, `src/runtime/message_stream.rs`, `src/runtime/tools.rs`, provider clients, provider error helpers, and README tool-ownership text.
- Relevant repro output consulted: `.audit/repro-run-turn-terminal-hang.txt` and `.audit/repro-message-stream-terminal-hang.txt`.

## Confirmed issues from this loop

### A5-1: Terminal SDK events can still leave production requests hanging

- Status: confirmed
- Severity: medium
- Evidence type: minimal-repro
- Files/lines: `src/runtime/turn.rs:33-44`, `src/runtime/message_stream.rs:117-158`
- Trigger: A model stream yields `StreamEvent::Finished` but does not close immediately.
- Observed behavior: Both `run_turn` and `stream_text_messages` keep awaiting the next stream item after the terminal event. The local repro outputs show timeouts for the core runtime path (`.audit/repro-run-turn-terminal-hang.txt`) and the message-stream adapter path (`.audit/repro-message-stream-terminal-hang.txt`).
- Expected behavior: A terminal SDK event should end the current model turn and let the caller/adaptor return terminal output without waiting for transport closure.
- User impact: In a live chatbot, a request can remain open after the model turn is complete, causing stuck browser loading states, occupied server tasks, and reliance on outer HTTP/server timeouts to recover.
- Root cause: Runtime loops use stream exhaustion as the completion condition and do not break when the provider-neutral terminal event is observed.
- Minimal fix: Break the relevant loop immediately after pushing a `StreamEvent::Finished`; in `run_turn`, return `acc.into_outcome()`, and in `message-stream`, proceed to the existing post-loop UI finalization path.
- Test to add: Permanent timeout-based tests for `run_turn` and `stream_text_messages` using `once(Ok(Finished { .. })).chain(pending())`.
- What would prove this false: The two repro commands passing at the audited commit, or an explicit checked-in contract that `Finished` is non-terminal and consumers must wait for stream EOF.

## Unproven / rejected candidates

### Candidate: Provider API keys are leaked in SDK error messages

- Why rejected/unproven: Request headers contain API keys, but inspected error construction formats provider error messages/status snippets and does not include header values. No failing local test or code path showed secrets included in returned `SdkError` values.

### Candidate: Missing default HTTP timeout is a standalone SDK bug

- Why rejected/unproven: Provider clients use `reqwest::Client::new()` and do not set a default timeout, but the audit rule says not to report "no timeout" by itself. The terminal-event hang above is the concrete SDK-level wait bug with proof.

### Candidate: Tool execution lacks built-in authorization

- Why rejected/unproven: README explicitly states tools are application-owned and the server decides authorization/side effects. This is a documented ownership boundary, not a contradiction.

### Candidate: Large provider error bodies are a confirmed memory-exhaustion bug

- Why rejected/unproven: Provider clients call `.bytes()` before truncating error text, so very large provider error bodies could increase memory use. However, public clients do not expose arbitrary base URLs, normal official provider error bodies are bounded in practice, and no local practical exhaustion repro was established under production provider assumptions.

### Candidate: Optional dependency tooling found known vulnerabilities

- Why rejected/unproven: `cargo audit` and `cargo deny` were unavailable in this environment, so no advisory-backed dependency finding was produced.

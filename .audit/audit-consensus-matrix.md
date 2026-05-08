# Consensus matrix

| Candidate ID | Root cause | Loop 1 | Loop 2 | Loop 3 | Loop 4 | Loop 5 | Hard evidence present? | Final status |
|---|---|---:|---:|---:|---:|---:|---|---|
| C-001 | Public provider model enum docs link to unqualified sibling chat model constructors, producing rustdoc broken-link warnings. | discovered | verified | verified | verified | verified | yes | unanimous-confirmed |
| C-002 | OpenAI non-streaming `chat_response_to_chat_result` maps provider `finish_reason` directly even when tool calls are present. | verified | discovered | verified | verified | verified | yes | unanimous-confirmed |
| C-003 | Runtime stream consumers use stream EOF, not `StreamEvent::Finished`, as the turn/step completion boundary. | verified | verified | discovered | discovered | verified | yes | unanimous-confirmed |
| C-004 | Unsupported UI-message parts are ignored by `message-stream`. | not-applicable | not-applicable | not-applicable | rejected | rejected | no | rejected |
| C-005 | Provider API keys are leaked in SDK error messages. | rejected | rejected | not-applicable | not-applicable | rejected | no | rejected |
| C-006 | Lack of default HTTP client timeout is a standalone bug. | not-applicable | rejected | rejected | not-applicable | rejected | no | rejected |
| C-007 | Large provider error bodies prove practical memory exhaustion. | not-applicable | rejected | not-applicable | not-applicable | rejected | no | unproven |
| C-008 | Optional dependency audit tooling found vulnerabilities. | rejected | not-applicable | not-applicable | not-applicable | rejected | no | rejected |

Consensus notes:

- C-001 was re-verified by reopening `src/providers/anthropic/models.rs:4`, `src/providers/gemini/models.rs:4`, and `src/providers/openai/models.rs:4`; `cargo doc --all-features --no-deps` output in `.audit/doc-all-features.txt` contains the three rustdoc warnings.
- C-002 was re-verified by reopening `src/providers/openai/types.rs:402-418`, the checked-in streaming expectation at `src/providers/openai/client.rs:1048-1082`, and Gemini's non-streaming override at `src/providers/gemini/types.rs:538-548`; the local failing repro is captured in `.audit/repro-openai-nonstream-tool-finish.txt`.
- C-003 was re-verified by reopening `src/runtime/turn.rs:33-44` and `src/runtime/message_stream.rs:117-158`; failing local repros are captured in `.audit/repro-run-turn-terminal-hang.txt` and `.audit/repro-message-stream-terminal-hang.txt`.

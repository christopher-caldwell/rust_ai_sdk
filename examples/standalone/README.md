# Examples

Run examples from the standalone examples package directory.

```sh
cd examples/standalone
OPENAI_API_KEY=... cargo run --bin openai-generate
OPENAI_API_KEY=... cargo run --bin openai-stream
OPENAI_API_KEY=... cargo run --bin openai-tool-use
OPENAI_API_KEY=... cargo run --bin openai-tool-stream
OPENAI_API_KEY=... cargo run --bin openai-event-inspection
ANTHROPIC_API_KEY=... cargo run --bin anthropic-tool-use
GEMINI_API_KEY=... cargo run --bin gemini-generate
GEMINI_API_KEY=... cargo run --bin gemini-stream
GEMINI_API_KEY=... cargo run --bin gemini-tool-use
```

`openai-tool-use` runs the tool loop through the runtime helper and prints only
the tool execution plus final answer. `openai-tool-stream` prints text as it
arrives, executes the requested tool in application code, appends the tool
result, and streams the final model turn. `openai-event-inspection` prints the
structured stream events directly.

The weather tool is deterministic demo code. It does not call an external
service; the model receives a hardcoded JSON result from the example.

`anthropic-tool-use` demonstrates the same structured non-streaming tool loop
against Anthropic.

`gemini-generate`, `gemini-stream`, and `gemini-tool-use` demonstrate the same
flows against Gemini using the native Gemini API.

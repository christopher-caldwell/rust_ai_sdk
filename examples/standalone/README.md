# Standalone Examples

Small provider-specific binaries for trying the SDK without the chatbot app.
Each binary is intentionally close to the SDK API it demonstrates.

## Setup

```sh
cd examples/standalone
cp .env.example .env
```

Fill in the key for the provider you want to run:

```sh
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

Optional model overrides:

```sh
OPENAI_MODEL=gpt-5.4-nano
ANTHROPIC_MODEL=claude-haiku-4-5
GEMINI_MODEL=gemini-2.5-flash-lite
```

The `Justfile` loads `.env`, so you can run examples with `just`.

## OpenAI

```sh
just openai-generate
just openai-stream
just openai-tool-use
just openai-tool-stream
just openai-event-inspection
```

- `openai-generate` calls `generate_text(...)` and prints the final text plus
  usage.
- `openai-stream` calls `stream_text(...)` and prints text deltas as they
  arrive.
- `openai-tool-use` runs a non-streaming model -> tool -> model loop with
  `run_turn(...)`.
- `openai-tool-stream` uses the high-level streaming turn helper, executes the
  requested tool in application code, appends the tool result, and runs the
  final model turn.
- `openai-event-inspection` prints raw provider-neutral `StreamEvent` values for
  debugging stream behavior.

## Anthropic

```sh
just anthropic-generate
just anthropic-stream
just anthropic-tool-use
```

These mirror the OpenAI generate, stream, and high-level tool-loop examples
against the Anthropic provider.

## Gemini

```sh
just gemini-generate
just gemini-stream
just gemini-tool-use
```

These mirror the same flows against Gemini using the native Gemini API.

## Demo Tools

The tool examples use deterministic weather demo code. They do not call an
external weather service. The point is to show the ownership boundary:

- the SDK sends provider-neutral tool definitions to the model;
- the provider returns provider-neutral tool calls;
- your application executes the tool and decides what result is safe to return;
- the SDK helper builds the continuation request for the next model turn.

## Direct Cargo Commands

If you do not use `just`, run the binaries directly:

```sh
cargo run --bin openai-generate
cargo run --bin openai-stream
cargo run --bin openai-tool-use
cargo run --bin openai-tool-stream
cargo run --bin openai-event-inspection
cargo run --bin anthropic-generate
cargo run --bin anthropic-stream
cargo run --bin anthropic-tool-use
cargo run --bin gemini-generate
cargo run --bin gemini-stream
cargo run --bin gemini-tool-use
```

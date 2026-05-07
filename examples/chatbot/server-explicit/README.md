# Explicit Chatbot Server Example

This is the "show the seams" version of the chatbot server. It still uses the
optional `message-stream` add-on for the AI SDK UI-message wire protocol, but
it builds the provider-neutral `TextRequest` explicitly instead of using
`compose_text_request(...)`.

Use this example when you want to see exactly where request conversion, model
options, tool definitions, stream startup, and HTTP response wrapping happen.

## Run It

This server uses the same Vite frontend as `examples/chatbot/web`, which proxies
`/api/chat` to `http://127.0.0.1:3001`.

Create `server-explicit/.env` or reuse `server/.env`:

```sh
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4.1-mini
PORT=3001
```

Run the explicit server:

```sh
cd examples/chatbot/server-explicit
just run
```

Run the web app in another terminal:

```sh
cd examples/chatbot
just web
```

## What This Example Uses

The server enables:

```toml
another-ai-sdk = { path = "../../..", features = ["message-stream"] }
```

It uses these SDK helpers:

- `MessageStreamRequest` for deserializing AI SDK UI-message JSON.
- `messages_to_sdk_messages(...)` for converting text UI message parts into
  provider-neutral `Message` values.
- `stream_text_messages(...)` for the reusable model/tool loop and SSE chunk
  protocol.
- `MESSAGE_STREAM_*` constants for the required response headers.

It does not use `compose_text_request(...)`; the request builder stays visible:

```rust
let messages = messages_to_sdk_messages(input, SYSTEM_PROMPT);
let tool_definitions = tools.definitions();

TextRequest::builder()
    .messages(messages)
    .max_output_tokens(options.max_output_tokens)
    .temperature(options.temperature)
    .tools(tool_definitions)
    .build()
```

## Boundary

The SDK add-on owns protocol boilerplate:

- AI SDK UI-message request shape.
- UI text part conversion.
- Model/tool/model streaming loop.
- SSE chunk formatting.
- Finish, error, and `[DONE]` chunks.

The app still owns server policy:

- Provider/model construction.
- System prompt.
- Which tools exist.
- Tool authorization and execution.
- Route shape.
- Framework response wrapping.

# Chatbot Server Example

This is a small Axum server that connects the Vite chat UI to the local
`another-ai-sdk` crate. It keeps the app code focused on provider setup, route
wrapping, and demo tools while the optional `message-stream` SDK feature owns
the reusable AI SDK UI-message stream protocol work.

For a less black-box version of the same server, see
`examples/chatbot/server-explicit`. That example uses the same stream protocol
helpers, but builds the `TextRequest` explicitly instead of calling
`compose_text_request(...)`.

The browser talks in AI SDK UI-message JSON and SSE chunks. The Rust server
talks to the SDK in provider-neutral `TextRequest`, `ToolDefinition`, and
`StreamEvent` types.

## Run It

Create `server/.env`:

```sh
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4.1-mini
PORT=3001
```

Run the server:

```sh
cd examples/chatbot/server
just run
```

Or from the chatbot root:

```sh
cd examples/chatbot
just server
```

The server listens on `http://127.0.0.1:3001`.

Routes:

- `GET /health` returns `ok`.
- `POST /api/chat` streams an AI SDK UI-message response.

## Dependency

The server enables the optional stream add-on:

```toml
another-ai-sdk = { path = "../../..", features = ["message-stream"] }
```

That feature provides:

- `MessageStreamRequest`
- `MessageStreamOptions`
- `compose_text_request(...)`
- `stream_text_messages(...)`
- SSE protocol header constants

## Request Shape

The frontend uses `useChat()` from `@ai-sdk/react` with `DefaultChatTransport`.
That transport sends UI messages like:

```json
{
  "messages": [
    {
      "id": "abc",
      "role": "user",
      "parts": [{ "type": "text", "text": "What is the weather in Paris?" }]
    }
  ]
}
```

The Axum handler deserializes that body directly into `MessageStreamRequest`.
For this first version, the SDK adapter converts text parts and ignores other
UI parts.

## Handler Flow

The route stays framework-specific:

```rust
Router::new()
    .route("/health", get(|| async { "ok" }))
    .route("/api/chat", post(chat_handler))
```

The handler composes a provider-neutral SDK request, starts the byte stream,
and wraps it in an Axum response:

```rust
let options = MessageStreamOptions::default();
let request = compose_text_request(input, SYSTEM_PROMPT, options, state.tools.definitions());
let stream = stream_text_messages(state.model, request, state.tools, options);

Response::builder()
    .header(header::CONTENT_TYPE, MESSAGE_STREAM_CONTENT_TYPE)
    .header(header::CACHE_CONTROL, MESSAGE_STREAM_CACHE_CONTROL)
    .header(MESSAGE_STREAM_PROTOCOL_HEADER, MESSAGE_STREAM_PROTOCOL_VERSION)
    .body(Body::from_stream(stream))
```

The SDK add-on is framework-independent because it emits
`Stream<Item = MessageStreamChunk>` where each chunk is SSE-formatted
`bytes::Bytes`. Axum only supplies JSON extraction, headers, and body wrapping.

## Stream Behavior

`stream_text_messages(...)` handles the reusable model/tool loop:

- Emits `start`, per-step `start-step` and `finish-step`, final `finish`, and
  `data: [DONE]`.
- Maps text deltas to `text-start`, `text-delta`, and `text-end`.
- Maps tool calls to `tool-input-start`, `tool-input-delta`,
  `tool-input-available`, and `tool-output-available`.
- Executes tools through the app-owned `ToolRegistry`.
- Builds continuation requests with the assistant turn and tool results.
- Stops after `MessageStreamOptions::max_model_steps` to avoid an accidental
  infinite tool/model loop.
- Emits protocol-shaped error chunks if SDK stream creation or consumption
  fails.

The response must include the AI SDK compatibility header:

```http
x-vercel-ai-ui-message-stream: v1
```

Use the exported constants instead of spelling these values in every adapter:

```rust
MESSAGE_STREAM_CONTENT_TYPE
MESSAGE_STREAM_CACHE_CONTROL
MESSAGE_STREAM_PROTOCOL_HEADER
MESSAGE_STREAM_PROTOCOL_VERSION
```

## Model Setup

The example currently uses OpenAI:

```rust
let state = AppState {
    model: OpenAiChatModel::new(openai_api_key, model),
    tools: demo_tool_registry(),
};
```

`OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`.

## Adding A Tool

Add the tool definition and handler to `demo_tool_registry()`:

```rust
ToolRegistry::new().register(
    ToolDefinition::new(
        "lookup_order",
        "Look up an order by ID.",
        json!({
            "type": "object",
            "properties": {
                "order_id": { "type": "string" }
            },
            "required": ["order_id"],
            "additionalProperties": false
        }),
    ),
    |call| async move {
        let order_id = call.input["order_id"].as_str().unwrap_or_default();
        Ok(json!({ "order_id": order_id, "status": "shipped" }))
    },
)
```

Keep browser input untrusted. The server should own the tool registry,
authorization checks, execution, and final tool outputs.

## Current Demo Tools

`get_weather`

Returns deterministic demo weather for a requested city.

`get_current_time`

Returns the server time as Unix seconds for a requested timezone label.

## Production Notes

This example is deliberately not production-ready. Real application code would
usually add:

- Authentication and per-user tool authorization.
- Request size limits.
- Tool timeouts and cancellation.
- Persistent canonical chat history.
- Audit logs for tool calls and outputs.
- Better error redaction.
- CORS policy if the web app is served from another origin.

The important boundary should stay the same: keep provider-neutral model and
tool orchestration in the Rust server, and expose only the AI SDK UI-message
stream protocol to the browser.

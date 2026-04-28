# Chatbot Server Example

This is a small Axum server that connects the Vite chat UI to the local
`another-ai-sdk` crate. It intentionally keeps the app code thin: one chat route,
provider setup, request/response adapters, and a couple of demo tools.

The important idea is that the browser talks in Vercel AI SDK UI-message
format, while the Rust server talks to this SDK in provider-neutral
`TextRequest`, `Message`, `ToolDefinition`, and `StreamEvent` types.

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
- `POST /api/chat` streams a Vercel AI SDK UI message response.

## Request Shape

The frontend uses `useChat()` from `@ai-sdk/react` with `DefaultChatTransport`.
That transport sends a JSON body containing UI messages:

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

The server defines a minimal compatible request type:

```rust
struct ChatRequest {
    messages: Vec<UiMessage>,
}

struct UiMessage {
    role: String,
    parts: Vec<UiPart>,
}
```

For this demo, only text parts are converted. Tool history and other UI parts
are ignored because the server executes tools itself and owns the trusted tool
results.

## Response Shape

The response is Server-Sent Events using the AI SDK UI message stream protocol.
The required compatibility header is:

```http
x-vercel-ai-ui-message-stream: v1
```

The server writes chunks like:

```text
data: {"type":"start","messageId":"msg_..."}

data: {"type":"start-step"}

data: {"type":"text-start","id":"text_0"}
data: {"type":"text-delta","id":"text_0","delta":"Hello"}
data: {"type":"text-end","id":"text_0"}

data: {"type":"finish-step"}
data: {"type":"finish","finishReason":"stop"}
data: [DONE]
```

For tool calls, the server also emits:

```text
data: {"type":"tool-input-start","toolCallId":"call_...","toolName":"get_weather"}
data: {"type":"tool-input-delta","toolCallId":"call_...","inputTextDelta":"..."}
data: {"type":"tool-input-available","toolCallId":"call_...","toolName":"get_weather","input":{"location":"Paris"}}
data: {"type":"tool-output-available","toolCallId":"call_...","output":{"forecast":"mild and cloudy"}}
```

The frontend can render these as tool parts without inventing a custom JSON or
NDJSON protocol.

## Server Flow

The route is:

```rust
Router::new()
    .route("/health", get(|| async { "ok" }))
    .route("/api/chat", post(chat_handler))
```

`chat_handler` does not buffer a full response. It immediately returns an Axum
`Body::from_stream(...)` with SSE headers:

```rust
Response::builder()
    .header(header::CONTENT_TYPE, "text/event-stream")
    .header(header::CACHE_CONTROL, "no-cache")
    .header("x-vercel-ai-ui-message-stream", "v1")
    .body(Body::from_stream(stream))
```

The actual orchestration happens in `chat_stream(...)`.

## Request Adapter

`ui_messages_to_sdk_messages(...)` converts browser UI messages into SDK
messages:

- `role: "user"` becomes `Message::user(text)`.
- `role: "assistant"` becomes `Message::assistant(text)`.
- `role: "system"` becomes `Message::system(text)`.
- Non-text UI parts are ignored in this first version.

The function also injects a server-side system prompt:

```rust
Message::system(
    "You are a concise demo chatbot running behind an Axum server..."
)
```

Then the handler builds a provider-neutral SDK request:

```rust
let mut request = TextRequest::builder()
    .messages(ui_messages_to_sdk_messages(input.messages))
    .max_output_tokens(800)
    .temperature(0.7)
    .tools(state.tools.definitions())
    .build();
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

The route calls the SDK through:

```rust
let sdk_stream = stream_text(&state.model, request.clone()).await?;
```

That returns provider-neutral stream events. The Axum app does not parse
provider-specific OpenAI tool-call chunks.

## Streaming Adapter

`chat_stream(...)` maps SDK stream events to AI SDK UI-message stream chunks:

- `StreamEvent::TextDelta` starts a text part if needed, then emits `text-delta`.
- `StreamEvent::ToolCallStarted` emits `tool-input-start`.
- `StreamEvent::ToolCallDelta` emits `tool-input-delta`.
- `StreamEvent::ToolCallReady` emits `tool-input-available`.
- `StreamEvent::Finished` stores the finish reason for the final `finish` part.

Each event is also pushed into `TurnAccumulator`, which reconstructs the
assistant turn for continuation without duplicating provider-specific parsing in
the example.

## Tool Loop

The model may finish in one of two ways:

- No tool calls: send `finish-step`, `finish`, and `[DONE]`.
- One or more tool calls: execute tools, append results, and run another model step.

The loop is bounded:

```rust
for step_index in 0..5 {
    // model -> optional tools -> continuation
}
```

This prevents an accidental infinite model/tool loop in the demo.

When tools are called, the app builds a continuation request:

```rust
let mut builder =
    ContinuationBuilder::from_request(request).with_assistant_turn(assistant_parts);

for call in &tool_calls {
    let output = state.tools.execute(call).await?;
    builder = builder.with_tool_result(&call.id, output.to_string());
}

request = builder.build();
```

This is the key SDK composition point. The server app, not the SDK, decides:

- Which tools are available.
- Whether a tool call is allowed.
- How the tool is executed.
- What result is sent back to the model.
- What tool output is streamed to the browser.

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

## Error Handling

If SDK stream creation or stream consumption fails, the server emits an AI SDK
UI-message error part, then closes the stream:

```text
data: {"type":"error","errorText":"..."}
data: {"type":"finish-step"}
data: {"type":"finish","finishReason":"error"}
data: [DONE]
```

This keeps the frontend parser in a valid state.

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
- Tests for exact SSE chunks from mocked SDK events.

The important boundary should stay the same: keep provider-neutral model and
tool orchestration in the Rust server, and expose only the AI SDK UI-message
stream protocol to the browser.

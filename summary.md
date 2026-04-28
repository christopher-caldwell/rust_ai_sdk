# Tool-Aware SDK Work Summary

## What changed in `rust_ai_sdk`

The SDK was extended from text-only chat primitives into a provider-neutral, tool-aware chat layer that can support a Rust server acting as the AI orchestration backend for a static Vite frontend.

Core additions:

- `Message` now supports structured parts through `MessagePart::Text`, `MessagePart::ToolCall`, and `MessagePart::ToolResult`.
- `TextRequest` can carry provider-neutral `ToolDefinition` values and an optional `ToolChoice`.
- `StreamEvent` now distinguishes text streaming from tool-call lifecycle events:
    - `TextDelta`
    - `ToolCallStarted`
    - `ToolCallDelta`
    - `ToolCallReady`
    - `Finished`
- `ChatResult` represents a structured assistant turn while preserving finish reason, usage, and response metadata.
- `FinishReason::ToolUse` was added.
- `runtime::turn` provides:
    - `run_turn(...)` for model-turn execution with tool detection.
    - `TurnOutcome::Completed` vs `TurnOutcome::ToolsRequired`.
    - `ContinuationBuilder` for appending assistant tool calls and tool results in the right order.

Provider support:

- OpenAI now translates tools, tool choices, assistant tool calls, tool result messages, structured final results, and tool-aware streaming events.
- Anthropic now supports the same core abstractions for tool definitions, tool-use response blocks, tool results, and streamed tool-use blocks.
- Both providers expose `generate_chat(...)` for structured non-streaming results while preserving the existing text-only `generate(...)` and `stream(...)` trait behavior.

Examples and verification:

- Added examples for OpenAI tool use, OpenAI tool streaming, OpenAI event inspection, and Anthropic tool use.
- Added Justfile recipes for the new examples.
- Live provider examples were run successfully with environment API keys.
- `cargo test` passes with 49 tests.
- `cargo build --manifest-path examples/Cargo.toml` passes.

## What remains to hook up a Vite frontend

The SDK is now ready to sit behind an Axum server, but the application server still needs to define the browser-facing protocol. The SDK intentionally does not own HTTP routing, auth, persistence, tool policy, or frontend transport.

The remaining work is application-level:

- Define a frontend chat message shape.
- Add an Axum `POST /api/chat` route that accepts the frontend messages.
- Convert frontend messages into `ai_sdk::core::request::TextRequest`.
- Register the tools allowed for that route or user.
- Stream model events from the Rust server to the browser.
- Execute tools on the server when `ToolCallReady` events arrive.
- Append tool results with `ContinuationBuilder`.
- Continue the model loop until the final assistant answer is complete.
- Persist the final assistant message, usage, metadata, and tool-call audit data after the stream finishes.

## Vercel AI SDK compatibility target

Assume the Vite frontend uses `useChat()` from `@ai-sdk/react`. In that case,
the Rust backend should not invent its own NDJSON protocol for the main chat
route. It should return the AI SDK UI message stream protocol so `useChat()`
can parse the response and update `messages` automatically.

For AI SDK v6, the custom backend compatibility target is:

- Frontend hook: `useChat()` from `@ai-sdk/react`.
- Frontend transport: default `/api/chat`, or `new DefaultChatTransport({ api: "/api/chat" })`.
- Request method: HTTP `POST`.
- Request body: AI SDK `UIMessage[]` payload sent by the transport.
- Response body: Server-Sent Events containing UI message stream parts.
- Required response header: `x-vercel-ai-ui-message-stream: v1`.
- Recommended response content type: `text/event-stream`.
- Stream termination: final `data: [DONE]` SSE event.

The AI SDK docs also describe a simpler text stream protocol, but that is not
the right target for this app because text streams only carry plain text. This
backend needs tool-call visibility, tool outputs, multi-step model calls, usage
metadata, and persistence hooks. Use the UI message/data stream protocol.

Relevant stream parts for this SDK:

```text
data: {"type":"start","messageId":"msg_..."}

data: {"type":"start-step"}

data: {"type":"text-start","id":"text_0"}
data: {"type":"text-delta","id":"text_0","delta":"The weather in Paris is "}
data: {"type":"text-end","id":"text_0"}

data: {"type":"tool-input-start","toolCallId":"call_1","toolName":"get_weather"}
data: {"type":"tool-input-delta","toolCallId":"call_1","inputTextDelta":"{\"location\":\"Paris\"}"}
data: {"type":"tool-input-available","toolCallId":"call_1","toolName":"get_weather","input":{"location":"Paris"}}
data: {"type":"tool-output-available","toolCallId":"call_1","output":{"forecast":"mild and cloudy","temperature_c":18}}

data: {"type":"finish-step"}
data: {"type":"start-step"}

data: {"type":"text-start","id":"text_1"}
data: {"type":"text-delta","id":"text_1","delta":"mild and cloudy."}
data: {"type":"text-end","id":"text_1"}

data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

The important mapping is direct:

- `StreamEvent::TextDelta` becomes `text-start` once, then `text-delta`, then `text-end` when the current model step ends.
- `StreamEvent::ToolCallStarted` becomes `tool-input-start`.
- `StreamEvent::ToolCallDelta` becomes `tool-input-delta`.
- `StreamEvent::ToolCallReady` becomes `tool-input-available`.
- Application tool execution result becomes `tool-output-available`.
- Each model call in a model -> tool -> model loop is wrapped by `start-step` and `finish-step`.
- The entire assistant response is wrapped by `start` and `finish`.

## Axum route shape

At a high level, the Axum handler should create an SSE stream body and run the
SDK loop inside that stream. The important detail is that tool execution stays
in your application code, not inside the SDK.

The route has two adapters:

- Request adapter: AI SDK `UIMessage[]` -> `Vec<Message>`.
- Response adapter: `rust_ai_sdk::StreamEvent` -> AI SDK UI message stream SSE parts.

The incoming body shape depends on the exact transport options you use, but with
the default AI SDK transport you should expect a JSON object containing
`messages`. Store any custom application context in `DefaultChatTransport`
`body` or headers and keep model/provider credentials only on the Rust server.

Pseudo-code:

```rust
async fn chat_handler(
    State(app): State<AppState>,
    Json(input): Json<ChatRequest>,
) -> impl IntoResponse {
    let stream = async_stream::try_stream! {
        let message_id = new_message_id();
        yield sse_json(json!({ "type": "start", "messageId": message_id }));

        let mut request = TextRequest {
            messages: ui_messages_to_sdk_messages(input.messages),
            max_output_tokens: Some(800),
            temperature: Some(0.7),
            tools: app.tool_registry.definitions_for_user(&input.user_id),
            tool_choice: None,
        };

        let mut step_index = 0usize;

        loop {
            yield sse_json(json!({ "type": "start-step" }));

            let mut sdk_stream = app.model.stream(request.clone()).await?;
            let mut assistant_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut text_buffer = String::new();
            let text_part_id = format!("text_{step_index}");
            let mut text_started = false;

            while let Some(event) = sdk_stream.next().await {
                match event? {
                    StreamEvent::TextDelta(text) => {
                        if !text_started {
                            yield sse_json(json!({
                                "type": "text-start",
                                "id": text_part_id,
                            }));
                            text_started = true;
                        }

                        text_buffer.push_str(&text);
                        yield sse_json(json!({
                            "type": "text-delta",
                            "id": text_part_id,
                            "delta": text,
                        }));
                    }
                    StreamEvent::ToolCallStarted { id, name, index } => {
                        yield sse_json(json!({
                            "type": "tool-input-start",
                            "toolCallId": id.clone(),
                            "toolName": name.clone(),
                        }));
                    }
                    StreamEvent::ToolCallDelta { id, index, input_delta } => {
                        yield sse_json(json!({
                            "type": "tool-input-delta",
                            "toolCallId": id.clone(),
                            "inputTextDelta": input_delta.clone(),
                        }));
                    }
                    StreamEvent::ToolCallReady { id, name, index, input } => {
                        yield sse_json(json!({
                            "type": "tool-input-available",
                            "toolCallId": id.clone(),
                            "toolName": name.clone(),
                            "input": input.clone(),
                        }));
                        tool_calls.push(ToolCall { id, name, input });
                    }
                    StreamEvent::Finished { finish_reason, usage, response } => {
                        if text_started {
                            yield sse_json(json!({
                                "type": "text-end",
                                "id": text_part_id,
                            }));
                        }

                        if !text_buffer.is_empty() {
                            assistant_parts.push(MessagePart::Text(text_buffer.clone()));
                        }

                        if tool_calls.is_empty() {
                            app.persist_final_message(&input, &assistant_parts, usage, response).await?;
                            yield sse_json(json!({ "type": "finish-step" }));
                            yield sse_json(json!({ "type": "finish" }));
                            yield sse_done();
                            return;
                        }

                        for call in &tool_calls {
                            assistant_parts.push(MessagePart::ToolCall(call.clone()));
                        }

                        yield sse_json(json!({ "type": "finish-step" }));
                    }
                }
            }

            let mut builder = ContinuationBuilder::from_request(request)
                .with_assistant_turn(assistant_parts);

            for call in &tool_calls {
                let result = app.tool_registry.execute(call).await?;
                yield sse_json(json!({
                    "type": "tool-output-available",
                    "toolCallId": call.id.clone(),
                    "output": parse_json_or_string(&result),
                }));
                builder = builder.with_tool_result(&call.id, result);
            }

            request = builder.build();
            step_index += 1;
        }
    };

    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-vercel-ai-ui-message-stream", "v1")
        .body(Body::from_stream(stream))
        .unwrap()
}
```

Helper shape:

```rust
fn sse_json(value: serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {}\n\n", value))
}

fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}
```

The real code should add your app error type, auth, cancellation behavior,
request limits, and tool timeouts.

## Vite client shape with `useChat()`

Install the AI SDK packages in the Vite app and let `useChat()` own the stream
parsing and message state.

```ts
npm install ai @ai-sdk/react
```

Basic Vite component:

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState } from 'react';

export function Chat() {
  const [input, setInput] = useState('');

  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
      credentials: 'include',
      headers: () => ({
        Authorization: `Bearer ${getAuthToken()}`,
      }),
      body: () => ({
        conversationId: getConversationId(),
      }),
    }),
  });

  return (
    <form
      onSubmit={event => {
        event.preventDefault();
        sendMessage({ text: input });
        setInput('');
      }}
    >
      {messages.map(message => (
        <article key={message.id}>
          {message.parts.map((part, index) => {
            switch (part.type) {
              case 'text':
                return <p key={index}>{part.text}</p>;
              default:
                return <pre key={index}>{JSON.stringify(part, null, 2)}</pre>;
            }
          })}
        </article>
      ))}

      <input value={input} onChange={event => setInput(event.target.value)} />
      <button disabled={status === 'streaming'}>Send</button>
    </form>
  );
}
```

Rendering notes:

- Text parts should render normally.
- Tool parts should be expected in `message.parts`; render them explicitly if you want the UI to show tool activity.
- If you do not want visible tool details, filter tool parts out and render only `part.type === "text"`.
- Use custom `data-*` stream parts for application-specific status updates that are not native AI SDK tool parts.
- Keep server-side tools authoritative. Do not let the browser choose arbitrary tools or submit trusted tool outputs.

## Request conversion details

The Vercel AI SDK sends UI messages, not this SDK's internal `Message` type.
The Axum app needs a conversion layer.

Practical mapping:

- UI user text part -> `Message::user(text)`.
- UI assistant text part -> `Message::assistant(text)` when replaying history.
- UI assistant tool-use part -> `MessagePart::ToolCall` if you choose to replay previous tool calls.
- UI tool output part -> `Message::tool_result(...)` if you choose to replay previous tool outputs.

For a first version, the server can ignore client-supplied historical tool
parts and load canonical history from the database instead. That is usually
cleaner: the browser sends the new message, and the server reconstructs trusted
conversation history, tool calls, and tool results from storage.

## Adapter code that should probably live outside this SDK

Keep the Vercel protocol adapter in the Axum application first:

- `ui_messages_to_sdk_messages(...)`
- `stream_event_to_ui_message_sse(...)`
- `sse_json(...)`
- `sse_done(...)`
- tool registry and execution policy
- final message persistence

Once the shape stabilizes, it may be worth adding an optional crate or feature
to this repo, for example `ai_sdk_vercel`, that exposes a reusable
`UiMessageStreamAdapter`. Do not put Axum-specific route code into the core SDK.

## Practical next steps

1. Create an Axum server crate or app module that depends on this SDK.
2. Define `ChatRequest` around AI SDK `UIMessage[]` plus app context such as `conversationId`.
3. Implement AI SDK UI message -> SDK message conversion.
4. Implement a server-side tool registry with explicit allowlists per route/user.
5. Implement the AI SDK UI message stream SSE adapter using `Body::from_stream`.
6. Add timeout and cancellation handling around tool execution and provider streams.
7. Add persistence after final `Finished`, including assistant text, structured parts, usage, provider response metadata, and tool audit records.
8. Wire the Vite chat UI to `useChat()` with `DefaultChatTransport`.
9. Add tests that feed mocked SDK events into the Vercel stream adapter and assert exact SSE output.

## References

- AI SDK stream protocols: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
- AI SDK `useChat()`: https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat
- AI SDK transport configuration: https://ai-sdk.dev/docs/ai-sdk-ui/transport

The SDK boundary is now suitable for this architecture: the outer app no longer needs to parse provider-specific tool-call formats or scrape meaning out of plain text.

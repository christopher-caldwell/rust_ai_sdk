# another-ai-sdk

A small Rust SDK for building provider-neutral AI chat applications with
streaming and tool calling.

This project is inspired by the ergonomics of the Vercel AI SDK: define a
model, build a request, stream events, execute tools in your application, and
continue the model loop. The goal is not to copy the JavaScript API directly,
but to bring the same practical composition model to Rust servers.

## Features

- **Provider-neutral chat primitives**: shared `Message`, `TextRequest`,
  `ChatResult`, `StreamEvent`, and tool-call types.
- **OpenAI, Anthropic, and Gemini providers**: generate text, generate structured chat
  turns, and stream provider events through a common interface.
- **Streaming text and tools**: text deltas, tool-call starts, argument deltas,
  ready tool calls, usage, finish reason, and response metadata.
- **Tool-aware runtime helpers**: `run_turn`, `ContinuationBuilder`,
  `TurnAccumulator`, and `ToolRegistry` for model -> tool -> model loops.
- **Optional message stream add-on**: feature-gated helpers for serving the
  AI SDK UI-message SSE wire protocol from any Rust HTTP framework that can
  stream bytes.
- **Server-friendly design**: tools execute in your application code, so you
  keep control of authorization, side effects, persistence, and auditing.
- **Examples**: standalone provider demos and an Axum + Vite chatbot using the
  Vercel AI SDK UI message stream protocol.

## Installation

This crate is currently used from this repository. Add it as a path dependency:

```toml
[dependencies]
another-ai-sdk = { path = "/path/to/rust_ai_sdk" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures-util = "0.3"
serde_json = "1"
```

For examples inside this repo, use:

```toml
another-ai-sdk = { path = "../.." }
```

To enable the framework-independent AI SDK UI-message stream adapter:

```toml
another-ai-sdk = { path = "/path/to/rust_ai_sdk", features = ["message-stream"] }
```

Provider examples require API keys:

```sh
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

The same interface works with Gemini:

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::gemini::{GeminiChatModel, GeminiModel};

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    let model = GeminiChatModel::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY is required"),
        GeminiModel::Gemini2_5Flash,
    );

    let result = generate_text(&model, TextRequest::prompt("Write a haiku about Rust.")).await?;
    println!("{}", result.text);

    Ok(())
}
```

## Basic Usage

Generate one text response:

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::openai::model::OpenAiChatModel;

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    let model = OpenAiChatModel::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
        "gpt-4.1-mini",
    );

    let request = TextRequest::builder()
        .system("You are concise.")
        .prompt("Explain Rust ownership in one sentence.")
        .max_output_tokens(200)
        .temperature(0.3)
        .build();

    let result = generate_text(&model, request).await?;
    println!("{}", result.text);

    Ok(())
}
```

Stream text deltas:

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::openai::model::OpenAiChatModel;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    let model = OpenAiChatModel::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
        "gpt-4.1-mini",
    );

    let request = TextRequest::prompt("Write a short haiku about Rust.");
    let mut stream = stream_text(&model, request).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(delta) => print!("{delta}"),
            StreamEvent::Finished { finish_reason, .. } => {
                println!("\nfinished: {finish_reason:?}");
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Tool Calling

Tools are intentionally application-owned. The SDK sends provider-neutral tool
definitions to the model and gives you provider-neutral tool calls back. Your
server decides what each tool is allowed to do.

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::openai::model::OpenAiChatModel;
use serde_json::{Value, json};

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    let model = OpenAiChatModel::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
        "gpt-4.1-mini",
    );

    let tools = ToolRegistry::new().register(
        ToolDefinition::new(
            "get_weather",
            "Get a demo weather report for a city.",
            json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
        ),
        |call| async move {
            let location = call
                .input
                .get("location")
                .and_then(Value::as_str)
                .unwrap_or("unknown");

            Ok(json!({
                "location": location,
                "forecast": "mild and cloudy",
                "temperature_c": 18
            }))
        },
    );

    let mut request = TextRequest::builder()
        .prompt("What is the weather in Paris?")
        .tools(tools.definitions())
        .build();

    loop {
        match run_turn(&model, request).await? {
            TurnOutcome::Completed(result) => {
                println!("{}", result.text());
                break;
            }
            TurnOutcome::ToolsRequired {
                assistant_parts,
                tool_calls,
                ..
            } => {
                let mut continuation =
                    ContinuationBuilder::from_request(request)
                        .with_assistant_turn(assistant_parts);

                for call in &tool_calls {
                    let output = tools.execute(call).await?;
                    continuation =
                        continuation.with_tool_result(&call.id, output.to_string());
                }

                request = continuation.build();
            }
        }
    }

    Ok(())
}
```

## Message Stream Add-On

Enable `message-stream` when a server needs to accept AI SDK UI-message JSON
and stream the matching SSE protocol back to a browser or other client. The
add-on returns `bytes::Bytes`, not Axum/Rocket/Actix response types.

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::openai::model::OpenAiChatModel;

async fn handler(input: MessageStreamRequest, model: OpenAiChatModel, tools: ToolRegistry) {
    let options = MessageStreamOptions::default();
    let request = compose_text_request(
        input,
        "You are a concise assistant.",
        options,
        tools.definitions(),
    );

    let stream = stream_text_messages(model, request, tools, options);

    // Framework code owns response wrapping. Set these protocol values on the
    // HTTP response and stream `stream` as the body.
    let _content_type = MESSAGE_STREAM_CONTENT_TYPE;
    let _cache_control = MESSAGE_STREAM_CACHE_CONTROL;
    let _protocol_header = MESSAGE_STREAM_PROTOCOL_HEADER;
    let _protocol_version = MESSAGE_STREAM_PROTOCOL_VERSION;
}
```

The helper intentionally stays thin: your app still owns JSON extraction,
model construction, tool registration and authorization, routing, response
headers, and HTTP body streaming.

## Examples

Run the standalone examples:

```sh
cd examples/standalone
cargo run --bin openai-stream
cargo run --bin openai-tool-use
cargo run --bin anthropic-stream
cargo run --bin gemini-stream
```

Run the chatbot example:

```sh
cd examples/chatbot
just server
just web
```

The chatbot server is Axum, the frontend is Vite + React, and the browser uses
`@ai-sdk/react` with the Vercel AI SDK UI message stream protocol.

For a more explicit server version that shows request composition while still
using the `message-stream` protocol helpers:

```sh
cd examples/chatbot
just server-explicit
just web
```

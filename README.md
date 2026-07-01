# another-ai-sdk

Provider-neutral Rust SDK for chat, streaming, and application-owned tool
calling across OpenAI, Anthropic, and Gemini.

## Features

- Generate text with OpenAI, Anthropic, or Gemini using one provider-neutral
  request API. See the [standalone examples](examples/standalone/README.md).
- Stream text and structured provider events through `stream_text(...)`. See
  the [streaming examples](examples/standalone/README.md#openai).
- Tool calling with app-owned execution through `ToolDefinition`,
  `ToolRegistry`, and `run_turn(...)`. See the
  [tool examples](examples/standalone/README.md#demo-tools).
- AI SDK UI-message stream helpers for Rust HTTP servers. See the
  [chatbot server guide](examples/chatbot/server/README.md).
- Full-stack Axum + Vite chatbot example using `@ai-sdk/react`. See the
  [chatbot example](examples/chatbot/README.md).

## Quickstart

```toml
[dependencies]
another-ai-sdk = "0.0.4"
# Needed for this example's async main. Skip this if your app already has a runtime.
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Generate a response with OpenAI:

```rust
use another_ai_sdk::prelude::*;
use another_ai_sdk::providers::openai::{OpenAiChatModel, OpenAiModel};

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    // Pick a provider and model. The SDK exposes the same core request API
    // across OpenAI, Anthropic, and Gemini.
    let model = OpenAiChatModel::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
        OpenAiModel::Gpt5_4Nano,
    );

    // Build a provider-neutral request with messages and generation options.
    let request = TextRequest::builder()
        .system("You are concise.")
        .prompt("Explain Rust ownership in one sentence.")
        .max_output_tokens(200)
        .temperature(0.3)
        .build();

    // Send the request and read the normalized text result.
    let result = generate_text(&model, request).await?;
    println!("{}", result.text);

    Ok(())
}
```

For more info, see the detailed explanations in the
[standalone examples](examples/standalone/README.md) and
[chatbot example](examples/chatbot/README.md).

## Cargo Feature Flags

Default features are `providers-all` plus `streaming`.

| Feature          | What it enables                                                                 |
| ---------------- | ------------------------------------------------------------------------------- |
| `openai`         | OpenAI provider support and the HTTP client dependency.                         |
| `anthropic`      | Anthropic provider support and the HTTP client dependency.                      |
| `gemini`         | Gemini provider support and the HTTP client dependency.                         |
| `providers-all`  | All provider adapters: OpenAI, Anthropic, and Gemini.                           |
| `streaming`      | Provider SSE streaming support.                                                 |
| `message-stream` | Framework-independent helpers for the Vercel AI SDK UI-message stream protocol. |

Use only the core SDK types without provider HTTP dependencies:

```toml
another-ai-sdk = { version = "0.0.4", default-features = false }
```

Enable one provider with streaming:

```toml
another-ai-sdk = { version = "0.0.4", default-features = false, features = ["openai", "streaming"] }
```

Enable the UI-message stream adapter for an HTTP server:

```toml
another-ai-sdk = { version = "0.0.4", features = ["message-stream"] }
```

## Validation

```sh
just test
just check-examples
just check-chatbot-web
just doc
```

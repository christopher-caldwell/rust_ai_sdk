//! Provider-neutral chat, streaming, and tool-calling primitives.
//!
//! The crate is organized around a small mental model:
//!
//! - [`LanguageModel`](crate::core::model::LanguageModel) is the provider-neutral
//!   interface implemented by provider wrappers such as
//!   [`OpenAiChatModel`](crate::providers::openai::OpenAiChatModel),
//!   [`AnthropicChatModel`](crate::providers::anthropic::AnthropicChatModel), and
//!   [`GeminiChatModel`](crate::providers::gemini::GeminiChatModel).
//! - [`TextRequest`](crate::core::request::TextRequest) holds messages, generation
//!   options, tool definitions, and tool choice policy.
//! - [`StreamEvent`](crate::core::stream::StreamEvent) is the shared streaming event
//!   format for text deltas, tool calls, usage, finish reason, and response metadata.
//! - [`SdkError`](crate::core::error::SdkError) is the crate-wide error type returned
//!   by models and runtime helpers.
//! - [`run_turn`](crate::runtime::turn::run_turn) and
//!   [`ToolRegistry`](crate::runtime::tools::ToolRegistry) provide the high-level path
//!   for application-owned tool loops.
//!
//! Most applications should start with [`prelude`] plus one provider module:
//!
//! ```no_run
//! use another_ai_sdk::prelude::*;
//! use another_ai_sdk::providers::openai::{OpenAiChatModel, OpenAiModel};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), SdkError> {
//! let model = OpenAiChatModel::new("api-key".to_string(), OpenAiModel::Gpt4_1Mini);
//! let request = TextRequest::prompt("Write one sentence about Rust ownership.");
//! let result = generate_text(&model, request).await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```
//!
//! Enable the `message-stream` feature when an HTTP server needs to adapt SDK
//! messages to the Vercel AI SDK UI-message stream protocol. The feature exposes
//! framework-independent request composition and byte-stream helpers through
//! [`prelude`].

pub mod core;
pub mod providers;
pub mod runtime;

pub mod prelude {
    pub use crate::core::{
        error::SdkError,
        message::{Message, MessagePart, Role, ToolCall, ToolOutput, ToolResult},
        model::LanguageModel,
        request::{TextRequest, TextRequestBuilder},
        result::{ChatResult, TextResult},
        stream::{StreamEvent, TextEventStream},
        tool::{ToolChoice, ToolDefinition},
        types::{FinishReason, ResponseMetadata, Usage},
    };
    #[cfg(feature = "message-stream")]
    pub use crate::runtime::message_stream::{
        MESSAGE_STREAM_CACHE_CONTROL, MESSAGE_STREAM_CONTENT_TYPE, MESSAGE_STREAM_PROTOCOL_HEADER,
        MESSAGE_STREAM_PROTOCOL_VERSION, MessageStreamInputError, MessageStreamOptions,
        MessageStreamRequest, compose_text_request, messages_to_sdk_messages, stream_text_messages,
    };
    pub use crate::runtime::{
        generate::{generate_chat, generate_text},
        stream::stream_text,
        tools::ToolRegistry,
        turn::{AccumulatedTurn, ContinuationBuilder, TurnAccumulator, TurnOutcome, run_turn},
    };
}

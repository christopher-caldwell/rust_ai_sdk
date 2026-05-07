pub mod core;
pub mod providers;
pub mod runtime;

pub mod prelude {
    pub use crate::core::{
        error::SdkError,
        message::{Message, MessagePart, Role, ToolCall, ToolResult},
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
        MESSAGE_STREAM_PROTOCOL_VERSION, MessageStreamOptions, MessageStreamRequest,
        compose_text_request, messages_to_sdk_messages, stream_text_messages,
    };
    pub use crate::runtime::{
        generate::{generate_chat, generate_text},
        stream::stream_text,
        tools::ToolRegistry,
        turn::{AccumulatedTurn, ContinuationBuilder, TurnAccumulator, TurnOutcome, run_turn},
    };
}

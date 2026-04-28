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
    pub use crate::runtime::{
        generate::{generate_chat, generate_text},
        stream::stream_text,
        tools::ToolRegistry,
        turn::{AccumulatedTurn, ContinuationBuilder, TurnAccumulator, TurnOutcome, run_turn},
    };
}

use futures_core::Stream;
use serde_json::Value;
use std::pin::Pin;

use super::{
    error::SdkError,
    types::{FinishReason, ResponseMetadata, Usage},
};

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    Finished {
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    },
    ToolCallStarted {
        id: String,
        name: String,
        index: u32,
    },
    ToolCallDelta {
        id: String,
        index: u32,
        input_delta: String,
    },
    ToolCallReady {
        id: String,
        name: String,
        index: u32,
        input: Value,
    },
}

pub type TextEventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, SdkError>> + Send>>;
pub type EventStream = TextEventStream;

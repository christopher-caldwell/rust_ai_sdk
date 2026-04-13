use std::pin::Pin;
use futures_core::Stream;

use super::{
    error::SdkError,
    types::{FinishReason, ResponseMetadata, Usage},
};

#[derive(Debug)]
pub enum StreamEvent {
    TextDelta(String),
    Finished {
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    },
}

pub type TextEventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, SdkError>> + Send>>;

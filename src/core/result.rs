use super::types::{FinishReason, ResponseMetadata, Usage};

#[derive(Debug, Clone)]
pub struct TextResult {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub response: ResponseMetadata,
}

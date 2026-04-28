use super::message::{MessagePart, ToolCall};
use super::types::{FinishReason, ResponseMetadata, Usage};

#[derive(Debug, Clone)]
pub struct TextResult {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub response: ResponseMetadata,
}

/// Richer result used when tools may be involved.
#[derive(Debug, Clone)]
pub struct ChatResult {
    /// The assistant's turn as structured parts (text + tool calls, in order).
    pub parts: Vec<MessagePart>,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub response: ResponseMetadata,
}

impl ChatResult {
    /// All text parts joined together.
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// All ToolCall parts.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::ToolCall(tc) = p {
                    Some(tc)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn has_tool_calls(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, MessagePart::ToolCall(_)))
    }
}

use super::message::{MessagePart, ToolCall};
use super::types::{FinishReason, ResponseMetadata, Usage};

#[derive(Debug, Clone)]
/// Text-only result returned by convenience generation helpers.
pub struct TextResult {
    /// Generated assistant text.
    pub text: String,
    /// Why the provider ended generation.
    pub finish_reason: FinishReason,
    /// Token usage reported by the provider.
    pub usage: Option<Usage>,
    /// Provider response identifier and model metadata.
    pub response: ResponseMetadata,
}

/// Richer result used when tools may be involved.
#[derive(Debug, Clone)]
pub struct ChatResult {
    /// The assistant's turn as structured parts (text + tool calls, in order).
    pub parts: Vec<MessagePart>,
    /// Why the provider ended generation.
    pub finish_reason: FinishReason,
    /// Token usage reported by the provider.
    pub usage: Option<Usage>,
    /// Provider response identifier and model metadata.
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

    /// Return whether the assistant emitted at least one tool call.
    pub fn has_tool_calls(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, MessagePart::ToolCall(_)))
    }
}

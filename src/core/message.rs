use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single unit of content within a message.
#[derive(Debug, Clone, PartialEq)]
pub enum MessagePart {
    Text(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

/// A tool invocation emitted by the assistant.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// The result of executing a tool, sent back as a user-role message.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// A chat message. Simple text messages use `content`; structured messages use `parts`.
///
/// Provider translators call `effective_parts()` so both old-style struct literals
/// (`Message { role, content, parts: vec![] }`) and new-style constructors work
/// identically at the wire level.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    /// Plain text content — used when `parts` is empty.
    pub content: String,
    /// Structured parts. When non-empty, supersedes `content` at the wire level.
    pub parts: Vec<MessagePart>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: text.into(),
            parts: vec![],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: text.into(),
            parts: vec![],
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: text.into(),
            parts: vec![],
        }
    }

    /// Build an assistant message from structured parts (e.g. after a tool turn).
    pub fn assistant_parts(parts: Vec<MessagePart>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            parts,
        }
    }

    /// Build a tool-result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: String::new(),
            parts: vec![MessagePart::ToolResult(ToolResult {
                tool_call_id: tool_call_id.into(),
                content: content.into(),
            })],
        }
    }

    /// Returns the effective parts: `parts` if non-empty, else a single `Text` from `content`.
    pub fn effective_parts(&self) -> Vec<MessagePart> {
        if !self.parts.is_empty() {
            self.parts.clone()
        } else {
            vec![MessagePart::Text(self.content.clone())]
        }
    }

    pub fn is_text_only(&self) -> bool {
        self.parts.is_empty()
    }
}

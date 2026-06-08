use serde_json::{Value, json};

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

/// Semi-opaque provider metadata preserved for model/tool continuations.
///
/// Provider adapters may use this to carry fields that must round-trip back to
/// the same provider. Application code should treat the value as provider-owned
/// unless it deliberately opts into `as_raw` or `into_raw`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderMetadata {
    raw: Value,
}

impl ProviderMetadata {
    pub fn new(raw: Value) -> Self {
        Self { raw }
    }

    pub fn as_raw(&self) -> &Value {
        &self.raw
    }

    pub fn into_raw(self) -> Value {
        self.raw
    }

    pub fn provider_value(&self, provider_key: &str) -> &Value {
        self.raw.get(provider_key).unwrap_or(&self.raw)
    }

    pub fn provider_string(&self, provider_key: &str, metadata_key: &str) -> Option<String> {
        self.provider_value(provider_key)
            .get(metadata_key)?
            .as_str()
            .map(ToString::to_string)
    }
}

impl From<Value> for ProviderMetadata {
    fn from(value: Value) -> Self {
        Self::new(value)
    }
}

/// A tool invocation emitted by the assistant.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
            provider_metadata: None,
        }
    }

    pub fn from_json_input(
        id: impl Into<String>,
        name: impl Into<String>,
        input_json: impl AsRef<str>,
    ) -> Self {
        let input_json = input_json.as_ref();

        match serde_json::from_str(input_json) {
            Ok(input) => Self::new(id, name, input),
            Err(error) => Self::malformed_json_input(id, name, input_json, error.to_string()),
        }
    }

    pub fn malformed_json_input(
        id: impl Into<String>,
        name: impl Into<String>,
        raw_input: impl Into<String>,
        parse_error: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input: Value::Null,
            provider_metadata: Some(Self::malformed_input_metadata(raw_input, parse_error).into()),
        }
    }

    #[must_use]
    pub fn with_provider_metadata(mut self, metadata: impl Into<ProviderMetadata>) -> Self {
        self.provider_metadata = Some(metadata.into());
        self
    }

    pub fn has_malformed_input(&self) -> bool {
        self.malformed_input().is_some()
    }

    pub fn malformed_input_raw(&self) -> Option<&str> {
        self.malformed_input()
            .and_then(|metadata| metadata.get("raw"))
            .and_then(Value::as_str)
    }

    pub fn malformed_input_error(&self) -> Option<&str> {
        self.malformed_input()
            .and_then(|metadata| metadata.get("error"))
            .and_then(Value::as_str)
    }

    pub fn malformed_input_metadata(
        raw_input: impl Into<String>,
        parse_error: impl Into<String>,
    ) -> Value {
        json!({
            "another_ai_sdk": {
                "tool_input": {
                    "state": "malformed_json",
                    "raw": raw_input.into(),
                    "error": parse_error.into(),
                }
            }
        })
    }

    pub fn metadata_with_malformed_input(
        existing: Option<ProviderMetadata>,
        raw_input: impl Into<String>,
        parse_error: impl Into<String>,
    ) -> ProviderMetadata {
        let marker = Self::malformed_input_metadata(raw_input, parse_error);
        let existing = existing.map(ProviderMetadata::into_raw);

        let raw = match existing {
            Some(Value::Object(mut existing)) => {
                if let Some(marker_value) = marker.get("another_ai_sdk") {
                    existing.insert("another_ai_sdk".to_string(), marker_value.clone());
                }

                Value::Object(existing)
            }
            Some(existing) => {
                json!({
                    "provider": existing,
                    "another_ai_sdk": marker["another_ai_sdk"].clone(),
                })
            }
            None => marker,
        };

        ProviderMetadata::new(raw)
    }

    fn malformed_input(&self) -> Option<&Value> {
        let tool_input = self
            .provider_metadata
            .as_ref()?
            .as_raw()
            .get("another_ai_sdk")?
            .get("tool_input")?;

        let state = tool_input.get("state").and_then(Value::as_str);
        if state == Some("malformed_json") {
            Some(tool_input)
        } else {
            None
        }
    }
}

/// Output produced by a tool.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
}

impl ToolOutput {
    pub fn as_provider_string(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Json(value) => value.to_string(),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ToolOutput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

/// The result of executing a tool, sent back as a tool-role message.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: ToolOutput,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
        }
    }
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
    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self {
            role: Role::Tool,
            content: String::new(),
            parts: vec![MessagePart::ToolResult(ToolResult::new(
                tool_call_id,
                output,
            ))],
        }
    }

    /// Build a tool-result message containing multiple tool results.
    pub fn tool_results(results: impl IntoIterator<Item = ToolResult>) -> Self {
        let parts = results.into_iter().map(MessagePart::ToolResult).collect();

        Self {
            role: Role::Tool,
            content: String::new(),
            parts,
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

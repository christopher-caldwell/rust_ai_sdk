use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq)]
/// Provider-neutral chat-message role.
pub enum Role {
    /// Application-owned system instruction.
    System,
    /// End-user message.
    User,
    /// Model-generated message.
    Assistant,
    /// Result of an assistant tool call.
    Tool,
}

/// A single unit of content within a message.
#[derive(Debug, Clone, PartialEq)]
pub enum MessagePart {
    /// Plain text content.
    Text(String),
    /// Tool invocation requested by the assistant.
    ToolCall(ToolCall),
    /// Application-produced result for a prior tool call.
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
    /// Wrap raw provider-owned metadata.
    pub fn new(raw: Value) -> Self {
        Self { raw }
    }

    /// Borrow the complete raw metadata value.
    pub fn as_raw(&self) -> &Value {
        &self.raw
    }

    /// Consume the wrapper and return the raw value.
    pub fn into_raw(self) -> Value {
        self.raw
    }

    /// Select a provider namespace, falling back to the complete value.
    pub fn provider_value(&self, provider_key: &str) -> &Value {
        self.raw.get(provider_key).unwrap_or(&self.raw)
    }

    /// Read a string from a provider metadata namespace.
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
    /// Provider tool-call identifier used to match the result.
    pub id: String,
    /// Registered tool name.
    pub name: String,
    /// Parsed JSON arguments supplied by the model.
    pub input: Value,
    /// Provider-owned continuation metadata.
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ToolCall {
    /// Create a tool call with parsed JSON input.
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
            provider_metadata: None,
        }
    }

    /// Parse raw JSON arguments, preserving malformed input as metadata.
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

    /// Create a non-executable tool call that records malformed JSON input.
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

    /// Attach provider-owned metadata.
    #[must_use]
    pub fn with_provider_metadata(mut self, metadata: impl Into<ProviderMetadata>) -> Self {
        self.provider_metadata = Some(metadata.into());
        self
    }

    /// Return whether the provider supplied malformed JSON arguments.
    pub fn has_malformed_input(&self) -> bool {
        self.malformed_input().is_some()
    }

    /// Return the malformed raw argument string, when present.
    pub fn malformed_input_raw(&self) -> Option<&str> {
        self.malformed_input()
            .and_then(|metadata| metadata.get("raw"))
            .and_then(Value::as_str)
    }

    /// Return the provider JSON parse error, when present.
    pub fn malformed_input_error(&self) -> Option<&str> {
        self.malformed_input()
            .and_then(|metadata| metadata.get("error"))
            .and_then(Value::as_str)
    }

    /// Build the SDK metadata marker used for malformed JSON input.
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

    /// Merge the malformed-input marker into existing provider metadata.
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
    /// Plain-text tool output.
    Text(String),
    /// Structured JSON tool output.
    Json(Value),
}

impl ToolOutput {
    /// Render output in the string representation expected by provider APIs.
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
    /// Identifier of the tool call being resolved.
    pub tool_call_id: String,
    /// Application-produced tool output.
    pub output: ToolOutput,
}

impl ToolResult {
    /// Create a result for a prior tool-call identifier.
    pub fn new(tool_call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
        }
    }
}

/// A chat message with one canonical sequence of structured parts.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// Role that determines the message's provider semantics.
    pub role: Role,
    parts: Vec<MessagePart>,
}

impl Message {
    /// Create a message from an explicit role and part sequence.
    pub fn from_parts(role: Role, parts: Vec<MessagePart>) -> Self {
        Self { role, parts }
    }

    /// Create a plain-text user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![MessagePart::Text(text.into())],
        }
    }

    /// Create a plain-text assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![MessagePart::Text(text.into())],
        }
    }

    /// Create a trusted application system instruction.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            parts: vec![MessagePart::Text(text.into())],
        }
    }

    /// Build an assistant message from structured parts (e.g. after a tool turn).
    pub fn assistant_parts(parts: Vec<MessagePart>) -> Self {
        Self {
            role: Role::Assistant,
            parts,
        }
    }

    /// Build a tool-result message.
    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self {
            role: Role::Tool,
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
            parts,
        }
    }

    /// Borrow the canonical message parts.
    pub fn parts(&self) -> &[MessagePart] {
        &self.parts
    }

    /// Return the text when this is a single-part text message.
    pub fn text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [MessagePart::Text(text)] => Some(text),
            _ => None,
        }
    }

    /// Borrow the canonical parts used by provider translators.
    pub fn effective_parts(&self) -> &[MessagePart] {
        &self.parts
    }

    /// Return whether all message parts are text.
    pub fn is_text_only(&self) -> bool {
        self.parts
            .iter()
            .all(|part| matches!(part, MessagePart::Text(_)))
    }
}

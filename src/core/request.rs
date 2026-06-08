use std::collections::HashSet;

use super::tool::{ToolChoice, ToolDefinition};
use super::{
    error::SdkError,
    message::{Message, MessagePart, Role, ToolResult},
};

/// Provider-neutral request for text, chat, streaming, and tool-calling APIs.
///
/// Prefer [`TextRequest::builder`] for examples and public application code. The
/// fields remain public so advanced callers can inspect or adapt requests at API
/// boundaries.
#[derive(Debug, Clone, Default)]
pub struct TextRequest {
    pub messages: Vec<Message>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
}

impl TextRequest {
    /// Build a request from an existing message list.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            ..Self::default()
        }
    }

    /// Start a fluent request builder.
    pub fn builder() -> TextRequestBuilder {
        TextRequestBuilder::default()
    }

    /// Build a request containing one user message.
    pub fn prompt(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(prompt)],
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Validate request shape before sending it to a provider.
    pub fn validate(&self) -> Result<(), SdkError> {
        validate_request_has_messages(self)?;
        validate_generation_options(self)?;
        validate_tool_choice(self)?;
        validate_message_sequence(self)?;

        Ok(())
    }
}

/// Fluent builder for [`TextRequest`].
#[derive(Debug, Default)]
pub struct TextRequestBuilder {
    request: TextRequest,
}

impl TextRequestBuilder {
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    #[must_use]
    pub fn messages(mut self, messages: impl Into<Vec<Message>>) -> Self {
        self.request.messages = messages.into();
        self
    }

    #[must_use]
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.messages.push(Message::user(prompt));
        self
    }

    #[must_use]
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.request.messages.push(Message::system(text));
        self
    }

    #[must_use]
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.request.max_output_tokens = Some(tokens);
        self
    }

    #[must_use]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.request.tools = tools;
        self
    }

    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.request.tool_choice = Some(choice);
        self
    }

    #[must_use]
    pub fn build(self) -> TextRequest {
        self.request
    }

    pub fn try_build(self) -> Result<TextRequest, SdkError> {
        self.request.validate()?;

        Ok(self.request)
    }
}

fn validate_request_has_messages(request: &TextRequest) -> Result<(), SdkError> {
    if request.messages.is_empty() {
        return Err(SdkError::Validation(
            "TextRequest must contain at least one message".to_string(),
        ));
    }

    Ok(())
}

fn validate_generation_options(request: &TextRequest) -> Result<(), SdkError> {
    if matches!(request.max_output_tokens, Some(0)) {
        return Err(SdkError::Validation(
            "max_output_tokens must be greater than zero".to_string(),
        ));
    }

    if let Some(temperature) = request.temperature {
        let is_finite = temperature.is_finite();
        let is_in_range = (0.0..=2.0).contains(&temperature);

        if !is_finite || !is_in_range {
            return Err(SdkError::Validation(
                "temperature must be a finite value between 0.0 and 2.0".to_string(),
            ));
        }
    }

    Ok(())
}

fn validate_tool_choice(request: &TextRequest) -> Result<(), SdkError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };

    match tool_choice {
        ToolChoice::Auto => validate_auto_tool_choice(request),
        ToolChoice::None => Ok(()),
        ToolChoice::Required { name } => validate_required_tool_choice(request, name),
    }
}

fn validate_auto_tool_choice(request: &TextRequest) -> Result<(), SdkError> {
    if request.tools.is_empty() {
        return Err(SdkError::Validation(
            "tool_choice Auto requires at least one tool definition".to_string(),
        ));
    }

    Ok(())
}

fn validate_required_tool_choice(request: &TextRequest, name: &str) -> Result<(), SdkError> {
    if name.trim().is_empty() {
        return Err(SdkError::Validation(
            "tool_choice Required must name a tool".to_string(),
        ));
    }

    let required_tool_exists = request.tools.iter().any(|tool| tool.name == name);
    if !required_tool_exists {
        return Err(SdkError::Validation(format!(
            "tool_choice Required references unknown tool: {name}",
        )));
    }

    Ok(())
}

fn validate_message_sequence(request: &TextRequest) -> Result<(), SdkError> {
    let mut pending_tool_calls = HashSet::new();

    for (message_index, message) in request.messages.iter().enumerate() {
        validate_message_content_storage(message_index, message)?;
        validate_message_shape(message_index, message)?;
        validate_tool_results(message_index, message, &mut pending_tool_calls)?;
        collect_assistant_tool_calls(message_index, message, &mut pending_tool_calls)?;
    }

    Ok(())
}

fn validate_message_content_storage(
    message_index: usize,
    message: &Message,
) -> Result<(), SdkError> {
    let has_content = !message.content.is_empty();
    let has_parts = !message.parts.is_empty();

    if has_content && has_parts {
        return Err(SdkError::Validation(format!(
            "message {message_index} has both content and parts; use one content representation",
        )));
    }

    if !has_content && !has_parts {
        return Err(SdkError::Validation(format!(
            "message {message_index} is empty",
        )));
    }

    Ok(())
}

fn validate_message_shape(message_index: usize, message: &Message) -> Result<(), SdkError> {
    match message.role {
        Role::System => validate_system_message(message_index, message),
        Role::User => validate_user_message(message_index, message),
        Role::Assistant => validate_assistant_message(message_index, message),
        Role::Tool => validate_tool_message(message_index, message),
    }
}

fn validate_system_message(message_index: usize, message: &Message) -> Result<(), SdkError> {
    for part in message.effective_parts() {
        if !matches!(part, MessagePart::Text(_)) {
            return Err(SdkError::Validation(format!(
                "message {message_index} has role System but contains non-text parts",
            )));
        }
    }

    Ok(())
}

fn validate_user_message(message_index: usize, message: &Message) -> Result<(), SdkError> {
    for part in message.effective_parts() {
        if !matches!(part, MessagePart::Text(_)) {
            return Err(SdkError::Validation(format!(
                "message {message_index} has role User but contains non-text parts",
            )));
        }
    }

    Ok(())
}

fn validate_assistant_message(message_index: usize, message: &Message) -> Result<(), SdkError> {
    for part in message.effective_parts() {
        if matches!(part, MessagePart::ToolResult(_)) {
            return Err(SdkError::Validation(format!(
                "message {message_index} has role Assistant but contains a tool result",
            )));
        }
    }

    Ok(())
}

fn validate_tool_message(message_index: usize, message: &Message) -> Result<(), SdkError> {
    if message.parts.is_empty() {
        return Err(SdkError::Validation(format!(
            "message {message_index} has role Tool but no tool result parts",
        )));
    }

    for part in &message.parts {
        if !matches!(part, MessagePart::ToolResult(_)) {
            return Err(SdkError::Validation(format!(
                "message {message_index} has role Tool but contains non-tool-result parts",
            )));
        }
    }

    Ok(())
}

fn validate_tool_results(
    message_index: usize,
    message: &Message,
    pending_tool_calls: &mut HashSet<String>,
) -> Result<(), SdkError> {
    for part in &message.parts {
        let MessagePart::ToolResult(result) = part else {
            continue;
        };

        validate_tool_result_id(message_index, result)?;

        let has_prior_tool_call = pending_tool_calls.remove(&result.tool_call_id);
        if !has_prior_tool_call {
            return Err(SdkError::Validation(format!(
                "message {message_index} references unknown or already-used tool call id: {}",
                result.tool_call_id,
            )));
        }
    }

    Ok(())
}

fn validate_tool_result_id(message_index: usize, result: &ToolResult) -> Result<(), SdkError> {
    if result.tool_call_id.trim().is_empty() {
        return Err(SdkError::Validation(format!(
            "message {message_index} contains a tool result without a tool_call_id",
        )));
    }

    Ok(())
}

fn collect_assistant_tool_calls(
    message_index: usize,
    message: &Message,
    pending_tool_calls: &mut HashSet<String>,
) -> Result<(), SdkError> {
    for part in &message.parts {
        let MessagePart::ToolCall(call) = part else {
            continue;
        };

        if call.id.trim().is_empty() {
            return Err(SdkError::Validation(format!(
                "message {message_index} contains a tool call without an id",
            )));
        }

        if call.name.trim().is_empty() {
            return Err(SdkError::Validation(format!(
                "message {message_index} contains a tool call without a name",
            )));
        }

        let is_new_tool_call = pending_tool_calls.insert(call.id.clone());
        if !is_new_tool_call {
            return Err(SdkError::Validation(format!(
                "message {message_index} duplicates pending tool call id: {}",
                call.id,
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builder_sets_messages_and_options() {
        let request = TextRequest::builder()
            .system("be concise")
            .prompt("hello")
            .max_output_tokens(128)
            .temperature(0.2)
            .tools(vec![ToolDefinition::new(
                "lookup",
                "Look something up",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Auto)
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.max_output_tokens, Some(128));
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.tools.len(), 1);
        assert!(matches!(request.tool_choice, Some(ToolChoice::Auto)));
    }

    #[test]
    fn validation_rejects_empty_requests() {
        let error = TextRequest::default().validate().unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("at least one message")
        ));
    }

    #[test]
    fn validation_rejects_message_with_content_and_parts() {
        let request = TextRequest::new(vec![Message {
            role: Role::User,
            content: "hello".to_string(),
            parts: vec![MessagePart::Text("world".to_string())],
        }]);

        let error = request.validate().unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("both content and parts")
        ));
    }

    #[test]
    fn validation_rejects_tool_result_on_user_role() {
        let request = TextRequest::new(vec![Message {
            role: Role::User,
            content: String::new(),
            parts: vec![MessagePart::ToolResult(ToolResult::new(
                "call_1",
                json!({"ok": true}),
            ))],
        }]);

        let error = request.validate().unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("role User")
        ));
    }

    #[test]
    fn validation_accepts_structured_tool_result_continuation() {
        let request = TextRequest::new(vec![
            Message::user("weather"),
            Message::assistant_parts(vec![MessagePart::ToolCall(
                crate::core::message::ToolCall::new(
                    "call_1",
                    "get_weather",
                    json!({"location": "Paris"}),
                ),
            )]),
            Message::tool_result("call_1", json!({"forecast": "cloudy"})),
        ]);

        request.validate().unwrap();
    }

    #[test]
    fn validation_rejects_tool_result_without_prior_tool_call() {
        let request = TextRequest::new(vec![Message::tool_result(
            "call_1",
            json!({"forecast": "cloudy"}),
        )]);

        let error = request.validate().unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("unknown or already-used")
        ));
    }

    #[test]
    fn validation_rejects_required_tool_choice_for_unknown_tool() {
        let request = TextRequest::prompt("hello").with_tool_choice(ToolChoice::required("lookup"));

        let error = request.validate().unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("unknown tool")
        ));
    }
}

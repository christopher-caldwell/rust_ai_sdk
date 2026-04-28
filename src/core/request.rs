use super::message::Message;
use super::tool::{ToolChoice, ToolDefinition};

#[derive(Debug, Clone)]
pub struct TextRequest {
    pub messages: Vec<Message>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
}

impl TextRequest {
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            ..Self::default()
        }
    }

    pub fn builder() -> TextRequestBuilder {
        TextRequestBuilder::default()
    }

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
}

impl Default for TextRequest {
    fn default() -> Self {
        Self {
            messages: vec![],
            max_output_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
        }
    }
}

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

    pub fn build(self) -> TextRequest {
        self.request
    }
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
}

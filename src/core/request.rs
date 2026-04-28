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
    pub fn prompt(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(prompt)],
            max_output_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }
}

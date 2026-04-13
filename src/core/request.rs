use super::message::Message;

#[derive(Debug, Clone)]
pub struct TextRequest {
    pub messages: Vec<Message>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl TextRequest {
    pub fn prompt(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message {
                role: super::message::Role::User,
                content: prompt.into(),
            }],
            max_output_tokens: None,
            temperature: None,
        }
    }
}

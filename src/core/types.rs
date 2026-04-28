#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolUse,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseMetadata {
    pub id: Option<String>,
    pub model: Option<String>,
}

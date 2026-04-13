#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ResponseMetadata {
    pub id: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Provider-reported token counts.
pub struct Usage {
    /// Input or prompt tokens, when reported.
    pub input_tokens: Option<u32>,
    /// Generated output tokens, when reported.
    pub output_tokens: Option<u32>,
    /// Total tokens, when reported or safely computed.
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Provider-neutral reason that generation ended.
pub enum FinishReason {
    /// Normal completion.
    Stop,
    /// Configured or provider output limit reached.
    Length,
    /// Provider safety or content policy stopped generation.
    ContentFilter,
    /// The assistant requested one or more tools.
    ToolUse,
    /// Provider-specific or unknown finish reason.
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Stable metadata identifying a provider response.
pub struct ResponseMetadata {
    /// Provider response identifier, when available.
    pub id: Option<String>,
    /// Model identifier reported by the provider, when available.
    pub model: Option<String>,
}

/// Known Anthropic model IDs as of the library release.
///
/// This list is opt-in. Pass any `&str` or `String` directly to
/// [`AnthropicChatModel::new`](super::AnthropicChatModel::new) to use a model not listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicModel {
    // ── Opus ──────────────────────────────────────
    /// Claude Opus 4.6.
    Opus4_6,
    /// Claude Opus 4.5.
    Opus4_5,
    /// Claude Opus 4.1.
    Opus4_1,
    // ── Sonnet ──────────────────────────────────────
    /// Claude Sonnet 4.6.
    Sonnet4_6,
    /// Claude Sonnet 4.5.
    Sonnet4_5,
    /// Claude Sonnet 4.0.
    Sonnet4_0,
    // ── Haiku ──────────────────────────────────────
    /// Claude Haiku 4.5.
    Haiku4_5,
}

impl AnthropicModel {
    /// Return the provider model identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            // Opus
            Self::Opus4_6 => "claude-opus-4-6",
            Self::Opus4_5 => "claude-opus-4-5",
            Self::Opus4_1 => "claude-opus-4-1",
            // Sonnet
            Self::Sonnet4_6 => "claude-sonnet-4-6",
            Self::Sonnet4_5 => "claude-sonnet-4-5",
            Self::Sonnet4_0 => "claude-sonnet-4-0",
            // Haiku
            Self::Haiku4_5 => "claude-haiku-4-5",
        }
    }
}

impl From<AnthropicModel> for String {
    fn from(m: AnthropicModel) -> String {
        m.as_str().to_string()
    }
}

impl std::fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Known OpenAI model IDs as of the library release.
///
/// This list is opt-in. Pass any `&str` or `String` directly to
/// [`OpenAiChatModel::new`] to use a model not listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiModel {
    // ── GPT-5.4 ─────────────────────────────────────────
    Gpt5_4,
    Gpt5_4Mini,
    Gpt5_4Nano,
    Gpt5_4Pro,
    // ── GPT-5.2 ─────────────────────────────────────────
    Gpt5_2,
    Gpt5_2Pro,
    // ── GPT-5.1 ─────────────────────────────────────────
    Gpt5_1,
    // ── GPT-5 ───────────────────────────────────────────
    Gpt5,
    Gpt5Mini,
    Gpt5Nano,
    Gpt5Pro,
    // ── GPT-4.1 ─────────────────────────────────────────
    Gpt4_1,
    Gpt4_1Mini,
    Gpt4_1Nano,
    // ── GPT-4o ──────────────────────────────────────────
    Gpt4o,
}

impl OpenAiModel {
    pub fn as_str(self) -> &'static str {
        match self {
            // GPT-5.4
            Self::Gpt5_4     => "gpt-5.4",
            Self::Gpt5_4Mini => "gpt-5.4-mini",
            Self::Gpt5_4Nano => "gpt-5.4-nano",
            Self::Gpt5_4Pro  => "gpt-5.4-pro",
            // GPT-5.2
            Self::Gpt5_2     => "gpt-5.2",
            Self::Gpt5_2Pro  => "gpt-5.2-pro",
            // GPT-5.1
            Self::Gpt5_1     => "gpt-5.1",
            // GPT-5
            Self::Gpt5       => "gpt-5",
            Self::Gpt5Mini   => "gpt-5-mini",
            Self::Gpt5Nano   => "gpt-5-nano",
            Self::Gpt5Pro    => "gpt-5-pro",
            // GPT-4.1
            Self::Gpt4_1     => "gpt-4.1",
            Self::Gpt4_1Mini => "gpt-4.1-mini",
            Self::Gpt4_1Nano => "gpt-4.1-nano",
            // GPT-4o
            Self::Gpt4o      => "gpt-4o",
        }
    }
}

impl From<OpenAiModel> for String {
    fn from(m: OpenAiModel) -> String {
        m.as_str().to_string()
    }
}

impl std::fmt::Display for OpenAiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Known Gemini model IDs as of the library release.
///
/// This list is opt-in. Pass any `&str` or `String` directly to
/// [`GeminiChatModel::new`] to use a model not listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiModel {
    // Gemini 3 preview text/function-capable models.
    Gemini3_1ProPreview,
    Gemini3_1ProPreviewCustomTools,
    Gemini3FlashPreview,
    Gemini3_1FlashLitePreview,
    // Stable Gemini 2.5 text/function-capable models.
    Gemini2_5Pro,
    Gemini2_5Flash,
    Gemini2_5FlashLite,
}

impl GeminiModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemini3_1ProPreview => "gemini-3.1-pro-preview",
            Self::Gemini3_1ProPreviewCustomTools => "gemini-3.1-pro-preview-customtools",
            Self::Gemini3FlashPreview => "gemini-3-flash-preview",
            Self::Gemini3_1FlashLitePreview => "gemini-3.1-flash-lite-preview",
            Self::Gemini2_5Pro => "gemini-2.5-pro",
            Self::Gemini2_5Flash => "gemini-2.5-flash",
            Self::Gemini2_5FlashLite => "gemini-2.5-flash-lite",
        }
    }
}

impl From<GeminiModel> for String {
    fn from(m: GeminiModel) -> String {
        m.as_str().to_string()
    }
}

impl std::fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

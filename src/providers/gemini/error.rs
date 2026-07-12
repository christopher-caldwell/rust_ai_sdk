use crate::core::error::SdkError;
pub(super) use crate::providers::transport::truncate_body;

#[derive(Debug)]
pub(super) enum GeminiClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<GeminiClientError> for SdkError {
    fn from(value: GeminiClientError) -> Self {
        match value {
            GeminiClientError::Reqwest(e) => {
                crate::providers::transport::reqwest_error("Gemini", e)
            }
            GeminiClientError::Serde(e) => SdkError::serialization(Some("Gemini"), e.to_string()),
        }
    }
}

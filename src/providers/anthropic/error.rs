use crate::core::error::SdkError;
pub(super) use crate::providers::transport::truncate_body;

#[derive(Debug)]
pub(super) enum AnthropicClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<AnthropicClientError> for SdkError {
    fn from(value: AnthropicClientError) -> Self {
        match value {
            AnthropicClientError::Reqwest(e) => {
                crate::providers::transport::reqwest_error("Anthropic", e)
            }
            AnthropicClientError::Serde(e) => {
                SdkError::serialization(Some("Anthropic"), e.to_string())
            }
        }
    }
}

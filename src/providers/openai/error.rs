use crate::core::error::SdkError;
pub(super) use crate::providers::transport::truncate_body;

#[derive(Debug)]
pub(super) enum OpenAiClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<OpenAiClientError> for SdkError {
    fn from(value: OpenAiClientError) -> Self {
        match value {
            OpenAiClientError::Reqwest(e) => {
                crate::providers::transport::reqwest_error("OpenAI", e)
            }
            OpenAiClientError::Serde(e) => SdkError::serialization(Some("OpenAI"), e.to_string()),
        }
    }
}

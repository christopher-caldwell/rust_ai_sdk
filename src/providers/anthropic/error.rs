use crate::core::error::SdkError;

#[derive(Debug)]
pub(super) enum AnthropicClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<AnthropicClientError> for SdkError {
    fn from(value: AnthropicClientError) -> Self {
        match value {
            AnthropicClientError::Reqwest(e) => SdkError::Http(e.to_string()),
            AnthropicClientError::Serde(e) => SdkError::Serialization(e.to_string()),
        }
    }
}

pub(super) fn truncate_body(body: &str, max_bytes: usize) -> String {
    if body.len() <= max_bytes {
        return body.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &body[..end])
}

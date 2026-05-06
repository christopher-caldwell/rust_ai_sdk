use crate::core::error::SdkError;

#[derive(Debug)]
pub(super) enum GeminiClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<GeminiClientError> for SdkError {
    fn from(value: GeminiClientError) -> Self {
        match value {
            GeminiClientError::Reqwest(e) => SdkError::Http(e.to_string()),
            GeminiClientError::Serde(e) => SdkError::Serialization(e.to_string()),
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
    format!("{}... (truncated)", &body[..end])
}

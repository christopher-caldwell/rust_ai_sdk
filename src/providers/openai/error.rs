use crate::core::error::SdkError;

#[derive(Debug)]
pub(super) enum OpenAiClientError {
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
}

impl From<OpenAiClientError> for SdkError {
    fn from(value: OpenAiClientError) -> Self {
        match value {
            OpenAiClientError::Reqwest(e) => SdkError::provider_http(
                "OpenAI",
                e.status().map(|status| status.as_u16()),
                e.to_string(),
                None,
            ),
            OpenAiClientError::Serde(e) => SdkError::serialization(Some("OpenAI"), e.to_string()),
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

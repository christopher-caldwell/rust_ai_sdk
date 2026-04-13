use crate::core::{error::SdkError, request::TextRequest, result::TextResult};

use super::{
    error::{OpenAiClientError, truncate_body},
    types::{
        ChatCompletionResponse, OpenAiErrorBody, chat_response_to_text_result,
        text_request_to_openai,
    },
};

const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

pub struct OpenAiClient {
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            http: reqwest::Client::new(),
        }
    }

    pub async fn generate(
        &self,
        model: &str,
        request: TextRequest,
    ) -> Result<TextResult, SdkError> {
        let body = text_request_to_openai(model, request);

        let response = self
            .http
            .post(CHAT_COMPLETIONS_URL)
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);
            if let Ok(err) = serde_json::from_slice::<OpenAiErrorBody>(&bytes) {
                return Err(SdkError::Api(format!(
                    "{} (HTTP {})",
                    err.error.message, status
                )));
            }
            return Err(SdkError::Http(format!("HTTP {}: {}", status, snippet)));
        }

        let parsed: ChatCompletionResponse = serde_json::from_slice(&bytes)
            .map_err(|e| SdkError::from(OpenAiClientError::Serde(e)))?;
        chat_response_to_text_result(parsed)
    }
}

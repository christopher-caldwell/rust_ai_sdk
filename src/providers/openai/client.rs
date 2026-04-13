use crate::core::{error::SdkError, request::TextRequest, result::TextResult};

use super::{
    error::{OpenAiClientError, truncate_body},
    types::{
        ChatCompletionResponse, OpenAiErrorBody, chat_response_to_text_result,
        text_request_to_openai,
    },
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

pub struct OpenAiClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            http: reqwest::Client::new(),
        }
    }

    pub async fn generate(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextResult, SdkError> {
        let body = text_request_to_openai(model, request);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .http
            .post(&url)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::{Message, Role};
    use mockito;
    use serde_json::json;

    fn test_request() -> TextRequest {
        TextRequest {
            messages: vec![Message {
                role: Role::User,
                content: "Hello!".to_string(),
            }],
            max_output_tokens: Some(10),
            temperature: Some(0.7),
        }
    }

    #[tokio::test]
    async fn test_generate_success() {
        let mut server = mockito::Server::new_async().await;
        
        let mock_response = json!({
            "id": "req_123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let mock = server.mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-api-key")
            .match_body(mockito::Matcher::Json(json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.7
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response.to_string())
            .create_async().await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("gpt-4", &test_request()).await.expect("Request should succeed");
        
        mock.assert_async().await;
        
        assert_eq!(result.text, "Hi there!");
        assert_eq!(result.response.model.as_deref(), Some("gpt-4"));
    }

    #[tokio::test]
    async fn test_generate_error_json() {
        let mut server = mockito::Server::new_async().await;
        
        let mock_error = json!({
            "error": {
                "message": "Invalid API key.",
                "type": "invalid_request_error"
            }
        });

        let mock = server.mock("POST", "/chat/completions")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(mock_error.to_string())
            .create_async().await;

        let client = OpenAiClient::with_base_url("invalid-api-key".to_string(), server.url());
        let result = client.generate("gpt-4", &test_request()).await;
        
        mock.assert_async().await;
        
        match result {
            Err(SdkError::Api(msg)) => assert!(msg.contains("Invalid API key.") && msg.contains("HTTP 401")),
            _ => panic!("Expected Api error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_generate_error_non_json() {
        let mut server = mockito::Server::new_async().await;

        let mock = server.mock("POST", "/chat/completions")
            .with_status(502)
            .with_header("content-type", "text/plain")
            .with_body("Bad Gateway Timeout Exception...")
            .create_async().await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("gpt-4", &test_request()).await;
        
        mock.assert_async().await;
        
        match result {
            Err(SdkError::Http(msg)) => assert!(msg.contains("Bad Gateway Timeout Exception...") && msg.contains("HTTP 502")),
            _ => panic!("Expected Http error, got {:?}", result),
        }
    }
}

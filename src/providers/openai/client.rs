use crate::core::{error::SdkError, request::TextRequest, result::TextResult};

use super::{
    error::{OpenAiClientError, truncate_body},
    types::{
        ChatCompletionResponse, ChatCompletionChunk, OpenAiErrorBody, chat_response_to_text_result,
        text_request_to_openai, map_finish_reason,
    },
};
use crate::core::stream::{StreamEvent, TextEventStream};
use crate::core::types::{FinishReason, ResponseMetadata, Usage as TokenUsage};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
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

    #[allow(dead_code)] // used in #[cfg(test)] blocks
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
        let body = text_request_to_openai(model, request, false);
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

    pub async fn stream(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        let body = text_request_to_openai(model, request, true);
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
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))?;
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

        // Track whether a Finished event has been emitted. This lets us emit
        // a fallback Finished at [DONE] in case OpenAI ever sends finish_reason
        // only on the sentinel instead of a prior delta chunk.
        let finished_emitted = Arc::new(AtomicBool::new(false));
        let fe = finished_emitted.clone();

        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) => {
                    if event.data == "[DONE]" {
                        // Emit a fallback Finished if the stream closed without one.
                        if !fe.load(Ordering::Relaxed) {
                            fe.store(true, Ordering::Relaxed);
                            return vec![Ok(StreamEvent::Finished {
                                finish_reason: FinishReason::Other("unknown".to_string()),
                                usage: None,
                                response: ResponseMetadata { id: None, model: None },
                            })];
                        }
                        return vec![];
                    }
                    match serde_json::from_str::<ChatCompletionChunk>(&event.data) {
                        Ok(chunk) => {
                            let mut events = Vec::new();
                            // Collect text deltas from all choices first.
                            let mut chunk_finish_reason: Option<String> = None;
                            for choice in &chunk.choices {
                                if let Some(content) = &choice.delta.content {
                                    events.push(Ok(StreamEvent::TextDelta(content.clone())));
                                }
                                // Use the first finish_reason found; ignore subsequent
                                // choices (n>1 is unsupported in this milestone).
                                if chunk_finish_reason.is_none() {
                                    chunk_finish_reason = choice.finish_reason.clone();
                                }
                            }
                            // Construct one Finished per chunk, not one per choice.
                            // Usage belongs to the chunk, not to any individual choice.
                            if let Some(reason) = chunk_finish_reason {
                                let usage = chunk.usage.as_ref().map(|u| TokenUsage {
                                    input_tokens: u.prompt_tokens,
                                    output_tokens: u.completion_tokens,
                                    total_tokens: u.total_tokens,
                                });
                                events.push(Ok(StreamEvent::Finished {
                                    finish_reason: map_finish_reason(&reason),
                                    usage,
                                    response: ResponseMetadata {
                                        id: chunk.id.clone(),
                                        model: chunk.model.clone(),
                                    },
                                }));
                                fe.store(true, Ordering::Relaxed);
                            }
                            events
                        }
                        Err(e) => vec![Err(SdkError::from(OpenAiClientError::Serde(e)))],
                    }
                }
                Err(e) => vec![Err(SdkError::Api(format!("EventSource stream error: {}", e)))],
            })
            .flat_map(futures_util::stream::iter);

        Ok(Box::pin(stream))
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

    #[tokio::test]
    async fn test_stream_success() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = "data: {\"id\":\"req_123\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
                         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n";

        let mock = server.mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-api-key")
            .match_body(mockito::Matcher::Json(json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.7,
                "stream": true
            })))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async().await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client.stream("gpt-4", &test_request()).await.expect("Stream should start");
        
        mock.assert_async().await;
        
        let mut text = String::new();
        let mut finished = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Finished { finish_reason, .. } => {
                    assert!(matches!(finish_reason, crate::core::types::FinishReason::Stop));
                    finished = true;
                }
            }
        }
        assert_eq!(text, "Hello");
        assert!(finished);
    }

    #[tokio::test]
    async fn test_stream_malformed_event() {
        let mut server = mockito::Server::new_async().await;
        let mock_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                         data: {bad json\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async().await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4", &test_request()).await.expect("Stream starts");
        
        let evt1 = stream.next().await.unwrap().unwrap();
        match evt1 {
            StreamEvent::TextDelta(t) => assert_eq!(t, "Hi"),
            _ => panic!("Expected text"),
        }
        
        let evt2 = stream.next().await.unwrap();
        assert!(matches!(evt2, Err(_)));
    }

    #[tokio::test]
    async fn test_stream_setup_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server.mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("Unauthorized")
            .create_async().await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let result = client.stream("gpt-4", &test_request()).await;

        mock.assert_async().await;

        // Must be a direct Err, not hidden inside a stream item.
        assert!(matches!(result, Err(SdkError::Http(_))));
    }
}

use crate::core::{error::SdkError, request::TextRequest, result::TextResult};

use super::{
    error::{AnthropicClientError, truncate_body},
    types::{
        AnthropicErrorResponse, AnthropicResponse,
        ContentBlockDeltaEvent, ErrorEvent, EventEnvelope,
        MessageDeltaEvent, MessageStartEvent,
        anthropic_response_to_text_result, text_request_to_anthropic, map_stop_reason,
    },
};
use crate::core::stream::{StreamEvent, TextEventStream};
use crate::core::types::{FinishReason, ResponseMetadata, Usage as TokenUsage};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl AnthropicClient {
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
        let body = text_request_to_anthropic(model, request, false);
        let url = format!("{}/messages", self.base_url);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes);
            let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);
            if let Ok(err) = serde_json::from_slice::<AnthropicErrorResponse>(&bytes) {
                return Err(SdkError::Api(format!(
                    "{} (HTTP {})",
                    err.error.message, status
                )));
            }
            return Err(SdkError::Http(format!("HTTP {}: {}", status, snippet)));
        }

        let parsed: AnthropicResponse = serde_json::from_slice(&bytes)
            .map_err(|e| SdkError::from(AnthropicClientError::Serde(e)))?;
        anthropic_response_to_text_result(parsed)
    }

    pub async fn stream(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        let body = text_request_to_anthropic(model, request, true);
        let url = format!("{}/messages", self.base_url);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;

        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;
            let text = String::from_utf8_lossy(&bytes);
            let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);
            if let Ok(err) = serde_json::from_slice::<AnthropicErrorResponse>(&bytes) {
                return Err(SdkError::Api(format!(
                    "{} (HTTP {})",
                    err.error.message, status
                )));
            }
            return Err(SdkError::Http(format!("HTTP {}: {}", status, snippet)));
        }

        let mut acc = StreamAccumulator::default();

        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) => process_event(&event.data, &mut acc),
                Err(e) => vec![Err(SdkError::Api(format!("EventSource stream error: {}", e)))],
            })
            .flat_map(futures_util::stream::iter);

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Internal streaming helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    finish_reason: Option<FinishReason>,
}

fn process_event(
    data: &str,
    acc: &mut StreamAccumulator,
) -> Vec<Result<StreamEvent, SdkError>> {
    // Step 1: read just the type tag — fail only on total JSON chaos.
    let envelope: EventEnvelope = match serde_json::from_str(data) {
        Ok(e) => e,
        Err(e) => return vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
    };

    match envelope.event_type.as_str() {
        "message_start" => {
            match serde_json::from_str::<MessageStartEvent>(data) {
                Ok(evt) => {
                    acc.id = evt.message.id;
                    acc.model = evt.message.model;
                    if let Some(usage) = evt.message.usage {
                        acc.input_tokens = usage.input_tokens;
                    }
                    vec![]
                }
                Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
            }
        }

        "content_block_delta" => {
            match serde_json::from_str::<ContentBlockDeltaEvent>(data) {
                Ok(evt) => {
                    if let Some(text) = evt.delta.text {
                        if !text.is_empty() {
                            return vec![Ok(StreamEvent::TextDelta(text))];
                        }
                    }
                    vec![]
                }
                Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
            }
        }

        "message_delta" => {
            match serde_json::from_str::<MessageDeltaEvent>(data) {
                Ok(evt) => {
                    if let Some(reason) = &evt.delta.stop_reason {
                        acc.finish_reason = Some(map_stop_reason(reason));
                    }
                    if let Some(usage) = evt.usage {
                        acc.output_tokens = usage.output_tokens;
                    }
                    vec![]
                }
                Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
            }
        }

        "message_stop" => {
            let total_tokens = match (acc.input_tokens, acc.output_tokens) {
                (Some(i), Some(o)) => Some(i + o),
                (Some(i), None) => Some(i),
                (None, Some(o)) => Some(o),
                (None, None) => None,
            };

            let usage = if acc.input_tokens.is_some() || acc.output_tokens.is_some() {
                Some(TokenUsage {
                    input_tokens: acc.input_tokens,
                    output_tokens: acc.output_tokens,
                    total_tokens,
                })
            } else {
                None
            };

            vec![Ok(StreamEvent::Finished {
                finish_reason: acc
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| FinishReason::Other("unknown".to_string())),
                usage,
                response: ResponseMetadata {
                    id: acc.id.clone(),
                    model: acc.model.clone(),
                },
            })]
        }

        "error" => {
            match serde_json::from_str::<ErrorEvent>(data) {
                Ok(evt) => vec![Err(SdkError::Api(evt.error.message))],
                Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
            }
        }

        // All other event types (ping, content_block_start, content_block_stop,
        // or any future events Anthropic may add) are silently ignored.
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::{Message, Role};
    use crate::core::types::FinishReason;
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

    // ------------------------------------------------------------------
    // Non-streaming tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_generate_success() {
        let mut server = mockito::Server::new_async().await;
        
        let mock_response = json!({
            "id": "msg_123",
            "model": "claude-3-5-sonnet",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hi there!"
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });

        let mock = server.mock("POST", "/messages")
            .match_header("x-api-key", "test-api-key")
            .match_header("anthropic-version", "2023-06-01")
            .match_body(mockito::Matcher::Json(json!({
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.7
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response.to_string())
            .create_async().await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("claude-3-5-sonnet", &test_request()).await.expect("Request should succeed");
        
        mock.assert_async().await;
        
        assert_eq!(result.text, "Hi there!");
        assert_eq!(result.response.model.as_deref(), Some("claude-3-5-sonnet"));
    }

    #[tokio::test]
    async fn test_generate_error_json() {
        let mut server = mockito::Server::new_async().await;
        
        let mock_error = json!({
            "error": {
                "message": "Invalid API key.",
                "type": "authentication_error"
            }
        });

        let mock = server.mock("POST", "/messages")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(mock_error.to_string())
            .create_async().await;

        let client = AnthropicClient::with_base_url("invalid-api-key".to_string(), server.url());
        let result = client.generate("claude-3-5-sonnet", &test_request()).await;
        
        mock.assert_async().await;
        
        match result {
            Err(SdkError::Api(msg)) => assert!(msg.contains("Invalid API key.") && msg.contains("HTTP 401")),
            _ => panic!("Expected Api error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_generate_error_non_json() {
        let mut server = mockito::Server::new_async().await;

        let mock = server.mock("POST", "/messages")
            .with_status(502)
            .with_header("content-type", "text/plain")
            .with_body("Bad Gateway Timeout Exception...")
            .create_async().await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("claude-3-5-sonnet", &test_request()).await;
        
        mock.assert_async().await;
        
        match result {
            Err(SdkError::Http(msg)) => assert!(msg.contains("Bad Gateway Timeout Exception...") && msg.contains("HTTP 502")),
            _ => panic!("Expected Http error, got {:?}", result),
        }
    }

    // ------------------------------------------------------------------
    // Streaming unit tests — process_event
    // ------------------------------------------------------------------

    /// Unknown event types must be silently ignored (no output, no error).
    #[test]
    fn test_process_event_unknown_type_ignored() {
        let mut acc = StreamAccumulator::default();
        // "future_event" is not a recognized type
        let data = r#"{"type":"future_event","some_field":"some_value"}"#;
        let result = process_event(data, &mut acc);
        assert!(result.is_empty(), "Unknown event should produce no output");
    }

    /// A recognized event type with malformed payload must produce exactly one
    /// `SdkError::Serde` error.
    #[test]
    fn test_process_event_malformed_known_type_errors() {
        let mut acc = StreamAccumulator::default();
        // `content_block_delta` recognized, but `delta` field is missing
        let data = r#"{"type":"content_block_delta","index":0}"#;
        let result = process_event(data, &mut acc);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(&result[0], Err(SdkError::Serialization(_))),
            "Expected SdkError::Serde, got {:?}",
            result[0]
        );
    }

    /// `ping` and `content_block_start` / `content_block_stop` should also be
    /// silently ignored (they are well-known Anthropic event types that we don't
    /// need to process, but must not error on).
    #[test]
    fn test_process_event_known_passthrough_types_ignored() {
        let mut acc = StreamAccumulator::default();
        for data in [
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
        ] {
            let result = process_event(data, &mut acc);
            assert!(result.is_empty(), "Event {:?} should produce no output", data);
        }
    }

    // ------------------------------------------------------------------
    // Streaming integration tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_stream_setup_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server.mock("POST", "/messages")
            .with_status(401)
            .with_body("Unauthorized")
            .create_async().await;

        let client = AnthropicClient::with_base_url("test".to_string(), server.url());
        let result = client.stream("claude", &test_request()).await;

        mock.assert_async().await;

        assert!(matches!(result, Err(SdkError::Http(_))));
    }

    #[tokio::test]
    async fn test_stream_success() {
        let mut server = mockito::Server::new_async().await;

        // The SSE body includes a `ping` event and a `content_block_start` to
        // verify that unhandled-but-known events are silently skipped.
        let mock_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":10}}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        );

        let mock = server.mock("POST", "/messages")
            .match_header("x-api-key", "test-api-key")
            .match_body(mockito::Matcher::Json(json!({
                "model": "claude",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.7,
                "stream": true
            })))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async().await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client.stream("claude", &test_request()).await.expect("Stream should start");
        
        mock.assert_async().await;
        
        let mut text = String::new();
        let mut finished = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Finished { finish_reason, usage, response } => {
                    assert!(matches!(finish_reason, FinishReason::Stop));
                    assert_eq!(response.id.as_deref(), Some("msg_123"));
                    assert_eq!(response.model.as_deref(), Some("claude-3-5-sonnet"));
                    let u = usage.expect("usage should be present");
                    assert_eq!(u.input_tokens, Some(10));
                    assert_eq!(u.output_tokens, Some(5));
                    assert_eq!(u.total_tokens, Some(15));
                    finished = true;
                }
            }
        }
        assert_eq!(text, "Hello");
        assert!(finished);
    }
}

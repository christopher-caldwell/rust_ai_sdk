#[cfg(feature = "streaming")]
use std::collections::HashMap;

use serde::de::DeserializeOwned;

#[cfg(feature = "streaming")]
use crate::core::message::ToolCall;
#[cfg(feature = "streaming")]
use crate::core::provider_policy::finish_reason_with_tool_override;
use crate::core::{
    error::SdkError,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

#[cfg(feature = "streaming")]
use super::types::{
    ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent, ContentBlockStopEvent,
    ErrorEvent, EventEnvelope, MessageDeltaEvent, MessageStartEvent, map_stop_reason,
};
use super::{
    super::ProviderHttpConfig,
    error::{AnthropicClientError, truncate_body},
    types::{
        AnthropicErrorResponse, AnthropicResponse, anthropic_response_to_chat_result,
        anthropic_response_to_text_result, text_request_to_anthropic,
    },
};
#[cfg(feature = "streaming")]
use crate::core::stream::StreamEvent;
#[cfg(feature = "streaming")]
use crate::core::stream::TextEventStream;
#[cfg(feature = "streaming")]
use crate::core::types::{FinishReason, ResponseMetadata, Usage as TokenUsage};
use crate::providers::transport::read_bounded_body;
#[cfg(feature = "streaming")]
use eventsource_stream::Eventsource;
#[cfg(feature = "streaming")]
use futures_util::StreamExt;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

#[derive(Clone)]
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Result<Self, SdkError> {
        Self::with_config(api_key, ProviderHttpConfig::new())
    }

    pub(crate) fn with_config(
        api_key: String,
        config: ProviderHttpConfig,
    ) -> Result<Self, SdkError> {
        let (http, base_url) = config.resolve("Anthropic", DEFAULT_BASE_URL)?;
        Ok(Self {
            api_key,
            base_url,
            http,
        })
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
        request.validate()?;

        let body = text_request_to_anthropic(model, request, false);
        let url = format!("{}/messages", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&body),
        )
        .await?;

        let bytes = decode_response_bytes(response).await?;
        let parsed: AnthropicResponse = decode_json_response(&bytes)?;
        anthropic_response_to_text_result(parsed)
    }

    pub async fn generate_chat(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<ChatResult, SdkError> {
        request.validate()?;

        let body = text_request_to_anthropic(model, request, false);
        let url = format!("{}/messages", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&body),
        )
        .await?;

        let bytes = decode_response_bytes(response).await?;
        let parsed: AnthropicResponse = decode_json_response(&bytes)?;
        anthropic_response_to_chat_result(parsed)
    }

    #[cfg(feature = "streaming")]
    pub async fn stream(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        request.validate()?;

        let body = text_request_to_anthropic(model, request, true);
        let url = format!("{}/messages", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&body),
        )
        .await?;

        let status = response.status();
        if !status.is_success() {
            let bytes = read_bounded_body(response, ERROR_BODY_SNIPPET_LEN + 1)
                .await
                .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;
            return Err(provider_error_from_bytes(status, &bytes));
        }

        let events = Box::pin(response.bytes_stream().eventsource());

        let stream = futures_util::stream::unfold(
            (events, StreamAccumulator::default(), false),
            |(mut events, mut acc, done)| async move {
                if done {
                    return None;
                }

                match events.next().await {
                    Some(Ok(event)) => {
                        let items = process_event(&event.data, &mut acc);
                        Some((items, (events, acc, false)))
                    }
                    Some(Err(e)) => {
                        let items = vec![Err(SdkError::api(format!(
                            "EventSource stream error: {}",
                            e
                        )))];
                        Some((items, (events, acc, false)))
                    }
                    None => {
                        let items = if acc.finished_emitted {
                            vec![]
                        } else {
                            vec![Err(SdkError::stream_terminated(
                                Some("anthropic"),
                                "SSE stream ended before message_stop",
                            ))]
                        };
                        Some((items, (events, acc, true)))
                    }
                }
            },
        )
        .flat_map(futures_util::stream::iter);

        Ok(Box::pin(stream))
    }
}

async fn send_json_request(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, SdkError> {
    request
        .send()
        .await
        .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))
}

async fn decode_response_bytes(response: reqwest::Response) -> Result<Vec<u8>, SdkError> {
    let status = response.status();
    if !status.is_success() {
        let bytes = read_bounded_body(response, ERROR_BODY_SNIPPET_LEN + 1)
            .await
            .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;
        return Err(provider_error_from_bytes(status, &bytes));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| SdkError::from(AnthropicClientError::Reqwest(e)))?;
    Ok(bytes.to_vec())
}

fn decode_json_response<T>(bytes: &[u8]) -> Result<T, SdkError>
where
    T: DeserializeOwned,
{
    serde_json::from_slice(bytes).map_err(|e| SdkError::from(AnthropicClientError::Serde(e)))
}

fn provider_error_from_bytes(status: reqwest::StatusCode, bytes: &[u8]) -> SdkError {
    let text = String::from_utf8_lossy(bytes);
    let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);

    if let Ok(err) = serde_json::from_slice::<AnthropicErrorResponse>(bytes) {
        return SdkError::provider_api(
            "Anthropic",
            Some(status.as_u16()),
            None,
            err.error.error_type,
            err.error.message,
            Some(snippet),
        );
    }

    SdkError::provider_http(
        "Anthropic",
        Some(status.as_u16()),
        snippet.clone(),
        Some(snippet),
    )
}

// ---------------------------------------------------------------------------
// Internal streaming helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "streaming")]
#[derive(Default)]
struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    finish_reason: Option<FinishReason>,
    finished_emitted: bool,
    has_tool_calls: bool,
    tool_call_buffers: HashMap<u32, ToolCallBuffer>,
}

#[cfg(feature = "streaming")]
#[derive(Default)]
struct ToolCallBuffer {
    id: String,
    name: String,
    input: String,
}

#[cfg(feature = "streaming")]
impl StreamAccumulator {
    fn build_finished_event(&mut self) -> StreamEvent {
        self.finished_emitted = true;

        let total_tokens = self
            .input_tokens
            .zip(self.output_tokens)
            .and_then(|(input, output)| input.checked_add(output));

        let usage = if self.input_tokens.is_some() || self.output_tokens.is_some() {
            Some(TokenUsage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                total_tokens,
            })
        } else {
            None
        };

        StreamEvent::Finished {
            finish_reason: finish_reason_with_tool_override(
                self.finish_reason.take(),
                self.has_tool_calls,
            ),
            usage,
            response: ResponseMetadata {
                id: self.id.take(),
                model: self.model.take(),
            },
        }
    }
}

#[cfg(feature = "streaming")]
fn process_event(data: &str, acc: &mut StreamAccumulator) -> Vec<Result<StreamEvent, SdkError>> {
    // Step 1: read just the type tag — fail only on total JSON chaos.
    let envelope: EventEnvelope = match serde_json::from_str(data) {
        Ok(e) => e,
        Err(e) => return vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
    };

    match envelope.event_type.as_str() {
        "message_start" => match serde_json::from_str::<MessageStartEvent>(data) {
            Ok(evt) => {
                acc.id = evt.message.id;
                acc.model = evt.message.model;
                if let Some(usage) = evt.message.usage {
                    acc.input_tokens = usage.input_tokens;
                }
                vec![]
            }
            Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
        },

        "content_block_delta" => match serde_json::from_str::<ContentBlockDeltaEvent>(data) {
            Ok(evt) => {
                if let Some(text) = evt.delta.text
                    && !text.is_empty()
                {
                    return vec![Ok(StreamEvent::TextDelta(text))];
                }
                if let Some(delta) = evt.delta.partial_json {
                    let buffer = acc.tool_call_buffers.entry(evt.index).or_default();
                    buffer.input.push_str(&delta);
                    if !delta.is_empty() {
                        return vec![Ok(StreamEvent::ToolCallDelta {
                            id: buffer.id.clone(),
                            index: evt.index,
                            input_delta: delta,
                        })];
                    }
                }
                vec![]
            }
            Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
        },

        "content_block_start" => match serde_json::from_str::<ContentBlockStartEvent>(data) {
            Ok(evt) => match evt.content_block {
                ContentBlockStart::Text { text } => {
                    if let Some(text) = text
                        && !text.is_empty()
                    {
                        return vec![Ok(StreamEvent::TextDelta(text))];
                    }
                    vec![]
                }
                ContentBlockStart::ToolUse { id, name } => {
                    acc.tool_call_buffers.insert(
                        evt.index,
                        ToolCallBuffer {
                            id: id.clone(),
                            name: name.clone(),
                            input: String::new(),
                        },
                    );
                    vec![Ok(StreamEvent::ToolCallStarted {
                        id,
                        name,
                        index: evt.index,
                    })]
                }
                ContentBlockStart::Unknown => vec![],
            },
            Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
        },

        "content_block_stop" => match serde_json::from_str::<ContentBlockStopEvent>(data) {
            Ok(evt) => {
                let Some(buffer) = acc.tool_call_buffers.remove(&evt.index) else {
                    return vec![];
                };

                let call = ToolCall::from_json_input(buffer.id, buffer.name, &buffer.input);
                acc.has_tool_calls = true;
                vec![Ok(StreamEvent::ToolCallReady {
                    id: call.id,
                    name: call.name,
                    index: evt.index,
                    input: call.input,
                    provider_metadata: call.provider_metadata,
                })]
            }
            Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
        },

        "message_delta" => match serde_json::from_str::<MessageDeltaEvent>(data) {
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
        },

        "message_stop" => {
            if acc.finished_emitted {
                return vec![];
            }
            vec![Ok(acc.build_finished_event())]
        }

        "error" => match serde_json::from_str::<ErrorEvent>(data) {
            Ok(evt) => vec![Err(SdkError::api(evt.error.message))],
            Err(e) => vec![Err(SdkError::from(AnthropicClientError::Serde(e)))],
        },

        // All other event types (ping or any future events Anthropic may add)
        // are silently ignored.
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::Message;
    #[cfg(feature = "streaming")]
    use crate::core::types::FinishReason;
    use mockito;
    use serde_json::json;

    fn test_request() -> TextRequest {
        TextRequest {
            messages: vec![Message::user("Hello!")],
            max_output_tokens: Some(10),
            temperature: Some(0.7),
            tools: vec![],
            tool_choice: None,
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

        let mock = server
            .mock("POST", "/messages")
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
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client
            .generate("claude-3-5-sonnet", &test_request())
            .await
            .expect("Request should succeed");

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

        let mock = server
            .mock("POST", "/messages")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(mock_error.to_string())
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("invalid-api-key".to_string(), server.url());
        let result = client.generate("claude-3-5-sonnet", &test_request()).await;

        mock.assert_async().await;

        match result {
            Err(SdkError::Api {
                message,
                status,
                error_type,
                ..
            }) => {
                assert!(message.contains("Invalid API key."));
                assert_eq!(status, Some(401));
                assert_eq!(error_type.as_deref(), Some("authentication_error"));
            }
            _ => panic!("Expected Api error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_generate_error_non_json() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/messages")
            .with_status(502)
            .with_header("content-type", "text/plain")
            .with_body("Bad Gateway Timeout Exception...")
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("claude-3-5-sonnet", &test_request()).await;

        mock.assert_async().await;

        match result {
            Err(SdkError::Http {
                message, status, ..
            }) => {
                assert!(message.contains("Bad Gateway Timeout Exception..."));
                assert_eq!(status, Some(502));
            }
            _ => panic!("Expected Http error, got {:?}", result),
        }
    }

    // ------------------------------------------------------------------
    // Streaming unit tests — process_event
    // ------------------------------------------------------------------

    /// Unknown event types must be silently ignored (no output, no error).
    #[cfg(feature = "streaming")]
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
    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_malformed_known_type_errors() {
        let mut acc = StreamAccumulator::default();
        // `content_block_delta` recognized, but `delta` field is missing
        let data = r#"{"type":"content_block_delta","index":0}"#;
        let result = process_event(data, &mut acc);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(&result[0], Err(SdkError::Serialization { .. })),
            "Expected SdkError::Serde, got {:?}",
            result[0]
        );
    }

    /// `ping` and `content_block_start` / `content_block_stop` should also be
    /// silently ignored (they are well-known Anthropic event types that we don't
    /// need to process, but must not error on).
    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_known_passthrough_types_ignored() {
        let mut acc = StreamAccumulator::default();
        for data in [
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
        ] {
            let result = process_event(data, &mut acc);
            assert!(
                result.is_empty(),
                "Event {:?} should produce no output",
                data
            );
        }
    }

    #[cfg(feature = "streaming")]
    #[test]
    fn test_unknown_content_block_kind_is_ignored() {
        let mut acc = StreamAccumulator::default();
        let result = process_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"future_block","value":"ignored"}}"#,
            &mut acc,
        );

        assert!(result.is_empty());
    }

    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_content_block_start_tool_use() {
        let mut acc = StreamAccumulator::default();
        let result = process_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            &mut acc,
        );

        assert_eq!(result.len(), 1);
        match result.into_iter().next().unwrap().unwrap() {
            StreamEvent::ToolCallStarted { id, name, index } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(index, 0);
            }
            other => panic!("Expected ToolCallStarted, got {:?}", other),
        }
    }

    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_input_json_delta() {
        let mut acc = StreamAccumulator::default();
        let _ = process_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            &mut acc,
        );
        let result = process_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\"}"}}"#,
            &mut acc,
        );

        assert_eq!(result.len(), 1);
        match result.into_iter().next().unwrap().unwrap() {
            StreamEvent::ToolCallDelta {
                id,
                index,
                input_delta,
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(index, 0);
                assert_eq!(input_delta, r#"{"location":"Paris"}"#);
            }
            other => panic!("Expected ToolCallDelta, got {:?}", other),
        }
    }

    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_preserves_malformed_tool_json() {
        let mut acc = StreamAccumulator::default();
        let _ = process_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_bad","name":"get_weather"}}"#,
            &mut acc,
        );
        let _ = process_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\""}}"#,
            &mut acc,
        );
        let result = process_event(r#"{"type":"content_block_stop","index":0}"#, &mut acc);

        assert_eq!(result.len(), 1);
        match result.into_iter().next().unwrap().unwrap() {
            StreamEvent::ToolCallReady {
                id,
                name,
                input,
                provider_metadata,
                ..
            } => {
                assert_eq!(id, "toolu_bad");
                assert_eq!(name, "get_weather");
                assert_eq!(input, serde_json::Value::Null);
                let metadata = provider_metadata.expect("malformed input metadata");
                assert_eq!(
                    metadata.as_raw()["another_ai_sdk"]["tool_input"]["state"],
                    "malformed_json"
                );
            }
            other => panic!("Expected ToolCallReady, got {:?}", other),
        }
    }

    #[cfg(feature = "streaming")]
    #[test]
    fn test_process_event_content_block_stop_tool_use() {
        let mut acc = StreamAccumulator::default();
        let _ = process_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            &mut acc,
        );
        let _ = process_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\"}"}}"#,
            &mut acc,
        );
        let result = process_event(r#"{"type":"content_block_stop","index":0}"#, &mut acc);

        assert_eq!(result.len(), 1);
        match result.into_iter().next().unwrap().unwrap() {
            StreamEvent::ToolCallReady {
                id,
                name,
                index,
                input,
                ..
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(index, 0);
                assert_eq!(input["location"], "Paris");
            }
            other => panic!("Expected ToolCallReady, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Streaming integration tests
    // ------------------------------------------------------------------

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_setup_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .with_status(401)
            .with_body("Unauthorized")
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test".to_string(), server.url());
        let result = client.stream("claude", &test_request()).await;

        mock.assert_async().await;

        assert!(matches!(result, Err(SdkError::Http { .. })));
    }

    #[cfg(feature = "streaming")]
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

        let mock = server
            .mock("POST", "/messages")
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
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client
            .stream("claude", &test_request())
            .await
            .expect("Stream should start");

        mock.assert_async().await;

        let mut text = String::new();
        let mut finished = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Finished {
                    finish_reason,
                    usage,
                    response,
                } => {
                    assert!(matches!(finish_reason, FinishReason::Stop));
                    assert_eq!(response.id.as_deref(), Some("msg_123"));
                    assert_eq!(response.model.as_deref(), Some("claude-3-5-sonnet"));
                    let u = usage.expect("usage should be present");
                    assert_eq!(u.input_tokens, Some(10));
                    assert_eq!(u.output_tokens, Some(5));
                    assert_eq!(u.total_tokens, Some(15));
                    finished = true;
                }
                _ => {}
            }
        }
        assert_eq!(text, "Hello");
        assert!(finished);
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_without_message_stop_returns_error() {
        let mut server = mockito::Server::new_async().await;

        // No message_stop event
        let mock_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_456\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-haiku\",\"usage\":{\"input_tokens\":8}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n"
        );

        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "test-api-key")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client
            .stream("claude-3-haiku", &test_request())
            .await
            .expect("Stream should start");

        mock.assert_async().await;

        let mut text = String::new();
        let mut termination_error = None;

        while let Some(evt) = stream.next().await {
            match evt {
                Ok(StreamEvent::TextDelta(d)) => text.push_str(&d),
                Err(error) => termination_error = Some(error),
                Ok(_) => {}
            }
        }
        assert_eq!(text, "Hi");
        assert!(matches!(
            termination_error,
            Some(SdkError::StreamTerminated { provider, .. })
                if provider.as_deref() == Some("anthropic")
        ));
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_tool_use_single_end_to_end() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_tool\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet\",\"usage\":{\"input_tokens\":20}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":8}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        );

        let _mock = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client
            .stream("claude-sonnet", &test_request())
            .await
            .unwrap();

        let mut started = false;
        let mut delta = String::new();
        let mut ready = false;
        let mut finished = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::ToolCallStarted { id, name, index } => {
                    assert_eq!(id, "toolu_1");
                    assert_eq!(name, "get_weather");
                    assert_eq!(index, 0);
                    started = true;
                }
                StreamEvent::ToolCallDelta { input_delta, .. } => delta.push_str(&input_delta),
                StreamEvent::ToolCallReady { input, .. } => {
                    assert_eq!(input["location"], "Paris");
                    ready = true;
                }
                StreamEvent::Finished {
                    finish_reason,
                    usage,
                    response,
                } => {
                    assert!(matches!(finish_reason, FinishReason::ToolUse));
                    assert_eq!(response.id.as_deref(), Some("msg_tool"));
                    let usage = usage.expect("usage should be present");
                    assert_eq!(usage.total_tokens, Some(28));
                    finished = true;
                }
                _ => {}
            }
        }

        assert!(started);
        assert_eq!(delta, r#"{"location":"Paris"}"#);
        assert!(ready);
        assert!(finished);
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_mixed_text_and_tool() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking \"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\":\\\"Austin\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        );

        let _mock = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = AnthropicClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client
            .stream("claude-sonnet", &test_request())
            .await
            .unwrap();

        let mut text = String::new();
        let mut input = None;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::ToolCallReady {
                    input: ready_input, ..
                } => input = Some(ready_input),
                _ => {}
            }
        }

        assert_eq!(text, "Checking ");
        assert_eq!(input.expect("tool should be ready")["location"], "Austin");
    }
}

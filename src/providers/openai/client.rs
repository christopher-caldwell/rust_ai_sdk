#[cfg(feature = "streaming")]
use std::collections::HashMap;

use serde::de::DeserializeOwned;

#[cfg(feature = "streaming")]
use crate::core::message::ToolCall;
use crate::core::{
    error::SdkError,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

#[cfg(feature = "streaming")]
use super::types::{ChatCompletionChunk, map_finish_reason};
use super::{
    error::{OpenAiClientError, truncate_body},
    types::{
        ChatCompletionResponse, OpenAiErrorBody, chat_response_to_chat_result,
        chat_response_to_text_result, text_request_to_openai,
    },
};
#[cfg(feature = "streaming")]
use crate::core::stream::StreamEvent;
use crate::core::stream::TextEventStream;
#[cfg(feature = "streaming")]
use crate::core::types::{FinishReason, ResponseMetadata, Usage as TokenUsage};
#[cfg(feature = "streaming")]
use eventsource_stream::Eventsource;
#[cfg(feature = "streaming")]
use futures_util::StreamExt;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

#[derive(Clone)]
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
        request.validate()?;

        let body = text_request_to_openai(model, request, false);
        let url = format!("{}/chat/completions", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.api_key),
                )
                .json(&body),
        )
        .await?;

        let bytes = decode_response_bytes(response).await?;
        let parsed: ChatCompletionResponse = decode_json_response(&bytes)?;
        chat_response_to_text_result(parsed)
    }

    pub async fn generate_chat(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<ChatResult, SdkError> {
        request.validate()?;

        let body = text_request_to_openai(model, request, false);
        let url = format!("{}/chat/completions", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.api_key),
                )
                .json(&body),
        )
        .await?;

        let bytes = decode_response_bytes(response).await?;
        let parsed: ChatCompletionResponse = decode_json_response(&bytes)?;
        chat_response_to_chat_result(parsed)
    }

    #[cfg(feature = "streaming")]
    pub async fn stream(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        request.validate()?;

        let body = text_request_to_openai(model, request, true);
        let url = format!("{}/chat/completions", self.base_url);

        let response = send_json_request(
            self.http
                .post(&url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.api_key),
                )
                .json(&body),
        )
        .await?;

        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))?;
            return Err(provider_error_from_bytes(status, &bytes));
        }

        // Per-stream mutable state accumulated across chunks.
        // It is captured directly as a mutable variable by the stream closure;
        // no interior mutability primitives are needed.
        let mut acc = StreamAccumulator::default();

        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) => {
                    if event.data == "[DONE]" {
                        // If no Finished has been emitted yet (e.g. finish_reason
                        // never arrived), emit one now using whatever state we have.
                        if !acc.finished_emitted {
                            acc.finished_emitted = true;
                            return vec![Ok(StreamEvent::Finished {
                                finish_reason: acc
                                    .finish_reason
                                    .take()
                                    .unwrap_or_else(|| FinishReason::Other("unknown".to_string())),
                                usage: acc.usage.take(),
                                response: ResponseMetadata {
                                    id: acc.id.clone(),
                                    model: acc.model.clone(),
                                },
                            })];
                        }
                        return vec![];
                    }

                    match serde_json::from_str::<ChatCompletionChunk>(&event.data) {
                        Ok(chunk) => process_chunk(chunk, &mut acc),
                        Err(e) => vec![Err(SdkError::from(OpenAiClientError::Serde(e)))],
                    }
                }
                Err(e) => vec![Err(SdkError::api(format!(
                    "EventSource stream error: {}",
                    e
                )))],
            })
            .flat_map(futures_util::stream::iter);

        Ok(Box::pin(stream))
    }

    #[cfg(not(feature = "streaming"))]
    pub async fn stream(
        &self,
        _model: &str,
        _request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        Err(SdkError::Validation(
            "OpenAI streaming requires the `streaming` feature.".to_string(),
        ))
    }
}

async fn send_json_request(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, SdkError> {
    request
        .send()
        .await
        .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))
}

async fn decode_response_bytes(response: reqwest::Response) -> Result<Vec<u8>, SdkError> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| SdkError::from(OpenAiClientError::Reqwest(e)))?;

    if !status.is_success() {
        return Err(provider_error_from_bytes(status, &bytes));
    }

    Ok(bytes.to_vec())
}

fn decode_json_response<T>(bytes: &[u8]) -> Result<T, SdkError>
where
    T: DeserializeOwned,
{
    serde_json::from_slice(bytes).map_err(|e| SdkError::from(OpenAiClientError::Serde(e)))
}

fn provider_error_from_bytes(status: reqwest::StatusCode, bytes: &[u8]) -> SdkError {
    let text = String::from_utf8_lossy(bytes);
    let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);

    if let Ok(err) = serde_json::from_slice::<OpenAiErrorBody>(bytes) {
        return SdkError::provider_api(
            "OpenAI",
            Some(status.as_u16()),
            err.error.code,
            err.error.error_type,
            err.error.message,
            Some(snippet),
        );
    }

    SdkError::provider_http(
        "OpenAI",
        Some(status.as_u16()),
        snippet.clone(),
        Some(snippet),
    )
}

// ---------------------------------------------------------------------------
// Internal streaming helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "streaming")]
struct PendingFinished {
    finish_reason: FinishReason,
    response: ResponseMetadata,
}

#[cfg(feature = "streaming")]
#[derive(Default)]
struct ToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
    started_emitted: bool,
}

/// Rolling state accumulated across chunks in a single stream response.
///
/// OpenAI frequently sends `id` and `model` only on the first chunk, and
/// `usage` only on a trailing usage-only chunk (when `include_usage: true`
/// is requested). We buffer the best-known values here and use them when
/// constructing the final `Finished` event.
#[cfg(feature = "streaming")]
#[derive(Default)]
struct StreamAccumulator {
    /// Last seen response id (often only in the first chunk).
    id: Option<String>,
    /// Last seen model name (often only in the first chunk).
    model: Option<String>,
    /// Accumulated usage, updated from any chunk that carries a usage field.
    usage: Option<TokenUsage>,
    /// The finish_reason observed from a choices entry.
    finish_reason: Option<FinishReason>,
    /// Whether a `Finished` event has already been emitted into the stream.
    finished_emitted: bool,
    /// A pending `Finished` event whose emission we have deferred.
    ///
    /// When a finish_reason chunk arrives we do not emit `Finished`
    /// immediately; instead we park it here so a subsequent usage-only
    /// chunk can update `acc.usage` before we flush it.  The pending event
    /// is flushed the next time `process_chunk` is called without a new
    /// finish_reason, or at `[DONE]`.
    pending_finished: Option<PendingFinished>,
    /// Incrementally assembled tool calls keyed by OpenAI's `index`.
    tool_call_buffers: HashMap<u32, ToolCallBuffer>,
}

/// Process one parsed `ChatCompletionChunk` and return the stream events to emit.
///
/// # Ordering note
/// OpenAI streams chunks in roughly this order (with `include_usage: true`):
///   1. First chunk: id, model, first delta content
///   2. Middle chunks: more delta content
///   3. Stop chunk: finish_reason set, choices may be empty or have empty delta
///   4. Usage chunk: choices is empty, usage is populated
///   5. [DONE]
///
/// We want the final `Finished` event to carry the usage from chunk 4.
/// We achieve this by *deferring* the `Finished` event after we observe
/// finish_reason (step 3), then flushing it in step 4 after we update usage.
#[cfg(feature = "streaming")]
fn process_chunk(
    chunk: ChatCompletionChunk,
    acc: &mut StreamAccumulator,
) -> Vec<Result<StreamEvent, SdkError>> {
    let mut events: Vec<Result<StreamEvent, SdkError>> = Vec::new();

    // Always update rolling id/model from any chunk that carries them.
    if let Some(id) = chunk.id {
        acc.id = Some(id);
    }
    if let Some(model) = chunk.model {
        acc.model = Some(model);
    }

    // Update rolling usage from this chunk (may be None on most chunks).
    if let Some(u) = chunk.usage {
        acc.usage = Some(TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
    }

    // If we stored a pending Finished from a previous chunk and this chunk
    // carries updated usage (or simply is a usage-only chunk arriving after
    // the stop chunk), flush the pending event now with the updated usage.
    if let Some(pending) = acc.pending_finished.take() {
        events.push(Ok(StreamEvent::Finished {
            finish_reason: pending.finish_reason,
            usage: acc.usage.clone(),
            response: pending.response,
        }));
        acc.finished_emitted = true;
    }

    // Collect text deltas and check for finish_reason across choices.
    let mut chunk_finish_reason: Option<String> = None;
    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            events.push(Ok(StreamEvent::TextDelta(content.clone())));
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tool_call in tool_calls {
                process_tool_call_delta(tool_call, acc, &mut events);
            }
        }
        // Use the first finish_reason found; n>1 not supported in this milestone.
        if chunk_finish_reason.is_none() {
            chunk_finish_reason = choice.finish_reason.clone();
        }
    }

    // If this chunk carries a finish_reason, build the pending Finished event but
    // defer emitting it so the usage-only chunk (if any) can arrive first.
    if let Some(reason) = chunk_finish_reason {
        let mut finish_reason = map_finish_reason(&reason);
        acc.finish_reason = Some(finish_reason.clone());

        let ready_tool_calls = drain_tool_calls(acc);
        if !ready_tool_calls.is_empty() {
            finish_reason = FinishReason::ToolUse;
            acc.finish_reason = Some(finish_reason.clone());
        }

        for (index, buf) in ready_tool_calls {
            let call = ToolCall::from_json_input(buf.id, buf.name, &buf.arguments);
            events.push(Ok(StreamEvent::ToolCallReady {
                id: call.id,
                name: call.name,
                index,
                input: call.input,
                provider_metadata: call.provider_metadata,
            }));
        }

        // Park the event; it will be flushed on the next chunk or at [DONE].
        acc.pending_finished = Some(PendingFinished {
            finish_reason,
            response: ResponseMetadata {
                id: acc.id.clone(),
                model: acc.model.clone(),
            },
        });
    }

    events
}

#[cfg(feature = "streaming")]
fn process_tool_call_delta(
    delta: &super::types::OaiChunkToolCallDelta,
    acc: &mut StreamAccumulator,
    events: &mut Vec<Result<StreamEvent, SdkError>>,
) {
    let buf = acc
        .tool_call_buffers
        .entry(delta.index)
        .or_insert_with(|| ToolCallBuffer {
            id: delta
                .id
                .clone()
                .unwrap_or_else(|| format!("tool_call_{}", delta.index)),
            ..ToolCallBuffer::default()
        });

    if let Some(id) = &delta.id {
        buf.id = id.clone();
    }
    if let Some(function) = &delta.function
        && let Some(name) = &function.name
    {
        buf.name = name.clone();
    }

    if !buf.started_emitted && !buf.name.is_empty() {
        events.push(Ok(StreamEvent::ToolCallStarted {
            id: buf.id.clone(),
            name: buf.name.clone(),
            index: delta.index,
        }));
        buf.started_emitted = true;
    }

    if let Some(function) = &delta.function
        && let Some(arguments) = &function.arguments
        && !arguments.is_empty()
    {
        buf.arguments.push_str(arguments);
        events.push(Ok(StreamEvent::ToolCallDelta {
            id: buf.id.clone(),
            index: delta.index,
            input_delta: arguments.clone(),
        }));
    }
}

#[cfg(feature = "streaming")]
fn drain_tool_calls(acc: &mut StreamAccumulator) -> Vec<(u32, ToolCallBuffer)> {
    let mut indexes: Vec<u32> = acc.tool_call_buffers.keys().copied().collect();
    indexes.sort_unstable();

    indexes
        .into_iter()
        .filter_map(|index| acc.tool_call_buffers.remove(&index).map(|buf| (index, buf)))
        .collect()
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
                parts: vec![],
            }],
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

        let mock = server
            .mock("POST", "/chat/completions")
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
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client
            .generate("gpt-4", &test_request())
            .await
            .expect("Request should succeed");

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
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });

        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(mock_error.to_string())
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("invalid-api-key".to_string(), server.url());
        let result = client.generate("gpt-4", &test_request()).await;

        mock.assert_async().await;

        match result {
            Err(SdkError::Api {
                message,
                status,
                code,
                error_type,
                ..
            }) => {
                assert!(message.contains("Invalid API key."));
                assert_eq!(status, Some(401));
                assert_eq!(code.as_deref(), Some("invalid_api_key"));
                assert_eq!(error_type.as_deref(), Some("invalid_request_error"));
            }
            _ => panic!("Expected Api error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_generate_error_non_json() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(502)
            .with_header("content-type", "text/plain")
            .with_body("Bad Gateway Timeout Exception...")
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let result = client.generate("gpt-4", &test_request()).await;

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
    // Streaming: setup error
    // ------------------------------------------------------------------

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_setup_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("Unauthorized")
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let result = client.stream("gpt-4", &test_request()).await;

        mock.assert_async().await;

        // Must be a direct Err, not hidden inside a stream item.
        assert!(matches!(result, Err(SdkError::Http { .. })));
    }

    // ------------------------------------------------------------------
    // Streaming: happy path (includes stream_options in request body)
    // ------------------------------------------------------------------

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_success() {
        let mut server = mockito::Server::new_async().await;

        // Chunk 1: id + model + text delta
        // Chunk 2: finish_reason (no id/model — tests metadata carry-forward)
        // Chunk 3: usage-only trailing chunk
        let mock_body = concat!(
            "data: {\"id\":\"req_123\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n"
        );

        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-api-key")
            .match_body(mockito::Matcher::Json(json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.7,
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test-api-key".to_string(), server.url());
        let mut stream = client
            .stream("gpt-4", &test_request())
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
                    // Metadata must be carried forward from chunk 1.
                    assert_eq!(response.id.as_deref(), Some("req_123"));
                    assert_eq!(response.model.as_deref(), Some("gpt-4"));
                    // Usage must arrive from the trailing usage chunk.
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

    // ------------------------------------------------------------------
    // Streaming: metadata carry-forward
    // ------------------------------------------------------------------

    /// First chunk carries id/model; the finish chunk does not.
    /// The emitted Finished must still include id and model.
    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_carries_forward_metadata() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"id\":\"resp_abc\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4o", &test_request()).await.unwrap();

        let mut got_finished = false;
        while let Some(evt) = stream.next().await {
            if let Ok(StreamEvent::Finished { response, .. }) = evt {
                assert_eq!(
                    response.id.as_deref(),
                    Some("resp_abc"),
                    "id should carry forward"
                );
                assert_eq!(
                    response.model.as_deref(),
                    Some("gpt-4o"),
                    "model should carry forward"
                );
                got_finished = true;
            }
        }
        assert!(got_finished, "Expected a Finished event");
    }

    // ------------------------------------------------------------------
    // Streaming: usage-only trailing chunk updates Finished
    // ------------------------------------------------------------------

    /// finish_reason arrives before the usage chunk.
    /// The deferred Finished should be flushed with usage from the trailing chunk.
    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_usage_from_trailing_chunk() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"id\":\"r1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Word\"}}]}\n\n",
            // finish chunk — no usage
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            // usage-only chunk — no choices
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4", &test_request()).await.unwrap();

        let mut text = String::new();
        let mut finished = false;
        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Finished {
                    usage, response, ..
                } => {
                    let u = usage.expect("usage must be present from trailing chunk");
                    assert_eq!(u.input_tokens, Some(5));
                    assert_eq!(u.output_tokens, Some(3));
                    assert_eq!(u.total_tokens, Some(8));
                    assert_eq!(response.id.as_deref(), Some("r1"));
                    finished = true;
                }
                _ => {}
            }
        }
        assert_eq!(text, "Word");
        assert!(finished);
    }

    // ------------------------------------------------------------------
    // Streaming: [DONE] fallback uses accumulated state
    // ------------------------------------------------------------------

    /// No finish_reason chunk ever arrives; [DONE] causes fallback Finished.
    /// The fallback must still use the id/model/usage accumulated so far.
    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_done_fallback_uses_accumulated_state() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"id\":\"r42\",\"model\":\"gpt-3.5\",\"choices\":[{\"delta\":{\"content\":\"Text\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-3.5", &test_request()).await.unwrap();

        let mut got_finished = false;
        while let Some(evt) = stream.next().await {
            if let Ok(StreamEvent::Finished {
                finish_reason,
                usage,
                response,
            }) = evt
            {
                // Reason should fall back to Other("unknown") since no finish_reason chunk.
                assert!(matches!(finish_reason, FinishReason::Other(ref s) if s == "unknown"));
                assert_eq!(response.id.as_deref(), Some("r42"));
                assert_eq!(response.model.as_deref(), Some("gpt-3.5"));
                // Usage must be from the preceding usage chunk.
                let u = usage.expect("usage should be accumulated");
                assert_eq!(u.input_tokens, Some(2));
                assert_eq!(u.total_tokens, Some(3));
                got_finished = true;
            }
        }
        assert!(got_finished);
    }

    // ------------------------------------------------------------------
    // Streaming: usage-only chunk with empty choices does not error
    // ------------------------------------------------------------------

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_usage_only_chunk_no_error() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            // Usage-only: choices is an empty array — must not produce an error event.
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4", &test_request()).await.unwrap();

        let mut error_count = 0usize;
        let mut finished_count = 0usize;
        while let Some(evt) = stream.next().await {
            match evt {
                Ok(StreamEvent::Finished { .. }) => finished_count += 1,
                Ok(_) => {}
                Err(_) => error_count += 1,
            }
        }
        assert_eq!(error_count, 0, "usage-only chunk must not produce an error");
        assert_eq!(finished_count, 1, "exactly one Finished event expected");
    }

    // ------------------------------------------------------------------
    // Streaming: malformed event produces an error item in the stream
    // ------------------------------------------------------------------

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_malformed_event() {
        let mut server = mockito::Server::new_async().await;
        let mock_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {bad json\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client
            .stream("gpt-4", &test_request())
            .await
            .expect("Stream starts");

        let evt1 = stream.next().await.unwrap().unwrap();
        match evt1 {
            StreamEvent::TextDelta(t) => assert_eq!(t, "Hi"),
            _ => panic!("Expected text"),
        }

        let evt2 = stream.next().await.unwrap();
        assert!(evt2.is_err());
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_single_tool_call() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"id\":\"req_tool_1\",\"model\":\"gpt-4.1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"Paris\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4.1", &test_request()).await.unwrap();

        let mut saw_started = false;
        let mut deltas = Vec::new();
        let mut saw_ready = false;
        let mut saw_finished = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::ToolCallStarted { id, name, index } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "get_weather");
                    assert_eq!(index, 0);
                    saw_started = true;
                }
                StreamEvent::ToolCallDelta { input_delta, .. } => deltas.push(input_delta),
                StreamEvent::ToolCallReady {
                    id,
                    name,
                    index,
                    input,
                    ..
                } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "get_weather");
                    assert_eq!(index, 0);
                    assert_eq!(input["location"], "Paris");
                    saw_ready = true;
                }
                StreamEvent::Finished {
                    finish_reason,
                    usage,
                    response,
                } => {
                    assert!(matches!(finish_reason, FinishReason::ToolUse));
                    assert_eq!(response.id.as_deref(), Some("req_tool_1"));
                    assert_eq!(response.model.as_deref(), Some("gpt-4.1"));
                    let usage = usage.expect("usage should be carried forward");
                    assert_eq!(usage.total_tokens, Some(16));
                    saw_finished = true;
                }
                StreamEvent::TextDelta(_) => panic!("unexpected text delta"),
            }
        }

        assert!(saw_started);
        assert_eq!(
            deltas,
            vec![r#"{"location":""#.to_string(), r#"Paris"}"#.to_string()]
        );
        assert!(saw_ready);
        assert!(saw_finished);
    }

    #[test]
    fn test_process_chunk_preserves_malformed_tool_json() {
        let mut acc = StreamAccumulator::default();
        let tool_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_bad",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Paris\""
                        }
                    }]
                }
            }]
        }))
        .unwrap();
        let finish_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {},
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();

        let start_and_delta_events = process_chunk(tool_chunk, &mut acc);
        let ready_events = process_chunk(finish_chunk, &mut acc);

        assert!(start_and_delta_events.iter().any(|event| {
            matches!(
                event,
                Ok(StreamEvent::ToolCallStarted {
                    id,
                    name,
                    index
                }) if id == "call_bad" && name == "get_weather" && *index == 0
            )
        }));
        let ready_event = ready_events
            .into_iter()
            .find_map(|event| match event.unwrap() {
                StreamEvent::ToolCallReady {
                    input,
                    provider_metadata,
                    ..
                } => Some((input, provider_metadata)),
                _ => None,
            })
            .expect("malformed tool call should still emit ToolCallReady");

        assert_eq!(ready_event.0, serde_json::Value::Null);
        let metadata = ready_event.1.expect("malformed input metadata");
        assert_eq!(
            metadata.as_raw()["another_ai_sdk"]["tool_input"]["state"],
            "malformed_json"
        );
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_parallel_tool_calls() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
            "{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"tool_a\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}},",
            "{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"tool_b\",\"arguments\":\"{\\\"units\\\":\\\"c\\\"}\"}}",
            "]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client.stream("gpt-4.1", &test_request()).await.unwrap();

        let mut started = Vec::new();
        let mut ready = Vec::new();

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::ToolCallStarted { name, index, .. } => started.push((index, name)),
                StreamEvent::ToolCallReady {
                    name, index, input, ..
                } => {
                    ready.push((index, name, input));
                }
                StreamEvent::Finished { finish_reason, .. } => {
                    assert!(matches!(finish_reason, FinishReason::ToolUse));
                }
                _ => {}
            }
        }

        assert_eq!(
            started,
            vec![(0, "tool_a".to_string()), (1, "tool_b".to_string())]
        );
        assert_eq!(ready.len(), 2);
        assert_eq!(ready[0].0, 0);
        assert_eq!(ready[0].1, "tool_a");
        assert_eq!(ready[0].2["city"], "Paris");
        assert_eq!(ready[1].0, 1);
        assert_eq!(ready[1].1, "tool_b");
        assert_eq!(ready[1].2["units"], "c");
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_tool_call_with_text_prefix() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"id\":\"req_mix\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"delta\":{\"content\":\"Checking weather... \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_mix\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Austin\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client
            .stream("gpt-4.1-mini", &test_request())
            .await
            .unwrap();

        let mut text = String::new();
        let mut ready_input = None;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::ToolCallReady { input, .. } => ready_input = Some(input),
                _ => {}
            }
        }

        assert_eq!(text, "Checking weather... ");
        let input = ready_input.expect("tool call should be ready");
        assert_eq!(input["location"], "Austin");
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_stream_tool_call_ready_when_provider_finish_is_stop() {
        let mut server = mockito::Server::new_async().await;

        let mock_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_stop\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OpenAiClient::with_base_url("test".to_string(), server.url());
        let mut stream = client
            .stream("gpt-4.1-mini", &test_request())
            .await
            .unwrap();

        let mut saw_ready = false;
        let mut saw_tool_finish = false;

        while let Some(evt) = stream.next().await {
            match evt.unwrap() {
                StreamEvent::ToolCallReady { input, .. } => {
                    assert_eq!(input["location"], "Paris");
                    saw_ready = true;
                }
                StreamEvent::Finished { finish_reason, .. } => {
                    assert!(matches!(finish_reason, FinishReason::ToolUse));
                    saw_tool_finish = true;
                }
                _ => {}
            }
        }

        assert!(saw_ready);
        assert!(saw_tool_finish);
    }
}

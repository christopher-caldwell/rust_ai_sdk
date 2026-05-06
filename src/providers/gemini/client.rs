use crate::core::{
    error::SdkError,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

use super::{
    error::{GeminiClientError, truncate_body},
    types::{
        GeminiErrorResponse, GenerateContentResponse, gemini_response_to_chat_result,
        gemini_response_to_text_result, gemini_tool_metadata, map_finish_reason,
        text_request_to_gemini,
    },
};
use crate::core::stream::{StreamEvent, TextEventStream};
use crate::core::types::{FinishReason, ResponseMetadata, Usage as TokenUsage};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const ERROR_BODY_SNIPPET_LEN: usize = 512;

#[derive(Clone)]
pub struct GeminiClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl GeminiClient {
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
        let body = text_request_to_gemini(request)?;
        let url = self.generate_url(model);

        let response = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(GeminiClientError::Reqwest(e)))?;

        let bytes = self.response_bytes(response).await?;
        let parsed: GenerateContentResponse = serde_json::from_slice(&bytes)
            .map_err(|e| SdkError::from(GeminiClientError::Serde(e)))?;
        gemini_response_to_text_result(parsed)
    }

    pub async fn generate_chat(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<ChatResult, SdkError> {
        let body = text_request_to_gemini(request)?;
        let url = self.generate_url(model);

        let response = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(GeminiClientError::Reqwest(e)))?;

        let bytes = self.response_bytes(response).await?;
        let parsed: GenerateContentResponse = serde_json::from_slice(&bytes)
            .map_err(|e| SdkError::from(GeminiClientError::Serde(e)))?;
        gemini_response_to_chat_result(parsed)
    }

    pub async fn stream(
        &self,
        model: &str,
        request: &TextRequest,
    ) -> Result<TextEventStream, SdkError> {
        let body = text_request_to_gemini(request)?;
        let url = self.stream_url(model);

        let response = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| SdkError::from(GeminiClientError::Reqwest(e)))?;

        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| SdkError::from(GeminiClientError::Reqwest(e)))?;
            return Err(gemini_error_from_bytes(status, &bytes));
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
                        if event.data == "[DONE]" {
                            let items = acc.finish_if_needed();
                            return Some((items, (events, acc, false)));
                        }
                        let items =
                            match serde_json::from_str::<GenerateContentResponse>(&event.data) {
                                Ok(resp) => process_stream_response(resp, &mut acc),
                                Err(e) => vec![Err(SdkError::from(GeminiClientError::Serde(e)))],
                            };
                        Some((items, (events, acc, false)))
                    }
                    Some(Err(e)) => {
                        let items = vec![Err(SdkError::Api(format!(
                            "EventSource stream error: {}",
                            e
                        )))];
                        Some((items, (events, acc, false)))
                    }
                    None => {
                        let items = acc.finish_if_needed();
                        Some((items, (events, acc, true)))
                    }
                }
            },
        )
        .flat_map(futures_util::stream::iter);

        Ok(Box::pin(stream))
    }

    async fn response_bytes(&self, response: reqwest::Response) -> Result<Vec<u8>, SdkError> {
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| SdkError::from(GeminiClientError::Reqwest(e)))?;

        if !status.is_success() {
            return Err(gemini_error_from_bytes(status, &bytes));
        }

        Ok(bytes.to_vec())
    }

    fn generate_url(&self, model: &str) -> String {
        format!("{}/models/{}:generateContent", self.base_url, model)
    }

    fn stream_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        )
    }
}

fn gemini_error_from_bytes(status: reqwest::StatusCode, bytes: &[u8]) -> SdkError {
    let text = String::from_utf8_lossy(bytes);
    let snippet = truncate_body(text.as_ref(), ERROR_BODY_SNIPPET_LEN);
    if let Ok(err) = serde_json::from_slice::<GeminiErrorResponse>(bytes) {
        return SdkError::Api(format!("{} (HTTP {})", err.error.message, status));
    }
    SdkError::Http(format!("HTTP {}: {}", status, snippet))
}

#[derive(Default)]
struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    usage: Option<TokenUsage>,
    finish_reason: Option<FinishReason>,
    finished_emitted: bool,
    next_tool_index: u32,
}

impl StreamAccumulator {
    fn finish_if_needed(&mut self) -> Vec<Result<StreamEvent, SdkError>> {
        if self.finished_emitted {
            return vec![];
        }
        self.finished_emitted = true;
        vec![Ok(StreamEvent::Finished {
            finish_reason: self
                .finish_reason
                .take()
                .unwrap_or_else(|| FinishReason::Other("unknown".to_string())),
            usage: self.usage.take(),
            response: ResponseMetadata {
                id: self.id.take(),
                model: self.model.take(),
            },
        })]
    }
}

fn process_stream_response(
    resp: GenerateContentResponse,
    acc: &mut StreamAccumulator,
) -> Vec<Result<StreamEvent, SdkError>> {
    if resp.response_id.is_some() {
        acc.id = resp.response_id;
    }
    if resp.model_version.is_some() {
        acc.model = resp.model_version;
    }
    if let Some(usage) = resp.usage_metadata {
        acc.usage = Some(TokenUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        });
    }

    let mut events = Vec::new();
    for candidate in resp.candidates {
        if let Some(content) = candidate.content {
            for part in content.parts {
                if let Some(text) = part.text {
                    if !text.is_empty() {
                        events.push(Ok(StreamEvent::TextDelta(text)));
                    }
                }
                if let Some(call) = part.function_call {
                    let index = acc.next_tool_index;
                    acc.next_tool_index += 1;

                    let id = call
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("gemini_call_{}", index));
                    let input = call.args;
                    let input_delta = input.to_string();
                    events.push(Ok(StreamEvent::ToolCallStarted {
                        id: id.clone(),
                        name: call.name.clone(),
                        index,
                    }));
                    if !input_delta.is_empty() {
                        events.push(Ok(StreamEvent::ToolCallDelta {
                            id: id.clone(),
                            index,
                            input_delta,
                        }));
                    }
                    events.push(Ok(StreamEvent::ToolCallReady {
                        id,
                        name: call.name.clone(),
                        index,
                        input,
                        provider_metadata: Some(gemini_tool_metadata(
                            call.id.as_deref(),
                            &call.name,
                            part.thought_signature.as_deref(),
                        )),
                    }));
                    acc.finish_reason = Some(FinishReason::ToolUse);
                }
            }
        }

        if acc.finish_reason != Some(FinishReason::ToolUse) {
            if let Some(reason) = candidate.finish_reason {
                acc.finish_reason = Some(map_finish_reason(&reason));
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{request::TextRequest, stream::StreamEvent};
    use futures_util::StreamExt;
    use mockito::Server;

    #[tokio::test]
    async fn generate_maps_success_response() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/gemini-2.5-flash:generateContent")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "responseId":"resp_1",
                    "modelVersion":"gemini-2.5-flash",
                    "candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}]
                }"#,
            )
            .create_async()
            .await;

        let client = GeminiClient::with_base_url("key".to_string(), server.url());
        let result = client
            .generate("gemini-2.5-flash", &TextRequest::prompt("hi"))
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(result.text, "Hello");
        assert_eq!(result.response.id.as_deref(), Some("resp_1"));
    }

    #[tokio::test]
    async fn generate_maps_error_response() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/gemini-2.5-flash:generateContent")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(r#"{"error":{"message":"bad request"}}"#)
            .create_async()
            .await;

        let client = GeminiClient::with_base_url("key".to_string(), server.url());
        let err = client
            .generate("gemini-2.5-flash", &TextRequest::prompt("hi"))
            .await
            .unwrap_err();

        mock.assert_async().await;
        assert!(matches!(err, SdkError::Api(message) if message.contains("bad request")));
    }

    #[tokio::test]
    async fn stream_emits_text_and_finished() {
        let mut server = Server::new_async().await;
        let body = concat!(
            "data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-flash\",",
            "\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},",
            "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,",
            "\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n"
        );
        let mock = server
            .mock(
                "POST",
                "/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            )
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = GeminiClient::with_base_url("key".to_string(), server.url());
        let mut stream = client
            .stream("gemini-2.5-flash", &TextRequest::prompt("hi"))
            .await
            .unwrap();

        let mut text = String::new();
        let mut finished = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Finished {
                    finish_reason,
                    usage,
                    response,
                } => finished = Some((finish_reason, usage, response)),
                _ => {}
            }
        }

        mock.assert_async().await;
        assert_eq!(text, "Hello");
        let (reason, usage, response) = finished.unwrap();
        assert!(matches!(reason, FinishReason::Stop));
        assert_eq!(usage.unwrap().total_tokens, Some(2));
        assert_eq!(response.id.as_deref(), Some("resp_1"));
    }

    #[tokio::test]
    async fn stream_emits_tool_call_sequence() {
        let mut server = Server::new_async().await;
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":",
            "{\"id\":\"call_1\",\"name\":\"get_weather\",\"args\":{\"location\":\"Paris\"}},",
            "\"thoughtSignature\":\"sig_123\"}]},",
            "\"finishReason\":\"STOP\"}]}\n\n"
        );
        let mock = server
            .mock(
                "POST",
                "/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            )
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = GeminiClient::with_base_url("key".to_string(), server.url());
        let mut stream = client
            .stream("gemini-2.5-flash", &TextRequest::prompt("weather"))
            .await
            .unwrap();

        let mut saw_started = false;
        let mut saw_delta = false;
        let mut saw_ready = false;
        let mut finish_reason = None;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::ToolCallStarted { id, name, index } => {
                    saw_started = true;
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "get_weather");
                    assert_eq!(index, 0);
                }
                StreamEvent::ToolCallDelta { input_delta, .. } => {
                    saw_delta = true;
                    assert!(input_delta.contains("Paris"));
                }
                StreamEvent::ToolCallReady {
                    input,
                    provider_metadata,
                    ..
                } => {
                    saw_ready = true;
                    assert_eq!(input["location"], "Paris");
                    assert_eq!(
                        provider_metadata.unwrap()["gemini"]["thoughtSignature"],
                        "sig_123"
                    );
                }
                StreamEvent::Finished {
                    finish_reason: r, ..
                } => finish_reason = Some(r),
                _ => {}
            }
        }

        mock.assert_async().await;
        assert!(saw_started);
        assert!(saw_delta);
        assert!(saw_ready);
        assert!(matches!(finish_reason.unwrap(), FinishReason::ToolUse));
    }

    #[tokio::test]
    async fn stream_reports_malformed_json() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            )
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body("data: not-json\n\n")
            .create_async()
            .await;

        let client = GeminiClient::with_base_url("key".to_string(), server.url());
        let mut stream = client
            .stream("gemini-2.5-flash", &TextRequest::prompt("hi"))
            .await
            .unwrap();

        let err = stream.next().await.unwrap().unwrap_err();
        mock.assert_async().await;
        assert!(matches!(err, SdkError::Serialization(_)));
    }
}

use std::{
    collections::VecDeque,
    convert::Infallible,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_stream::stream;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    core::{
        error::SdkError,
        message::{Message, MessagePart, ToolCall},
        model::StreamingLanguageModel,
        request::TextRequest,
        stream::StreamEvent,
        tool::ToolDefinition,
        types::FinishReason,
    },
    runtime::{
        stream::stream_text,
        tools::ToolRegistry,
        turn::{ContinuationBuilder, TurnAccumulator},
    },
};

/// One framework-independent SSE byte chunk for the UI-message stream protocol.
pub type MessageStreamChunk = Result<Bytes, Infallible>;

/// Content-Type value to set on HTTP responses that stream UI messages.
pub const MESSAGE_STREAM_CONTENT_TYPE: &str = "text/event-stream";
/// Cache-Control value suitable for server-sent event responses.
pub const MESSAGE_STREAM_CACHE_CONTROL: &str = "no-cache";
/// HTTP response header that identifies the UI-message stream protocol version.
pub const MESSAGE_STREAM_PROTOCOL_HEADER: &str = "x-vercel-ai-ui-message-stream";
/// Protocol version value for [`MESSAGE_STREAM_PROTOCOL_HEADER`].
pub const MESSAGE_STREAM_PROTOCOL_VERSION: &str = "v1";

/// Deserialized request body accepted by the message-stream adapter.
#[derive(Debug, Deserialize)]
pub struct MessageStreamRequest {
    messages: Vec<MessageStreamMessage>,
}

/// Error returned when inbound UI messages violate this adapter's trust contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageStreamInputError {
    /// A client supplied a role that is not accepted by the adapter.
    UnknownRole {
        /// Zero-based index of the message.
        message_index: usize,
        /// Rejected role value.
        role: String,
    },
    /// A client attempted to supply a privileged system instruction.
    ClientSystemRole {
        /// Zero-based index of the rejected message.
        message_index: usize,
    },
    /// The converted SDK request violated a request invariant.
    InvalidRequest {
        /// Provider-neutral validation message.
        message: String,
    },
}

impl fmt::Display for MessageStreamInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRole {
                message_index,
                role,
            } => write!(f, "message {message_index} has unsupported role '{role}'",),
            Self::ClientSystemRole { message_index } => write!(
                f,
                "message {message_index} uses the system role; system instructions must be supplied by the server",
            ),
            Self::InvalidRequest { message } => {
                write!(f, "invalid message-stream request: {message}")
            }
        }
    }
}

impl std::error::Error for MessageStreamInputError {}

#[derive(Debug, Deserialize)]
struct MessageStreamMessage {
    role: String,
    #[serde(default)]
    parts: Vec<MessageStreamPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum MessageStreamPart {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Runtime limits and model options used while composing and streaming messages.
#[derive(Debug, Clone, Copy)]
pub struct MessageStreamOptions {
    /// Maximum number of model/tool continuation steps.
    pub max_model_steps: usize,
    /// Maximum provider output tokens per model step.
    pub max_output_tokens: u32,
    /// Provider-neutral sampling temperature.
    pub temperature: f32,
    /// Maximum number of independent tool calls executed concurrently.
    pub max_parallel_tools: usize,
}

impl Default for MessageStreamOptions {
    fn default() -> Self {
        Self {
            max_model_steps: 5,
            max_output_tokens: 800,
            temperature: 0.7,
            max_parallel_tools: 8,
        }
    }
}

/// Maps internal SDK failures to safe browser- and model-visible values.
///
/// The mapper runs at the HTTP/UI protocol boundary. Implementations may log
/// the detailed [`SdkError`] before returning a redacted public value.
pub trait MessageStreamErrorMapper: Send + Sync {
    /// Map a model or stream failure to browser-visible text.
    fn stream_error(&self, error: &SdkError) -> String;

    /// Map a tool failure to the value shown to the browser and sent to the model.
    fn tool_error(&self, call: &ToolCall, error: &SdkError) -> Value;
}

/// Safe default mapping that does not expose provider or application details.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedactedMessageStreamErrors;

impl MessageStreamErrorMapper for RedactedMessageStreamErrors {
    fn stream_error(&self, _error: &SdkError) -> String {
        "The model request could not be completed.".to_string()
    }

    fn tool_error(&self, _call: &ToolCall, _error: &SdkError) -> Value {
        json!({
            "error": {
                "code": "tool_execution_failed",
                "message": "The tool could not be completed."
            }
        })
    }
}

/// Convert a UI-message stream request into an SDK [`TextRequest`].
pub fn compose_text_request(
    request: MessageStreamRequest,
    system_prompt: impl Into<String>,
    options: MessageStreamOptions,
    tools: impl IntoIterator<Item = ToolDefinition>,
) -> Result<TextRequest, MessageStreamInputError> {
    let messages = messages_to_sdk_messages(request, system_prompt)?;

    TextRequest::builder()
        .messages(messages)
        .max_output_tokens(options.max_output_tokens)
        .temperature(options.temperature)
        .tools(tools.into_iter().collect())
        .build()
        .map_err(|error| MessageStreamInputError::InvalidRequest {
            message: error.message().to_string(),
        })
}

/// Stream a UI-message response from a provider model and application-owned tools.
pub fn stream_text_messages<M>(
    model: M,
    request: TextRequest,
    tools: ToolRegistry,
    options: MessageStreamOptions,
) -> impl Stream<Item = MessageStreamChunk> + Send + 'static
where
    M: StreamingLanguageModel + Send + Sync + 'static,
{
    stream_text_messages_with_error_mapper(
        model,
        request,
        tools,
        options,
        RedactedMessageStreamErrors,
    )
}

/// Stream a UI-message response with an application-defined public error mapper.
pub fn stream_text_messages_with_error_mapper<M, E>(
    model: M,
    request: TextRequest,
    tools: ToolRegistry,
    options: MessageStreamOptions,
    error_mapper: E,
) -> impl Stream<Item = MessageStreamChunk> + Send + 'static
where
    M: StreamingLanguageModel + Send + Sync + 'static,
    E: MessageStreamErrorMapper + 'static,
{
    stream_message_response(model, request, tools, options, Arc::new(error_mapper))
}

fn stream_message_response<M>(
    model: M,
    mut request: TextRequest,
    tools: ToolRegistry,
    options: MessageStreamOptions,
    error_mapper: Arc<dyn MessageStreamErrorMapper>,
) -> impl Stream<Item = MessageStreamChunk> + Send + 'static
where
    M: StreamingLanguageModel + Send + Sync + 'static,
{
    stream! {
        yield start_message_chunk();

        for step_index in 0..options.max_model_steps {
            yield sse(json!({ "type": "start-step" }));

            let sdk_stream = match stream_text(&model, request.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    for chunk in error_chunks(error_mapper.stream_error(&error)) {
                        yield chunk;
                    }
                    return;
                }
            };

            let text_part_id = format!("text_{step_index}");
            let mut turn = TurnAccumulator::default();
            let mut text_started = false;
            let mut fallback_tool_ids = VecDeque::new();
            let mut finished = false;
            futures_util::pin_mut!(sdk_stream);

            while let Some(event) = sdk_stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(error) => {
                        for chunk in error_chunks(error_mapper.stream_error(&error)) {
                            yield chunk;
                        }
                        return;
                    }
                };

                track_fallback_tool_id(&event, &mut fallback_tool_ids);
                let is_finished = matches!(event, StreamEvent::Finished { .. });
                for chunk in chunks_for_sdk_event(event.clone(), &text_part_id, &mut text_started) {
                    yield chunk;
                }
                turn.push_event(event);

                if is_finished {
                    finished = true;
                    break;
                }
            }

            if !finished {
                let error = SdkError::stream_terminated(
                    Some(model.provider_name()),
                    "stream ended before a terminal Finished event",
                );
                for chunk in error_chunks(error_mapper.stream_error(&error)) {
                    yield chunk;
                }
                return;
            }

            if text_started {
                yield sse(json!({ "type": "text-end", "id": text_part_id }));
            }

            let accumulated_turn = match turn.into_accumulated() {
                Ok(turn) => turn,
                Err(error) => {
                    for chunk in error_chunks(error_mapper.stream_error(&error)) {
                        yield chunk;
                    }
                    return;
                }
            };
            let finish_reason = accumulated_turn.finish_reason.clone();
            let assistant_parts =
                normalize_tool_call_ids(accumulated_turn.parts, fallback_tool_ids);
            let tool_calls = tool_calls_from_parts(&assistant_parts);
            yield sse(json!({ "type": "finish-step" }));

            if tool_calls.is_empty() {
                yield sse(json!({
                    "type": "finish",
                    "finishReason": finish_reason_to_ai_sdk(&finish_reason),
                }));
                yield done();
                return;
            }

            let tool_step = execute_tool_calls(
                &tools,
                &tool_calls,
                options.max_parallel_tools,
                Arc::clone(&error_mapper),
            )
            .await;
            let ToolExecutionStep {
                chunks,
                results,
            } = tool_step;

            for chunk in chunks {
                yield chunk;
            }

            request = build_continuation_request(request, assistant_parts, results);
        }

        yield sse(json!({
            "type": "error",
            "errorText": format!(
                "Stopped after {} tool/model steps to avoid an infinite loop.",
                options.max_model_steps,
            )
        }));
        yield sse(json!({ "type": "finish", "finishReason": "error" }));
        yield done();
    }
}

struct ToolExecutionStep {
    chunks: Vec<MessageStreamChunk>,
    results: Vec<(String, Value)>,
}

fn track_fallback_tool_id(event: &StreamEvent, fallback_tool_ids: &mut VecDeque<String>) {
    if let StreamEvent::ToolCallReady { id, index, .. } = event
        && id.is_empty()
    {
        fallback_tool_ids.push_back(format!("tool_call_{index}"));
    }
}

async fn execute_tool_calls(
    tools: &ToolRegistry,
    tool_calls: &[ToolCall],
    max_parallel_tools: usize,
    error_mapper: Arc<dyn MessageStreamErrorMapper>,
) -> ToolExecutionStep {
    let executions = futures_util::stream::iter(tool_calls.iter().cloned())
        .map(|call| {
            let tools = tools.clone();
            let error_mapper = Arc::clone(&error_mapper);
            async move {
                let output = match tools.execute(&call).await {
                    Ok(output) => output,
                    Err(error) => error_mapper.tool_error(&call, &error),
                };
                (call, output)
            }
        })
        .buffered(max_parallel_tools.max(1))
        .collect::<Vec<_>>()
        .await;

    let mut chunks = Vec::with_capacity(executions.len());
    let mut results = Vec::with_capacity(executions.len());

    for (call, output) in executions {
        chunks.push(sse(json!({
            "type": "tool-output-available",
            "toolCallId": call.id,
            "output": &output,
        })));
        results.push((call.id, output));
    }

    ToolExecutionStep { chunks, results }
}

fn build_continuation_request(
    request: TextRequest,
    assistant_parts: Vec<MessagePart>,
    tool_results: Vec<(String, Value)>,
) -> TextRequest {
    let mut builder =
        ContinuationBuilder::from_request(request).with_assistant_turn(assistant_parts);

    for (tool_call_id, output) in tool_results {
        builder = builder.with_tool_result(tool_call_id, output);
    }

    builder.build()
}

/// Convert UI-message request messages into SDK messages with a leading system prompt.
pub fn messages_to_sdk_messages(
    request: MessageStreamRequest,
    system_prompt: impl Into<String>,
) -> Result<Vec<Message>, MessageStreamInputError> {
    let mut sdk_messages = vec![Message::system(system_prompt)];

    for (message_index, message) in request.messages.into_iter().enumerate() {
        let role = sdk_role_for_message(&message.role, message_index)?;
        let text = text_for_message_parts(message.parts);

        if text.trim().is_empty() {
            continue;
        }

        match role {
            MessageStreamRole::User => sdk_messages.push(Message::user(text)),
            MessageStreamRole::Assistant => sdk_messages.push(Message::assistant(text)),
        }
    }

    Ok(sdk_messages)
}

fn sdk_role_for_message(
    role: &str,
    message_index: usize,
) -> Result<MessageStreamRole, MessageStreamInputError> {
    match role {
        "system" => Err(MessageStreamInputError::ClientSystemRole { message_index }),
        "user" => Ok(MessageStreamRole::User),
        "assistant" => Ok(MessageStreamRole::Assistant),
        _ => Err(MessageStreamInputError::UnknownRole {
            message_index,
            role: role.to_string(),
        }),
    }
}

fn text_for_message_parts(parts: Vec<MessageStreamPart>) -> String {
    let mut text_parts = Vec::new();

    for part in parts {
        match part {
            MessageStreamPart::Text { text } => text_parts.push(text),
            MessageStreamPart::Other => {}
        }
    }

    text_parts.join("\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageStreamRole {
    User,
    Assistant,
}

fn start_message_chunk() -> MessageStreamChunk {
    sse(json!({ "type": "start", "messageId": message_id() }))
}

fn error_chunks(error_text: impl Into<String>) -> Vec<MessageStreamChunk> {
    vec![
        sse(json!({ "type": "error", "errorText": error_text.into() })),
        sse(json!({ "type": "finish-step" })),
        sse(json!({ "type": "finish", "finishReason": "error" })),
        done(),
    ]
}

fn chunks_for_sdk_event(
    event: StreamEvent,
    text_part_id: &str,
    text_started: &mut bool,
) -> Vec<MessageStreamChunk> {
    match event {
        StreamEvent::TextDelta(delta) => {
            let mut chunks = Vec::new();

            if !*text_started {
                *text_started = true;
                chunks.push(sse(json!({
                    "type": "text-start",
                    "id": text_part_id,
                })));
            }

            chunks.push(sse(json!({
                "type": "text-delta",
                "id": text_part_id,
                "delta": delta,
            })));

            chunks
        }
        StreamEvent::ToolCallStarted { id, name, index } => vec![sse(json!({
            "type": "tool-input-start",
            "toolCallId": tool_call_id(id, index),
            "toolName": name,
        }))],
        StreamEvent::ToolCallDelta {
            id,
            index,
            input_delta,
        } => vec![sse(json!({
            "type": "tool-input-delta",
            "toolCallId": tool_call_id(id, index),
            "inputTextDelta": input_delta,
        }))],
        StreamEvent::ToolCallReady {
            id,
            name,
            index,
            input,
            ..
        } => vec![sse(json!({
            "type": "tool-input-available",
            "toolCallId": tool_call_id(id, index),
            "toolName": name,
            "input": input,
        }))],
        StreamEvent::Finished { .. } => Vec::new(),
    }
}

fn normalize_tool_call_ids(
    parts: Vec<MessagePart>,
    mut fallback_tool_ids: VecDeque<String>,
) -> Vec<MessagePart> {
    parts
        .into_iter()
        .map(|part| match part {
            MessagePart::ToolCall(mut call) if call.id.is_empty() => {
                call.id = fallback_tool_ids
                    .pop_front()
                    .unwrap_or_else(|| "tool_call_0".to_string());
                MessagePart::ToolCall(call)
            }
            part => part,
        })
        .collect()
}

fn tool_calls_from_parts(parts: &[MessagePart]) -> Vec<crate::core::message::ToolCall> {
    parts
        .iter()
        .filter_map(|part| {
            if let MessagePart::ToolCall(tool_call) = part {
                Some(tool_call.clone())
            } else {
                None
            }
        })
        .collect()
}

fn sse(value: Value) -> MessageStreamChunk {
    Ok(Bytes::from(format!("data: {value}\n\n")))
}

fn done() -> MessageStreamChunk {
    Ok(Bytes::from_static(b"data: [DONE]\n\n"))
}

fn message_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("msg_{millis}_{}_{sequence}", std::process::id())
}

fn tool_call_id(id: String, index: u32) -> String {
    if id.is_empty() {
        format!("tool_call_{index}")
    } else {
        id
    }
}

fn finish_reason_to_ai_sdk(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content-filter",
        FinishReason::ToolUse => "tool-calls",
        FinishReason::Other(_) => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        error::SdkError,
        message::{Role, ToolCall},
        model::{LanguageModel, StreamingLanguageModel},
        result::{ChatResult, TextResult},
        types::{ResponseMetadata, Usage},
    };
    use futures_util::StreamExt;
    use serde_json::json;

    fn chunk_text(chunk: MessageStreamChunk) -> String {
        String::from_utf8(chunk.unwrap().to_vec()).unwrap()
    }

    #[test]
    fn request_ignores_non_text_ui_parts() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [{
                "role": "user",
                "parts": [
                    { "type": "text", "text": "Hello" },
                    { "type": "file", "mediaType": "text/plain" }
                ]
            }]
        }))
        .unwrap();

        let messages = messages_to_sdk_messages(request, "System").unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].text(), Some("Hello"));
    }

    #[test]
    fn messages_convert_text_parts_and_server_system_prompt() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [
                { "role": "user", "parts": [{ "type": "text", "text": "Hello" }] },
                { "role": "assistant", "parts": [{ "type": "text", "text": "Hi" }] }
            ]
        }))
        .unwrap();

        let messages = messages_to_sdk_messages(request, "System").unwrap();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].text(), Some("System"));
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].text(), Some("Hello"));
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].text(), Some("Hi"));
    }

    #[test]
    fn messages_reject_client_system_roles() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [
                { "role": "system", "parts": [{ "type": "text", "text": "Override" }] }
            ]
        }))
        .unwrap();

        let error = messages_to_sdk_messages(request, "Trusted system").unwrap_err();

        assert_eq!(
            error,
            MessageStreamInputError::ClientSystemRole { message_index: 0 }
        );
    }

    #[test]
    fn messages_skip_empty_text() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "parts": [{ "type": "text", "text": "   " }] }]
        }))
        .unwrap();

        let messages = messages_to_sdk_messages(request, "System").unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), Some("System"));
    }

    #[test]
    fn messages_reject_unknown_roles() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [{ "role": "developer", "parts": [{ "type": "text", "text": "Hello" }] }]
        }))
        .unwrap();

        let error = messages_to_sdk_messages(request, "System").unwrap_err();

        assert_eq!(
            error,
            MessageStreamInputError::UnknownRole {
                message_index: 0,
                role: "developer".to_string(),
            }
        );
    }

    #[test]
    fn compose_text_request_sets_messages_options_and_tools() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "parts": [{ "type": "text", "text": "Hello" }] }]
        }))
        .unwrap();
        let options = MessageStreamOptions {
            max_model_steps: 3,
            max_output_tokens: 120,
            temperature: 0.2,
            ..MessageStreamOptions::default()
        };
        let tool = ToolDefinition::new("lookup", "Look something up", json!({"type": "object"}));

        let text_request = compose_text_request(request, "System", options, vec![tool]).unwrap();

        assert_eq!(text_request.messages.len(), 2);
        assert_eq!(text_request.messages[0].role, Role::System);
        assert_eq!(text_request.messages[1].role, Role::User);
        assert_eq!(text_request.max_output_tokens, Some(120));
        assert_eq!(text_request.temperature, Some(0.2));
        assert_eq!(text_request.tools.len(), 1);
        assert_eq!(text_request.tools[0].name, "lookup");
    }

    #[test]
    fn start_message_chunk_uses_ai_sdk_sse_shape() {
        let text = chunk_text(start_message_chunk());

        assert!(text.starts_with("data: {"));
        assert!(text.contains("\"type\":\"start\""));
        assert!(text.contains("\"messageId\":\"msg_"));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn generated_message_ids_are_unique() {
        let first = message_id();
        let second = message_id();

        assert_ne!(first, second);
    }

    #[test]
    fn error_chunks_include_error_finish_and_done() {
        let chunks = error_chunks("failed")
            .into_iter()
            .map(chunk_text)
            .collect::<Vec<_>>();

        assert_eq!(chunks.len(), 4);
        assert!(chunks[0].contains("\"type\":\"error\""));
        assert!(chunks[0].contains("\"errorText\":\"failed\""));
        assert!(chunks[1].contains("\"type\":\"finish-step\""));
        assert!(chunks[2].contains("\"finishReason\":\"error\""));
        assert_eq!(chunks[3], "data: [DONE]\n\n");
    }

    #[test]
    fn default_error_mapper_redacts_internal_details() {
        let mapper = RedactedMessageStreamErrors;
        let internal = SdkError::Unknown("database password appeared here".to_string());
        let call = ToolCall::new("call_1", "lookup", json!({}));

        let stream_error = mapper.stream_error(&internal);
        let tool_error = mapper.tool_error(&call, &internal).to_string();

        assert!(!stream_error.contains("password"));
        assert!(!tool_error.contains("password"));
        assert!(tool_error.contains("tool_execution_failed"));
    }

    #[test]
    fn finish_chunk_maps_finish_reason() {
        assert_eq!(finish_reason_to_ai_sdk(&FinishReason::Stop), "stop");
        assert_eq!(finish_reason_to_ai_sdk(&FinishReason::Length), "length");
        assert_eq!(
            finish_reason_to_ai_sdk(&FinishReason::ContentFilter),
            "content-filter"
        );
        assert_eq!(
            finish_reason_to_ai_sdk(&FinishReason::ToolUse),
            "tool-calls"
        );
        assert_eq!(
            finish_reason_to_ai_sdk(&FinishReason::Other("x".to_string())),
            "other"
        );
    }

    #[test]
    fn sdk_events_convert_to_tool_chunks_with_fallback_ids() {
        let ready_chunks = chunks_for_sdk_event(
            StreamEvent::ToolCallReady {
                id: String::new(),
                name: "lookup".to_string(),
                index: 2,
                input: json!({ "key": "value" }),
                provider_metadata: None,
            },
            "text_0",
            &mut false,
        )
        .into_iter()
        .map(chunk_text)
        .collect::<Vec<_>>();

        assert_eq!(ready_chunks.len(), 1);
        assert!(ready_chunks[0].contains("\"type\":\"tool-input-available\""));
        assert!(ready_chunks[0].contains("\"toolCallId\":\"tool_call_2\""));
        assert!(ready_chunks[0].contains("\"toolName\":\"lookup\""));
    }

    #[test]
    fn empty_tool_call_ids_are_normalized_for_continuation() {
        let parts = normalize_tool_call_ids(
            vec![MessagePart::ToolCall(ToolCall::new(
                "",
                "lookup",
                json!({ "key": "value" }),
            ))],
            VecDeque::from(["tool_call_2".to_string()]),
        );

        let tool_calls = tool_calls_from_parts(&parts);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tool_call_2");
    }

    #[tokio::test]
    async fn stream_finishes_after_sdk_finished_even_if_stream_stays_open() {
        struct NeverEndingAfterFinished;

        #[async_trait::async_trait]
        impl LanguageModel for NeverEndingAfterFinished {
            async fn generate(&self, _request: TextRequest) -> Result<TextResult, SdkError> {
                unimplemented!()
            }

            async fn generate_chat(&self, _request: TextRequest) -> Result<ChatResult, SdkError> {
                unimplemented!()
            }

            fn model_id(&self) -> &str {
                "never-ending"
            }

            fn provider_name(&self) -> &str {
                "test"
            }
        }

        #[async_trait::async_trait]
        impl StreamingLanguageModel for NeverEndingAfterFinished {
            async fn stream(
                &self,
                _request: TextRequest,
            ) -> Result<crate::core::stream::TextEventStream, SdkError> {
                Ok(Box::pin(
                    futures_util::stream::once(async {
                        Ok(StreamEvent::Finished {
                            finish_reason: FinishReason::Stop,
                            usage: Option::<Usage>::None,
                            response: ResponseMetadata {
                                id: Some("r1".to_string()),
                                model: Some("m1".to_string()),
                            },
                        })
                    })
                    .chain(futures_util::stream::pending()),
                ))
            }
        }

        let stream = stream_text_messages(
            NeverEndingAfterFinished,
            TextRequest::prompt("hi"),
            ToolRegistry::new(),
            MessageStreamOptions {
                max_model_steps: 1,
                ..MessageStreamOptions::default()
            },
        );
        futures_util::pin_mut!(stream);

        let _start = stream.next().await.unwrap();
        let _start_step = stream.next().await.unwrap();
        let next = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .expect("message stream should finish after SDK Finished");

        assert!(
            chunk_text(next.unwrap()).contains("\"type\":\"finish-step\""),
            "expected finish-step after SDK Finished"
        );
    }

    #[tokio::test]
    async fn stream_yields_text_before_provider_finishes() {
        struct PendingAfterDelta;

        #[async_trait::async_trait]
        impl LanguageModel for PendingAfterDelta {
            async fn generate(&self, _request: TextRequest) -> Result<TextResult, SdkError> {
                unimplemented!()
            }

            fn model_id(&self) -> &str {
                "pending-after-delta"
            }

            fn provider_name(&self) -> &str {
                "test"
            }
        }

        #[async_trait::async_trait]
        impl StreamingLanguageModel for PendingAfterDelta {
            async fn stream(
                &self,
                _request: TextRequest,
            ) -> Result<crate::core::stream::TextEventStream, SdkError> {
                Ok(Box::pin(
                    futures_util::stream::once(async {
                        Ok(StreamEvent::TextDelta("hello".to_string()))
                    })
                    .chain(futures_util::stream::pending()),
                ))
            }
        }

        let stream = stream_text_messages(
            PendingAfterDelta,
            TextRequest::prompt("hi"),
            ToolRegistry::new(),
            MessageStreamOptions::default(),
        );
        futures_util::pin_mut!(stream);

        let _start = stream.next().await.unwrap();
        let _start_step = stream.next().await.unwrap();
        let text_start = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .expect("text-start should arrive before provider completion")
            .unwrap();
        let text_delta = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .expect("text-delta should arrive before provider completion")
            .unwrap();

        assert!(chunk_text(text_start).contains("\"type\":\"text-start\""));
        assert!(chunk_text(text_delta).contains("\"delta\":\"hello\""));
    }
}

use std::{collections::VecDeque, convert::Infallible, fmt};

use async_stream::stream;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    core::{
        message::{Message, MessagePart, ToolCall},
        model::LanguageModel,
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

/// Error returned when inbound UI messages do not match this adapter's
/// text-only ingestion contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageStreamInputError {
    UnsupportedPart {
        message_index: usize,
        part_index: usize,
    },
    UnknownRole {
        message_index: usize,
        role: String,
    },
}

impl fmt::Display for MessageStreamInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPart {
                message_index,
                part_index,
            } => write!(
                f,
                "message {message_index} part {part_index} is not supported by the text-only message stream adapter",
            ),
            Self::UnknownRole {
                message_index,
                role,
            } => write!(f, "message {message_index} has unsupported role '{role}'",),
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
    pub max_model_steps: usize,
    pub max_output_tokens: u32,
    pub temperature: f32,
}

impl Default for MessageStreamOptions {
    fn default() -> Self {
        Self {
            max_model_steps: 5,
            max_output_tokens: 800,
            temperature: 0.7,
        }
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

    Ok(TextRequest::builder()
        .messages(messages)
        .max_output_tokens(options.max_output_tokens)
        .temperature(options.temperature)
        .tools(tools.into_iter().collect())
        .build())
}

/// Stream a UI-message response from a provider model and application-owned tools.
pub fn stream_text_messages<M>(
    model: M,
    request: TextRequest,
    tools: ToolRegistry,
    options: MessageStreamOptions,
) -> impl Stream<Item = MessageStreamChunk> + Send + 'static
where
    M: LanguageModel + Send + Sync + 'static,
{
    stream_message_response(model, request, tools, options)
}

fn stream_message_response<M>(
    model: M,
    mut request: TextRequest,
    tools: ToolRegistry,
    options: MessageStreamOptions,
) -> impl Stream<Item = MessageStreamChunk> + Send + 'static
where
    M: LanguageModel + Send + Sync + 'static,
{
    stream! {
        yield start_message_chunk();

        for step_index in 0..options.max_model_steps {
            let model_step = match run_model_step(&model, request.clone(), step_index).await {
                Ok(model_step) => model_step,
                Err(error_text) => {
                    for chunk in error_chunks(error_text) {
                        yield chunk;
                    }
                    return;
                }
            };

            for chunk in model_step.chunks {
                yield chunk;
            }

            if model_step.tool_calls.is_empty() {
                yield sse(json!({
                    "type": "finish",
                    "finishReason": finish_reason_to_ai_sdk(&model_step.finish_reason),
                }));
                yield done();
                return;
            }

            let tool_step = execute_tool_calls(&tools, &model_step.tool_calls).await;
            let ToolExecutionStep {
                chunks,
                results,
            } = tool_step;

            for chunk in chunks {
                yield chunk;
            }

            request = build_continuation_request(request, model_step.assistant_parts, results);
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

struct ModelStep {
    chunks: Vec<MessageStreamChunk>,
    assistant_parts: Vec<MessagePart>,
    tool_calls: Vec<ToolCall>,
    finish_reason: FinishReason,
}

struct ToolExecutionStep {
    chunks: Vec<MessageStreamChunk>,
    results: Vec<(String, Value)>,
}

async fn run_model_step<M>(
    model: &M,
    request: TextRequest,
    step_index: usize,
) -> Result<ModelStep, String>
where
    M: LanguageModel + Send + Sync + 'static,
{
    let mut chunks = vec![sse(json!({ "type": "start-step" }))];
    let sdk_stream = stream_text(model, request)
        .await
        .map_err(|error| format!("SDK stream failed: {error}"))?;
    let text_part_id = format!("text_{step_index}");
    let streamed_turn = collect_streamed_turn(sdk_stream, &text_part_id, &mut chunks).await?;

    if streamed_turn.text_started {
        chunks.push(sse(json!({ "type": "text-end", "id": text_part_id })));
    }

    let accumulated_turn = streamed_turn.turn.into_accumulated();
    let finish_reason = accumulated_turn.finish_reason.clone();
    let assistant_parts =
        normalize_tool_call_ids(accumulated_turn.parts, streamed_turn.fallback_tool_ids);
    let tool_calls = tool_calls_from_parts(&assistant_parts);

    chunks.push(sse(json!({ "type": "finish-step" })));

    Ok(ModelStep {
        chunks,
        assistant_parts,
        tool_calls,
        finish_reason,
    })
}

struct StreamedTurn {
    turn: TurnAccumulator,
    text_started: bool,
    fallback_tool_ids: VecDeque<String>,
}

async fn collect_streamed_turn(
    sdk_stream: crate::core::stream::TextEventStream,
    text_part_id: &str,
    chunks: &mut Vec<MessageStreamChunk>,
) -> Result<StreamedTurn, String> {
    let mut turn = TurnAccumulator::default();
    let mut text_started = false;
    let mut fallback_tool_ids = VecDeque::new();
    futures_util::pin_mut!(sdk_stream);

    while let Some(event) = sdk_stream.next().await {
        let event = event.map_err(|error| format!("SDK event failed: {error}"))?;

        track_fallback_tool_id(&event, &mut fallback_tool_ids);

        let is_finished = matches!(event, StreamEvent::Finished { .. });
        turn.push_event(event.clone());
        chunks.extend(chunks_for_sdk_event(event, text_part_id, &mut text_started));

        if is_finished {
            break;
        }
    }

    Ok(StreamedTurn {
        turn,
        text_started,
        fallback_tool_ids,
    })
}

fn track_fallback_tool_id(event: &StreamEvent, fallback_tool_ids: &mut VecDeque<String>) {
    if let StreamEvent::ToolCallReady { id, index, .. } = event
        && id.is_empty()
    {
        fallback_tool_ids.push_back(format!("tool_call_{index}"));
    }
}

async fn execute_tool_calls(tools: &ToolRegistry, tool_calls: &[ToolCall]) -> ToolExecutionStep {
    let mut chunks = Vec::new();
    let mut results = Vec::new();

    for call in tool_calls {
        let output = match tools.execute(call).await {
            Ok(output) => output,
            Err(error) => json!({ "error": error.to_string() }),
        };

        chunks.push(sse(json!({
            "type": "tool-output-available",
            "toolCallId": &call.id,
            "output": &output,
        })));
        results.push((call.id.clone(), output));
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
        let text = text_for_message_parts(message.parts, message_index)?;

        if text.trim().is_empty() {
            continue;
        }

        match role {
            MessageStreamRole::System => sdk_messages.push(Message::system(text)),
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
        "system" => Ok(MessageStreamRole::System),
        "user" => Ok(MessageStreamRole::User),
        "assistant" => Ok(MessageStreamRole::Assistant),
        _ => Err(MessageStreamInputError::UnknownRole {
            message_index,
            role: role.to_string(),
        }),
    }
}

fn text_for_message_parts(
    parts: Vec<MessageStreamPart>,
    message_index: usize,
) -> Result<String, MessageStreamInputError> {
    let mut text_parts = Vec::new();

    for (part_index, part) in parts.into_iter().enumerate() {
        match part {
            MessageStreamPart::Text { text } => text_parts.push(text),
            MessageStreamPart::Other => {
                return Err(MessageStreamInputError::UnsupportedPart {
                    message_index,
                    part_index,
                });
            }
        }
    }

    Ok(text_parts.join("\n"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageStreamRole {
    System,
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
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("msg_{millis}")
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
        model::LanguageModel,
        result::{ChatResult, TextResult},
        types::{ResponseMetadata, Usage},
    };
    use futures_util::StreamExt;
    use serde_json::json;

    fn chunk_text(chunk: MessageStreamChunk) -> String {
        String::from_utf8(chunk.unwrap().to_vec()).unwrap()
    }

    #[test]
    fn request_rejects_unsupported_parts() {
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

        let error = messages_to_sdk_messages(request, "System").unwrap_err();

        assert_eq!(
            error,
            MessageStreamInputError::UnsupportedPart {
                message_index: 0,
                part_index: 1,
            }
        );
    }

    #[test]
    fn messages_convert_text_parts_and_system_prompt() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [
                { "role": "user", "parts": [{ "type": "text", "text": "Hello" }] },
                { "role": "assistant", "parts": [{ "type": "text", "text": "Hi" }] },
                { "role": "system", "parts": [{ "type": "text", "text": "Extra system" }] }
            ]
        }))
        .unwrap();

        let messages = messages_to_sdk_messages(request, "System").unwrap();

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "System");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].content, "Hi");
        assert_eq!(messages[3].role, Role::System);
        assert_eq!(messages[3].content, "Extra system");
    }

    #[test]
    fn messages_skip_empty_text() {
        let request: MessageStreamRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "parts": [{ "type": "text", "text": "   " }] }]
        }))
        .unwrap();

        let messages = messages_to_sdk_messages(request, "System").unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "System");
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

            fn model_id(&self) -> &str {
                "never-ending"
            }

            fn provider_name(&self) -> &str {
                "test"
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
}

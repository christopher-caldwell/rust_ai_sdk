use std::{collections::VecDeque, convert::Infallible};

use async_stream::stream;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    core::{
        message::{Message, MessagePart},
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

pub type MessageStreamChunk = Result<Bytes, Infallible>;

pub const MESSAGE_STREAM_CONTENT_TYPE: &str = "text/event-stream";
pub const MESSAGE_STREAM_CACHE_CONTROL: &str = "no-cache";
pub const MESSAGE_STREAM_PROTOCOL_HEADER: &str = "x-vercel-ai-ui-message-stream";
pub const MESSAGE_STREAM_PROTOCOL_VERSION: &str = "v1";

#[derive(Debug, Deserialize)]
pub struct MessageStreamRequest {
    messages: Vec<MessageStreamMessage>,
}

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

pub fn compose_text_request(
    request: MessageStreamRequest,
    system_prompt: impl Into<String>,
    options: MessageStreamOptions,
    tools: impl IntoIterator<Item = ToolDefinition>,
) -> TextRequest {
    TextRequest::builder()
        .messages(messages_to_sdk_messages(request, system_prompt))
        .max_output_tokens(options.max_output_tokens)
        .temperature(options.temperature)
        .tools(tools.into_iter().collect())
        .build()
}

pub fn stream_text_messages<M>(
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
            yield sse(json!({ "type": "start-step" }));

            let sdk_stream = match stream_text(&model, request.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    for chunk in error_chunks(format!("SDK stream failed: {error}")) {
                        yield chunk;
                    }
                    return;
                }
            };

            let mut turn = TurnAccumulator::default();
            let text_part_id = format!("text_{step_index}");
            let mut text_started = false;
            let mut fallback_tool_ids = VecDeque::new();
            futures_util::pin_mut!(sdk_stream);

            while let Some(event) = sdk_stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(error) => {
                        for chunk in error_chunks(format!("SDK event failed: {error}")) {
                            yield chunk;
                        }
                        return;
                    }
                };

                if let StreamEvent::ToolCallReady { id, index, .. } = &event
                    && id.is_empty()
                {
                    fallback_tool_ids.push_back(format!("tool_call_{index}"));
                }

                turn.push_event(event.clone());

                for chunk in chunks_for_sdk_event(event, &text_part_id, &mut text_started) {
                    yield chunk;
                }
            }

            if text_started {
                yield sse(json!({ "type": "text-end", "id": text_part_id }));
            }

            let accumulated_turn = turn.into_accumulated();
            let finish_reason = accumulated_turn.finish_reason.clone();
            let assistant_parts = normalize_tool_call_ids(accumulated_turn.parts, fallback_tool_ids);
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

            let mut builder =
                ContinuationBuilder::from_request(request).with_assistant_turn(assistant_parts);

            for call in &tool_calls {
                let output = match tools.execute(call).await {
                    Ok(output) => output,
                    Err(error) => json!({ "error": error.to_string() }),
                };
                yield sse(json!({
                    "type": "tool-output-available",
                    "toolCallId": &call.id,
                    "output": output,
                }));
                builder = builder.with_tool_result(&call.id, output.to_string());
            }

            request = builder.build();
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

pub fn messages_to_sdk_messages(
    request: MessageStreamRequest,
    system_prompt: impl Into<String>,
) -> Vec<Message> {
    let mut sdk_messages = vec![Message::system(system_prompt)];

    for message in request.messages {
        let text = message
            .parts
            .into_iter()
            .filter_map(|part| match part {
                MessageStreamPart::Text { text } => Some(text),
                MessageStreamPart::Other => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if text.trim().is_empty() {
            continue;
        }

        match message.role.as_str() {
            "system" => sdk_messages.push(Message::system(text)),
            "assistant" => sdk_messages.push(Message::assistant(text)),
            _ => sdk_messages.push(Message::user(text)),
        }
    }

    sdk_messages
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
    use crate::core::message::{Role, ToolCall};
    use serde_json::json;

    fn chunk_text(chunk: MessageStreamChunk) -> String {
        String::from_utf8(chunk.unwrap().to_vec()).unwrap()
    }

    #[test]
    fn request_deserializes_text_parts_and_ignores_unsupported_parts() {
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

        let messages = messages_to_sdk_messages(request, "System");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");
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

        let messages = messages_to_sdk_messages(request, "System");

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

        let messages = messages_to_sdk_messages(request, "System");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "System");
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

        let text_request = compose_text_request(request, "System", options, vec![tool]);

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
}

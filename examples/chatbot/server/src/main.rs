use std::{collections::BTreeMap, convert::Infallible, net::SocketAddr};

use ai_sdk::{
    core::{
        message::{Message, MessagePart, ToolCall},
        request::TextRequest,
        stream::StreamEvent,
        tool::ToolDefinition,
        types::FinishReason,
    },
    providers::openai::model::OpenAiChatModel,
    runtime::{stream::stream_text, turn::ContinuationBuilder},
};
use async_stream::stream;
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing_subscriber::fmt;

#[derive(Clone)]
struct AppState {
    openai_api_key: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<UiMessage>,
}

#[derive(Debug, Deserialize)]
struct UiMessage {
    role: String,
    #[serde(default)]
    parts: Vec<UiPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum UiPart {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    fmt::init();

    let openai_api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in server/.env");
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string());
    let port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3001);

    let state = AppState {
        openai_api_key,
        model,
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/chat", post(chat_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("chatbot server listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(input): Json<ChatRequest>,
) -> impl IntoResponse {
    let stream = chat_stream(state, input);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-vercel-ai-ui-message-stream", "v1")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn chat_stream(
    state: AppState,
    input: ChatRequest,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
    stream! {
        let message_id = message_id();
        yield Ok(sse_json(json!({ "type": "start", "messageId": message_id })));

        let mut request = TextRequest {
            messages: ui_messages_to_sdk_messages(input.messages),
            max_output_tokens: Some(800),
            temperature: Some(0.7),
            tools: demo_tools(),
            tool_choice: None,
        };

        let model = OpenAiChatModel::new(state.openai_api_key, state.model);

        for step_index in 0..5 {
            yield Ok(sse_json(json!({ "type": "start-step" })));

            let sdk_stream = match stream_text(&model, request.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    yield Ok(sse_json(json!({
                        "type": "error",
                        "errorText": format!("SDK stream failed: {error}")
                    })));
                    yield Ok(sse_json(json!({ "type": "finish-step" })));
                    yield Ok(sse_json(json!({ "type": "finish", "finishReason": "error" })));
                    yield Ok(sse_done());
                    return;
                }
            };

            let mut turn = StreamedTurn::new(step_index);
            futures_util::pin_mut!(sdk_stream);

            while let Some(event) = sdk_stream.next().await {
                match event {
                    Ok(StreamEvent::TextDelta(delta)) => {
                        if !turn.text_started {
                            turn.text_started = true;
                            yield Ok(sse_json(json!({
                                "type": "text-start",
                                "id": turn.text_part_id,
                            })));
                        }

                        turn.text.push_str(&delta);
                        yield Ok(sse_json(json!({
                            "type": "text-delta",
                            "id": turn.text_part_id,
                            "delta": delta,
                        })));
                    }
                    Ok(StreamEvent::ToolCallStarted { id, name, index }) => {
                        turn.tool_buffers.insert(index, ToolBuffer {
                            id: id.clone(),
                            name: name.clone(),
                            input_json: String::new(),
                        });
                        yield Ok(sse_json(json!({
                            "type": "tool-input-start",
                            "toolCallId": id,
                            "toolName": name,
                        })));
                    }
                    Ok(StreamEvent::ToolCallDelta { id: _, index, input_delta }) => {
                        if let Some(buffer) = turn.tool_buffers.get_mut(&index) {
                            buffer.input_json.push_str(&input_delta);
                        }
                        let tool_call_id = turn
                            .tool_buffers
                            .get(&index)
                            .map(|buffer| buffer.id.clone())
                            .unwrap_or_else(|| format!("tool_call_{index}"));
                        yield Ok(sse_json(json!({
                            "type": "tool-input-delta",
                            "toolCallId": tool_call_id,
                            "inputTextDelta": input_delta,
                        })));
                    }
                    Ok(StreamEvent::ToolCallReady { id, name, index, input }) => {
                        turn.tool_buffers.insert(index, ToolBuffer {
                            id: id.clone(),
                            name: name.clone(),
                            input_json: input.to_string(),
                        });
                        yield Ok(sse_json(json!({
                            "type": "tool-input-available",
                            "toolCallId": id,
                            "toolName": name,
                            "input": input,
                        })));
                    }
                    Ok(StreamEvent::Finished { finish_reason, .. }) => {
                        turn.finish_reason = Some(finish_reason);
                    }
                    Err(error) => {
                        yield Ok(sse_json(json!({
                            "type": "error",
                            "errorText": format!("SDK event failed: {error}")
                        })));
                        yield Ok(sse_json(json!({ "type": "finish-step" })));
                        yield Ok(sse_json(json!({ "type": "finish", "finishReason": "error" })));
                        yield Ok(sse_done());
                        return;
                    }
                }
            }

            if turn.text_started {
                yield Ok(sse_json(json!({ "type": "text-end", "id": turn.text_part_id })));
            }

            let finish_reason = turn.finish_reason.clone().unwrap_or(FinishReason::Stop);
            let (assistant_parts, tool_calls) = turn.into_parts();

            yield Ok(sse_json(json!({ "type": "finish-step" })));

            if tool_calls.is_empty() {
                yield Ok(sse_json(json!({
                    "type": "finish",
                    "finishReason": finish_reason_to_ai_sdk(&finish_reason),
                })));
                yield Ok(sse_done());
                return;
            }

            let mut builder =
                ContinuationBuilder::from_request(request).with_assistant_turn(assistant_parts);

            for call in &tool_calls {
                let output = execute_tool(call);
                yield Ok(sse_json(json!({
                    "type": "tool-output-available",
                    "toolCallId": &call.id,
                    "output": output,
                })));
                builder = builder.with_tool_result(&call.id, output.to_string());
            }

            request = builder.build();
        }

        yield Ok(sse_json(json!({
            "type": "error",
            "errorText": "Stopped after 5 tool/model steps to avoid an infinite loop."
        })));
        yield Ok(sse_json(json!({ "type": "finish", "finishReason": "error" })));
        yield Ok(sse_done());
    }
}

struct StreamedTurn {
    text_part_id: String,
    text_started: bool,
    text: String,
    tool_buffers: BTreeMap<u32, ToolBuffer>,
    finish_reason: Option<FinishReason>,
}

impl StreamedTurn {
    fn new(step_index: usize) -> Self {
        Self {
            text_part_id: format!("text_{step_index}"),
            text_started: false,
            text: String::new(),
            tool_buffers: BTreeMap::new(),
            finish_reason: None,
        }
    }

    fn into_parts(self) -> (Vec<MessagePart>, Vec<ToolCall>) {
        let mut parts = Vec::new();
        if !self.text.is_empty() {
            parts.push(MessagePart::Text(self.text));
        }

        let tool_calls: Vec<ToolCall> = self
            .tool_buffers
            .into_values()
            .map(|buffer| {
                let input = serde_json::from_str(&buffer.input_json)
                    .unwrap_or_else(|_| Value::String(buffer.input_json));
                ToolCall {
                    id: buffer.id,
                    name: buffer.name,
                    input,
                }
            })
            .collect();

        for call in &tool_calls {
            parts.push(MessagePart::ToolCall(call.clone()));
        }

        (parts, tool_calls)
    }
}

struct ToolBuffer {
    id: String,
    name: String,
    input_json: String,
}

fn ui_messages_to_sdk_messages(messages: Vec<UiMessage>) -> Vec<Message> {
    let mut sdk_messages = vec![Message::system(
        "You are a concise demo chatbot running behind an Axum server. Use tools when they are relevant, then explain the result naturally.",
    )];

    for message in messages {
        let text = message
            .parts
            .into_iter()
            .filter_map(|part| match part {
                UiPart::Text { text } => Some(text),
                UiPart::Other => None,
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

fn demo_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "get_weather",
            "Get a deterministic demo weather report for a city.",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, for example Paris or Chicago"
                    }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
        ),
        ToolDefinition::new(
            "get_current_time",
            "Get the current server time for a named timezone.",
            json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone label such as America/Chicago"
                    }
                },
                "required": ["timezone"],
                "additionalProperties": false
            }),
        ),
    ]
}

fn execute_tool(call: &ToolCall) -> Value {
    match call.name.as_str() {
        "get_weather" => {
            let location = call
                .input
                .get("location")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            fake_weather(location)
        }
        "get_current_time" => {
            let timezone = call
                .input
                .get("timezone")
                .and_then(Value::as_str)
                .unwrap_or("America/Chicago");
            let unix_seconds = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or_default();
            json!({
                "timezone": timezone,
                "current_unix_seconds": unix_seconds,
                "note": "Demo tool returns server time as a Unix timestamp."
            })
        }
        name => json!({ "error": format!("unknown tool: {name}") }),
    }
}

fn fake_weather(location: &str) -> Value {
    let normalized = location.to_lowercase();
    if normalized.contains("paris") {
        json!({ "location": location, "forecast": "mild and cloudy", "temperature_c": 18 })
    } else if normalized.contains("chicago") {
        json!({ "location": location, "forecast": "breezy with lake clouds", "temperature_c": 11 })
    } else {
        json!({ "location": location, "forecast": "clear demo skies", "temperature_c": 21 })
    }
}

fn sse_json(value: Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

fn message_id() -> String {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("msg_{millis}")
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

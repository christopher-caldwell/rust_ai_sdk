use std::{convert::Infallible, net::SocketAddr};

use another_ai_sdk::{
    core::{
        error::SdkError, message::Message, request::TextRequest, stream::StreamEvent,
        tool::ToolDefinition, types::FinishReason,
    },
    providers::openai::model::OpenAiChatModel,
    runtime::{
        stream::stream_text,
        tools::ToolRegistry,
        turn::{ContinuationBuilder, TurnAccumulator},
    },
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
    model: OpenAiChatModel,
    tools: ToolRegistry,
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
        model: OpenAiChatModel::new(openai_api_key, model),
        tools: demo_tool_registry(),
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

        let mut request = TextRequest::builder()
            .messages(ui_messages_to_sdk_messages(input.messages))
            .max_output_tokens(800)
            .temperature(0.7)
            .tools(state.tools.definitions())
            .build();

        for step_index in 0..5 {
            yield Ok(sse_json(json!({ "type": "start-step" })));

            let sdk_stream = match stream_text(&state.model, request.clone()).await {
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

            let mut turn = TurnAccumulator::default();
            let text_part_id = format!("text_{step_index}");
            let mut text_started = false;
            futures_util::pin_mut!(sdk_stream);

            while let Some(event) = sdk_stream.next().await {
                match event {
                    Ok(event) => {
                        turn.push_event(event.clone());

                        match event {
                            StreamEvent::TextDelta(delta) => {
                                if !text_started {
                                    text_started = true;
                                    yield Ok(sse_json(json!({
                                        "type": "text-start",
                                        "id": text_part_id,
                                    })));
                                }

                                yield Ok(sse_json(json!({
                                    "type": "text-delta",
                                    "id": text_part_id,
                                    "delta": delta,
                                })));
                            }
                            StreamEvent::ToolCallStarted { id, name, .. } => {
                                yield Ok(sse_json(json!({
                                    "type": "tool-input-start",
                                    "toolCallId": id,
                                    "toolName": name,
                                })));
                            }
                            StreamEvent::ToolCallDelta { id, index, input_delta } => {
                                let tool_call_id = if id.is_empty() {
                                    format!("tool_call_{index}")
                                } else {
                                    id
                                };
                                yield Ok(sse_json(json!({
                                    "type": "tool-input-delta",
                                    "toolCallId": tool_call_id,
                                    "inputTextDelta": input_delta,
                                })));
                            }
                            StreamEvent::ToolCallReady { id, name, input, .. } => {
                                yield Ok(sse_json(json!({
                                    "type": "tool-input-available",
                                    "toolCallId": id,
                                    "toolName": name,
                                    "input": input,
                                })));
                            }
                            StreamEvent::Finished { .. } => {}
                        }
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

            if text_started {
                yield Ok(sse_json(json!({ "type": "text-end", "id": text_part_id })));
            }

            let accumulated_turn = turn.into_accumulated();
            let finish_reason = accumulated_turn.finish_reason.clone();
            let tool_calls = accumulated_turn.tool_calls_cloned();
            let assistant_parts = accumulated_turn.parts;

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
                let output = match state.tools.execute(call).await {
                    Ok(output) => output,
                    Err(error) => json!({ "error": error.to_string() }),
                };
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

fn demo_tool_registry() -> ToolRegistry {
    ToolRegistry::new()
        .register(
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
            |call| async move {
                let location = call
                    .input
                    .get("location")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                Ok::<Value, SdkError>(fake_weather(location))
            },
        )
        .register(
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
            |call| async move {
                let timezone = call
                    .input
                    .get("timezone")
                    .and_then(Value::as_str)
                    .unwrap_or("America/Chicago");
                let unix_seconds = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|duration| duration.as_secs())
                    .unwrap_or_default();
                Ok::<Value, SdkError>(json!({
                    "timezone": timezone,
                    "current_unix_seconds": unix_seconds,
                    "note": "Demo tool returns server time as a Unix timestamp."
                }))
            },
        )
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

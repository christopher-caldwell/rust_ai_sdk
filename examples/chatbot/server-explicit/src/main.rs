use std::net::SocketAddr;

use another_ai_sdk::{
    core::{error::SdkError, request::TextRequest, tool::ToolDefinition},
    providers::openai::model::OpenAiChatModel,
    runtime::{
        message_stream::{
            MESSAGE_STREAM_CACHE_CONTROL, MESSAGE_STREAM_CONTENT_TYPE,
            MESSAGE_STREAM_PROTOCOL_HEADER, MESSAGE_STREAM_PROTOCOL_VERSION, MessageStreamChunk,
            MessageStreamOptions, MessageStreamRequest, messages_to_sdk_messages,
            stream_text_messages,
        },
        tools::ToolRegistry,
    },
};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing_subscriber::fmt;

const SYSTEM_PROMPT: &str = "You are a concise demo chatbot running behind an Axum server. Use tools when they are relevant, then explain the result naturally.";

#[derive(Clone)]
struct AppState {
    model: OpenAiChatModel,
    tools: ToolRegistry,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();
    fmt::init();

    let state = AppState {
        model: OpenAiChatModel::new(openai_api_key(), openai_model()),
        tools: demo_tool_registry(),
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/chat", post(chat_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port()));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("explicit chatbot server listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(input): Json<MessageStreamRequest>,
) -> impl IntoResponse {
    let options = MessageStreamOptions::default();
    let request = build_text_request(input, &state.tools, options);
    let stream = stream_text_messages(state.model, request, state.tools, options);

    sse_response(stream)
}

fn build_text_request(
    input: MessageStreamRequest,
    tools: &ToolRegistry,
    options: MessageStreamOptions,
) -> TextRequest {
    let messages = messages_to_sdk_messages(input, SYSTEM_PROMPT);
    let tool_definitions = tools.definitions();

    TextRequest::builder()
        .messages(messages)
        .max_output_tokens(options.max_output_tokens)
        .temperature(options.temperature)
        .tools(tool_definitions)
        .build()
}

fn sse_response<S>(stream: S) -> Response
where
    S: futures_core::Stream<Item = MessageStreamChunk> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, MESSAGE_STREAM_CONTENT_TYPE)
        .header(header::CACHE_CONTROL, MESSAGE_STREAM_CACHE_CONTROL)
        .header(
            MESSAGE_STREAM_PROTOCOL_HEADER,
            MESSAGE_STREAM_PROTOCOL_VERSION,
        )
        .body(Body::from_stream(stream))
        .unwrap()
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

fn load_env() {
    dotenvy::dotenv().ok();
    dotenvy::from_path("../server/.env").ok();
}

fn openai_api_key() -> String {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env")
}

fn openai_model() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string())
}

fn port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3001)
}

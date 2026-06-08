use std::net::SocketAddr;

use another_ai_sdk::{
    core::{error::SdkError, tool::ToolDefinition},
    providers::openai::{OpenAiChatModel, OpenAiModel},
    runtime::{
        message_stream::{
            MESSAGE_STREAM_CACHE_CONTROL, MESSAGE_STREAM_CONTENT_TYPE,
            MESSAGE_STREAM_PROTOCOL_HEADER, MESSAGE_STREAM_PROTOCOL_VERSION, MessageStreamOptions,
            MessageStreamRequest, compose_text_request, stream_text_messages,
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
    dotenvy::dotenv().ok();
    fmt::init();

    let openai_api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in server/.env");
    let model =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| OpenAiModel::Gpt5_4Nano.to_string());
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
    Json(input): Json<MessageStreamRequest>,
) -> Response {
    let options = MessageStreamOptions::default();
    let request =
        match compose_text_request(input, SYSTEM_PROMPT, options, state.tools.definitions()) {
            Ok(request) => request,
            Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        };
    let stream = stream_text_messages(state.model, request, state.tools, options);

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

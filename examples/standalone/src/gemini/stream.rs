// Requires GEMINI_API_KEY in the environment.

use another_ai_sdk::{
    core::{request::TextRequest, stream::StreamEvent},
    providers::gemini::{GeminiChatModel, GeminiModel},
    runtime::stream::stream_text,
};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model = GeminiChatModel::new(api_key, GeminiModel::Gemini2_5Flash);

    let request = TextRequest::prompt("Write a short haiku about the Rust programming language");

    let mut stream = stream_text(&model, request)
        .await
        .expect("Failed to start stream");

    while let Some(event) = stream.next().await {
        match event.expect("Stream error") {
            StreamEvent::TextDelta(text) => {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            }
            StreamEvent::Finished {
                finish_reason,
                usage,
                ..
            } => {
                println!("\n\n[Finished: {:?}]", finish_reason);
                if let Some(u) = usage {
                    println!(
                        "[Usage - Input: {}, Output: {}, Total: {}]",
                        u.input_tokens.unwrap_or(0),
                        u.output_tokens.unwrap_or(0),
                        u.total_tokens.unwrap_or(0)
                    );
                }
            }
            _ => {}
        }
    }
}

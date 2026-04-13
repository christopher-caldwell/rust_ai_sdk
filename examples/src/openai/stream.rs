// Requires OPENAI_API_KEY in the environment.

use ai_sdk::{
    core::{request::TextRequest, stream::StreamEvent},
    providers::openai::{model::OpenAiChatModel, OpenAiModel},
    runtime::stream::stream_text,
};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model = OpenAiChatModel::new(api_key, OpenAiModel::Gpt4_1Mini);

    let request = TextRequest::prompt("Write a short haiku about the Rust programming language");

    let mut stream = stream_text(&model, request).await.expect("Failed to start stream");

    while let Some(event) = stream.next().await {
        match event.expect("Stream error") {
            StreamEvent::TextDelta(text) => {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            }
            StreamEvent::Finished { finish_reason, usage, .. } => {
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
        }
    }
}

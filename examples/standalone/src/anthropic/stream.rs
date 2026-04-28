// Requires ANTHROPIC_API_KEY in the environment.

use ai_sdk::{
    core::{request::TextRequest, stream::StreamEvent},
    providers::anthropic::{AnthropicChatModel, AnthropicModel},
    runtime::stream::stream_text,
};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let model = AnthropicChatModel::new(api_key, AnthropicModel::Haiku4_5);


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
            _ => {}
        }
    }
}

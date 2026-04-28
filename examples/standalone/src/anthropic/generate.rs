// Requires ANTHROPIC_API_KEY in the environment.

use another_ai_sdk::{
    core::request::TextRequest,
    providers::anthropic::{AnthropicChatModel, AnthropicModel},
    runtime::generate::generate_text,
};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let model = AnthropicChatModel::new(api_key, AnthropicModel::Haiku4_5);

    let request = TextRequest::prompt("Write a haiku about Rust");

    let result = generate_text(&model, request).await.unwrap();

    println!("{}", result.text);

    if let Some(u) = result.usage {
        println!(
            "\n[Usage - Input: {}, Output: {}, Total: {}]",
            u.input_tokens.unwrap_or(0),
            u.output_tokens.unwrap_or(0),
            u.total_tokens.unwrap_or(0)
        );
    }
}

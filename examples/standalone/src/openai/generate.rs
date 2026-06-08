// Requires OPENAI_API_KEY in the environment.

use another_ai_sdk::{
    core::request::TextRequest,
    providers::openai::{OpenAiChatModel, OpenAiModel},
    runtime::generate::generate_text,
};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model_id =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| OpenAiModel::Gpt5_4Nano.to_string());
    let model = OpenAiChatModel::new(api_key, model_id);

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

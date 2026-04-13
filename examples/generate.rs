// Requires OPENAI_API_KEY in the environment.

use ai_sdk::{
    core::request::TextRequest,
    providers::openai::model::OpenAiChatModel,
    runtime::generate::generate_text,
};

#[tokio::main]
async fn main() {
    let model = OpenAiChatModel::new(
        std::env::var("OPENAI_API_KEY").unwrap(),
        "gpt-4.1",
    );

    let request = TextRequest::prompt("Write a haiku about Rust");

    let result = generate_text(&model, request).await.unwrap();

    println!("{}", result.text);
}

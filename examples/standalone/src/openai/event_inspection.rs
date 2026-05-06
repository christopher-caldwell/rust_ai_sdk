// Requires OPENAI_API_KEY in the environment.
//
// Prints raw provider-neutral stream events so the event API is easy to inspect.

use another_ai_sdk::{
    core::{
        request::TextRequest,
        stream::StreamEvent,
        tool::{ToolChoice, ToolDefinition},
    },
    providers::openai::{model::OpenAiChatModel, OpenAiModel},
    runtime::stream::stream_text,
};
use futures_util::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model = OpenAiChatModel::new(api_key, OpenAiModel::Gpt4_1Mini);

    let mut request = TextRequest::prompt(
        "Call get_weather for Paris. Do not answer from memory.",
    )
    .with_tools(vec![weather_tool()])
    .with_tool_choice(ToolChoice::Required {
        name: "get_weather".to_string(),
    });
    request.max_output_tokens = Some(300);

    let mut stream = stream_text(&model, request).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(delta) => println!("TextDelta({delta:?})"),
            StreamEvent::ToolCallStarted { id, name, index } => {
                println!("ToolCallStarted(index={index}, id={id:?}, name={name:?})");
            }
            StreamEvent::ToolCallDelta {
                id,
                index,
                input_delta,
            } => {
                println!(
                    "ToolCallDelta(index={index}, id={id:?}, input_delta={input_delta:?})"
                );
            }
            StreamEvent::ToolCallReady {
                id,
                name,
                index,
                input,
                ..
            } => {
                println!(
                    "ToolCallReady(index={index}, id={id:?}, name={name:?}, input={input})"
                );
            }
            StreamEvent::Finished {
                finish_reason,
                usage,
                response,
            } => {
                println!(
                    "Finished(reason={finish_reason:?}, usage={usage:?}, response={response:?})"
                );
            }
        }
    }

    Ok(())
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        "Get a deterministic weather report for a city.",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, for example Paris"
                }
            },
            "required": ["location"],
            "additionalProperties": false
        }),
    )
}

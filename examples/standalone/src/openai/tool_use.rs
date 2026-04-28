// Requires OPENAI_API_KEY in the environment.
//
// Demonstrates a tool-aware turn using the structured non-streaming result.

use another_ai_sdk::{
    core::{
        message::ToolCall,
        request::TextRequest,
        tool::{ToolChoice, ToolDefinition},
    },
    providers::openai::{model::OpenAiChatModel, OpenAiModel},
    runtime::turn::ContinuationBuilder,
};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model = OpenAiChatModel::new(api_key, OpenAiModel::Gpt4_1Mini);

    let mut request = TextRequest::prompt(
        "What is the weather in Paris? Use the get_weather tool before answering.",
    )
    .with_tools(vec![weather_tool()])
    .with_tool_choice(ToolChoice::Required {
        name: "get_weather".to_string(),
    });
    request.max_output_tokens = Some(500);

    loop {
        let result = model.generate_chat(request.clone()).await?;

        if !result.has_tool_calls() {
            println!("{}", result.text());
            break;
        }

        let mut builder =
            ContinuationBuilder::from_request(request).with_assistant_turn(result.parts.clone());

        for call in result.tool_calls() {
            let output = execute_tool(call);
            println!("[tool:{}] {}", call.name, output);
            builder = builder.with_tool_result(&call.id, output);
        }

        request = builder.build();
        request.tool_choice = Some(ToolChoice::None);
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

fn execute_tool(call: &ToolCall) -> String {
    if call.name != "get_weather" {
        return json!({ "error": format!("unknown tool: {}", call.name) }).to_string();
    }

    let location = call
        .input
        .get("location")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    fake_weather(location).to_string()
}

fn fake_weather(location: &str) -> Value {
    let normalized = location.to_lowercase();
    if normalized.contains("paris") {
        json!({
            "location": location,
            "forecast": "mild and cloudy",
            "temperature_c": 18
        })
    } else if normalized.contains("austin") {
        json!({
            "location": location,
            "forecast": "clear and warm",
            "temperature_c": 29
        })
    } else {
        json!({
            "location": location,
            "forecast": "not available in the demo data",
            "temperature_c": null
        })
    }
}

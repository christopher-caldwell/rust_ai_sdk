// Requires OPENAI_API_KEY in the environment.
//
// Demonstrates the high-level streaming tool loop. The runtime streams and
// accumulates each turn internally, then returns a provider-neutral outcome.
// Use openai-event-inspection when you need to inspect raw StreamEvent values.

use another_ai_sdk::{
    prelude::*,
    providers::openai::{OpenAiChatModel, OpenAiModel},
};
use serde_json::{Value, json};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model_id =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| OpenAiModel::Gpt5_4Nano.to_string());
    let model = OpenAiChatModel::new(api_key, model_id);

    let tools = weather_tools();
    let mut request = TextRequest::builder()
        .prompt("What is the weather in Paris? Use the get_weather tool before answering.")
        .max_output_tokens(500)
        .tools(tools.definitions())
        .tool_choice(ToolChoice::required("get_weather"))
        .build();

    loop {
        let base_request = request.clone();
        let outcome = run_turn(&model, request).await?;

        match outcome {
            TurnOutcome::Completed(result) => {
                println!("{}", result.text());
                print_finish(&result.finish_reason, &result.usage, &result.response);
                break;
            }
            TurnOutcome::ToolsRequired {
                assistant_parts,
                tool_calls,
                finish_reason,
                usage,
                response,
            } => {
                print_finish(&finish_reason, &usage, &response);

                let mut continuation = ContinuationBuilder::from_request(base_request)
                    .with_assistant_turn(assistant_parts);

                for call in &tool_calls {
                    let output = tools.execute(call).await?;
                    println!("[tool:{}] {}", call.name, output);
                    continuation = continuation.with_tool_result(&call.id, output);
                }

                request = continuation.build().with_tool_choice(ToolChoice::None);
                println!();
            }
        }
    }

    Ok(())
}

fn weather_tools() -> ToolRegistry {
    ToolRegistry::new().register(weather_tool_definition(), |call| async move {
        let location = call
            .input
            .get("location")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();

        Ok::<Value, SdkError>(fake_weather(&location))
    })
}

fn weather_tool_definition() -> ToolDefinition {
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

fn fake_weather(location: &str) -> Value {
    let normalized = location.to_lowercase();
    if normalized.contains("paris") {
        json!({
            "location": location,
            "forecast": "mild and cloudy",
            "temperature_c": 18
        })
    } else {
        json!({
            "location": location,
            "forecast": "not available in the demo data",
            "temperature_c": null
        })
    }
}

fn print_finish(reason: &FinishReason, usage: &Option<Usage>, response: &ResponseMetadata) {
    println!(
        "[finished: {:?} id={:?} model={:?}]",
        reason, response.id, response.model
    );

    if let Some(usage) = usage {
        println!(
            "[usage input={} output={} total={}]",
            usage.input_tokens.unwrap_or(0),
            usage.output_tokens.unwrap_or(0),
            usage.total_tokens.unwrap_or(0)
        );
    }
}

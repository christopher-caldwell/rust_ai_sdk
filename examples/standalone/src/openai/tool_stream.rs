// Requires OPENAI_API_KEY in the environment.
//
// Mirrors the target server shape: stream one model turn, execute requested
// tools in application code, append tool results, then stream the final answer.

use ai_sdk::{
    core::{
        message::{Message, MessagePart, ToolCall},
        request::TextRequest,
        stream::StreamEvent,
        tool::{ToolChoice, ToolDefinition},
        types::{FinishReason, ResponseMetadata, Usage},
    },
    providers::openai::{model::OpenAiChatModel, OpenAiModel},
    runtime::{stream::stream_text, turn::ContinuationBuilder},
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write};

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
        let turn = stream_one_turn(&model, request.clone()).await?;

        if turn.tool_calls.is_empty() {
            println!();
            print_finish(&turn.finish_reason, &turn.usage, &turn.response);
            break;
        }

        println!();
        let mut builder =
            ContinuationBuilder::from_request(request).with_assistant_turn(turn.assistant_parts);

        for call in &turn.tool_calls {
            let output = execute_tool(call);
            println!("[tool:{}] {}", call.name, output);
            builder = builder.with_tool_result(&call.id, output);
        }

        request = builder.build();
        request.tool_choice = Some(ToolChoice::None);
        println!();
    }

    Ok(())
}

struct StreamedTurn {
    assistant_parts: Vec<MessagePart>,
    tool_calls: Vec<ToolCall>,
    finish_reason: FinishReason,
    usage: Option<Usage>,
    response: ResponseMetadata,
}

struct ToolBuffer {
    id: String,
    name: String,
    input: String,
}

async fn stream_one_turn(
    model: &OpenAiChatModel,
    request: TextRequest,
) -> Result<StreamedTurn, Box<dyn std::error::Error>> {
    let mut stream = stream_text(model, request).await?;
    let mut text = String::new();
    let mut tool_buffers: BTreeMap<u32, ToolBuffer> = BTreeMap::new();
    let mut finish_reason = FinishReason::Other("unknown".to_string());
    let mut usage = None;
    let mut response = ResponseMetadata {
        id: None,
        model: None,
    };

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(delta) => {
                print!("{}", delta);
                std::io::stdout().flush()?;
                text.push_str(&delta);
            }
            StreamEvent::ToolCallStarted { id, name, index } => {
                println!("[tool-call-start:{}:{}]", index, name);
                tool_buffers.insert(
                    index,
                    ToolBuffer {
                        id,
                        name,
                        input: String::new(),
                    },
                );
            }
            StreamEvent::ToolCallDelta {
                index,
                input_delta,
                ..
            } => {
                if let Some(buffer) = tool_buffers.get_mut(&index) {
                    buffer.input.push_str(&input_delta);
                }
            }
            StreamEvent::ToolCallReady {
                id,
                name,
                index,
                input,
            } => {
                println!("[tool-call-ready:{}:{} {}]", index, name, input);
                tool_buffers.insert(
                    index,
                    ToolBuffer {
                        id,
                        name,
                        input: input.to_string(),
                    },
                );
            }
            StreamEvent::Finished {
                finish_reason: reason,
                usage: final_usage,
                response: metadata,
            } => {
                finish_reason = reason;
                usage = final_usage;
                response = metadata;
            }
        }
    }

    let mut assistant_parts = Vec::new();
    if !text.is_empty() {
        assistant_parts.push(MessagePart::Text(text));
    }

    let mut tool_calls = Vec::new();
    for (_, buffer) in tool_buffers {
        let input = serde_json::from_str(&buffer.input)
            .unwrap_or_else(|_| Value::String(buffer.input.clone()));
        let call = ToolCall {
            id: buffer.id,
            name: buffer.name,
            input,
        };
        assistant_parts.push(MessagePart::ToolCall(call.clone()));
        tool_calls.push(call);
    }

    Ok(StreamedTurn {
        assistant_parts,
        tool_calls,
        finish_reason,
        usage,
        response,
    })
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
    } else {
        json!({
            "location": location,
            "forecast": "not available in the demo data",
            "temperature_c": null
        })
    }
}

fn print_finish(reason: &FinishReason, usage: &Option<Usage>, response: &ResponseMetadata) {
    println!("[finished: {:?} id={:?} model={:?}]", reason, response.id, response.model);
    if let Some(usage) = usage {
        println!(
            "[usage input={} output={} total={}]",
            usage.input_tokens.unwrap_or(0),
            usage.output_tokens.unwrap_or(0),
            usage.total_tokens.unwrap_or(0)
        );
    }
}

#[allow(dead_code)]
fn _assistant_message(parts: Vec<MessagePart>) -> Message {
    Message::assistant_parts(parts)
}

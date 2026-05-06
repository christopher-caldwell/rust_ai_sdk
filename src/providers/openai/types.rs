use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{
    error::SdkError,
    message::{Message, MessagePart, Role, ToolCall},
    request::TextRequest,
    result::{ChatResult, TextResult},
    tool::ToolChoice,
    types::{FinishReason, ResponseMetadata, Usage as TokenUsage},
};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OaiToolChoice>,
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCallOut>>,
}

#[derive(Debug, Serialize)]
pub(super) struct OaiToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: OaiFunctionCallOut,
}

#[derive(Debug, Serialize)]
pub(super) struct OaiFunctionCallOut {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub(super) struct OaiTool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: OaiFunctionDef,
}

#[derive(Debug, Serialize)]
pub(super) struct OaiFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(super) enum OaiToolChoice {
    String(&'static str),
    Function(OaiToolChoiceFunction),
}

#[derive(Debug, Serialize)]
pub(super) struct OaiToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: &'static str,
    pub function: OaiToolChoiceName,
}

#[derive(Debug, Serialize)]
pub(super) struct OaiToolChoiceName {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response types (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Choice {
    pub message: AssistantMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AssistantMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OaiToolCallIn>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OaiToolCallIn {
    pub id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub call_type: String,
    pub function: OaiFunctionCallIn,
}

#[derive(Debug, Deserialize)]
pub(super) struct OaiFunctionCallIn {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct UsageBody {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAiErrorBody {
    pub error: OpenAiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAiErrorDetail {
    pub message: String,
}

// ---------------------------------------------------------------------------
// Streaming chunk types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionChunk {
    pub id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChunkChoice {
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct ChunkDelta {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OaiChunkToolCallDelta>>,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct OaiChunkToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub function: Option<OaiFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct OaiFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Translation functions
// ---------------------------------------------------------------------------

pub(super) fn text_request_to_openai(
    model: &str,
    request: &TextRequest,
    stream_mode: bool,
) -> ChatCompletionRequest {
    let tools: Vec<OaiTool> = request
        .tools
        .iter()
        .map(|t| OaiTool {
            tool_type: "function",
            function: OaiFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    let tool_choice = request.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Auto => OaiToolChoice::String("auto"),
        ToolChoice::None => OaiToolChoice::String("none"),
        ToolChoice::Required { name } => OaiToolChoice::Function(OaiToolChoiceFunction {
            choice_type: "function",
            function: OaiToolChoiceName { name: name.clone() },
        }),
    });

    ChatCompletionRequest {
        model: model.to_string(),
        messages: request
            .messages
            .iter()
            .flat_map(message_to_chat_messages)
            .collect(),
        max_tokens: request.max_output_tokens,
        temperature: request.temperature,
        stream: if stream_mode { Some(true) } else { None },
        stream_options: if stream_mode {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        tools,
        tool_choice,
    }
}

fn message_to_chat_messages(msg: &Message) -> Vec<ChatMessage> {
    let parts = msg.effective_parts();

    // Check if this is a tool-result message (all ToolResult parts, role=User).
    let tool_results: Vec<_> = parts
        .iter()
        .filter_map(|p| {
            if let MessagePart::ToolResult(tr) = p {
                Some(tr)
            } else {
                None
            }
        })
        .collect();

    if !tool_results.is_empty() {
        return tool_results
            .into_iter()
            .map(|tr| ChatMessage {
                role: "tool".to_string(),
                content: Some(tr.content.clone()),
                tool_call_id: Some(tr.tool_call_id.clone()),
                tool_calls: None,
            })
            .collect();
    }

    // Check if this is an assistant message with tool calls.
    let tool_calls_out: Vec<_> = parts
        .iter()
        .filter_map(|p| {
            if let MessagePart::ToolCall(tc) = p {
                Some(OaiToolCallOut {
                    id: tc.id.clone(),
                    call_type: "function",
                    function: OaiFunctionCallOut {
                        name: tc.name.clone(),
                        arguments: tc.input.to_string(),
                    },
                })
            } else {
                None
            }
        })
        .collect();

    let text_content: Option<String> = {
        let texts: Vec<_> = parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    };

    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    if !tool_calls_out.is_empty() {
        vec![ChatMessage {
            role: role.to_string(),
            content: text_content,
            tool_call_id: None,
            tool_calls: Some(tool_calls_out),
        }]
    } else {
        vec![ChatMessage {
            role: role.to_string(),
            content: text_content.or_else(|| {
                if msg.is_text_only() {
                    Some(msg.content.clone())
                } else {
                    None
                }
            }),
            tool_call_id: None,
            tool_calls: None,
        }]
    }
}

pub(super) fn map_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" => FinishReason::ToolUse,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(super) fn chat_response_to_text_result(
    resp: ChatCompletionResponse,
) -> Result<TextResult, SdkError> {
    let Some(choice) = resp.choices.first() else {
        return Err(SdkError::Api(
            "OpenAI response contained no choices".to_string(),
        ));
    };

    let text = choice.message.content.clone().unwrap_or_default();
    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(map_finish_reason)
        .unwrap_or_else(|| FinishReason::Other("unknown".to_string()));

    let usage = resp.usage.map(|u| TokenUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(TextResult {
        text,
        finish_reason,
        usage,
        response: ResponseMetadata {
            id: resp.id,
            model: resp.model,
        },
    })
}

pub(super) fn chat_response_to_chat_result(
    resp: ChatCompletionResponse,
) -> Result<ChatResult, SdkError> {
    let Some(choice) = resp.choices.first() else {
        return Err(SdkError::Api(
            "OpenAI response contained no choices".to_string(),
        ));
    };

    let mut parts: Vec<MessagePart> = Vec::new();

    if let Some(text) = &choice.message.content
        && !text.is_empty()
    {
        parts.push(MessagePart::Text(text.clone()));
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            let input: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::String(tc.function.arguments.clone()));
            parts.push(MessagePart::ToolCall(ToolCall::new(
                tc.id.clone(),
                tc.function.name.clone(),
                input,
            )));
        }
    }

    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(map_finish_reason)
        .unwrap_or_else(|| FinishReason::Other("unknown".to_string()));

    let usage = resp.usage.map(|u| TokenUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(ChatResult {
        parts,
        finish_reason,
        usage,
        response: ResponseMetadata {
            id: resp.id,
            model: resp.model,
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tool::ToolDefinition;
    use serde_json::json;

    #[test]
    fn test_map_finish_reason() {
        assert!(matches!(map_finish_reason("stop"), FinishReason::Stop));
        assert!(matches!(map_finish_reason("length"), FinishReason::Length));
        assert!(matches!(
            map_finish_reason("content_filter"),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            map_finish_reason("tool_calls"),
            FinishReason::ToolUse
        ));

        let other = map_finish_reason("max_tokens");
        match other {
            FinishReason::Other(s) => assert_eq!(s, "max_tokens"),
            _ => panic!("Expected Other"),
        }
    }

    #[test]
    fn test_chat_response_to_text_result_success() {
        let resp = ChatCompletionResponse {
            id: Some("req_123".to_string()),
            model: Some("gpt-4".to_string()),
            choices: vec![Choice {
                message: AssistantMessage {
                    content: Some("Hello world".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(UsageBody {
                prompt_tokens: Some(10),
                completion_tokens: Some(2),
                total_tokens: Some(12),
            }),
        };

        let res = chat_response_to_text_result(resp).expect("Should succeed");
        assert_eq!(res.text, "Hello world");
        assert!(matches!(res.finish_reason, FinishReason::Stop));
        assert_eq!(res.response.id.as_deref(), Some("req_123"));

        let usg = res.usage.expect("Usage expected");
        assert_eq!(usg.input_tokens, Some(10));
        assert_eq!(usg.output_tokens, Some(2));
        assert_eq!(usg.total_tokens, Some(12));
    }

    #[test]
    fn test_chat_response_to_text_result_missing_finish_reason() {
        let resp = ChatCompletionResponse {
            id: None,
            model: None,
            choices: vec![Choice {
                message: AssistantMessage {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let res = chat_response_to_text_result(resp).unwrap();
        match res.finish_reason {
            FinishReason::Other(s) => assert_eq!(s, "unknown"),
            _ => panic!("Expected Other(unknown)"),
        }
    }

    #[test]
    fn test_chat_response_to_text_result_empty_choices() {
        let resp = ChatCompletionResponse {
            id: None,
            model: None,
            choices: vec![],
            usage: None,
        };

        let err = chat_response_to_text_result(resp).unwrap_err();
        match err {
            SdkError::Api(msg) => assert!(msg.contains("no choices")),
            _ => panic!("Expected API error"),
        }
    }

    #[test]
    fn test_tool_def_serialization() {
        let req = TextRequest {
            messages: vec![Message::user("hello")],
            max_output_tokens: None,
            temperature: None,
            tools: vec![ToolDefinition::new(
                "get_weather",
                "Get weather for a location",
                json!({ "type": "object", "properties": { "location": { "type": "string" } }, "required": ["location"] }),
            )],
            tool_choice: None,
        };

        let body = text_request_to_openai("gpt-4", &req, false);
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].tool_type, "function");
        assert_eq!(body.tools[0].function.name, "get_weather");
        assert_eq!(
            body.tools[0].function.description,
            "Get weather for a location"
        );

        let json = serde_json::to_value(&body.tools[0]).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
    }

    #[test]
    fn test_tool_choice_auto_serialization() {
        let req = TextRequest {
            messages: vec![Message::user("hi")],
            max_output_tokens: None,
            temperature: None,
            tools: vec![ToolDefinition::new("t", "desc", json!({}))],
            tool_choice: Some(crate::core::tool::ToolChoice::Auto),
        };
        let body = text_request_to_openai("gpt-4", &req, false);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tool_choice"], "auto");
    }

    #[test]
    fn test_tool_choice_none_serialization() {
        let req = TextRequest {
            messages: vec![Message::user("hi")],
            max_output_tokens: None,
            temperature: None,
            tools: vec![ToolDefinition::new("t", "desc", json!({}))],
            tool_choice: Some(crate::core::tool::ToolChoice::None),
        };
        let body = text_request_to_openai("gpt-4", &req, false);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tool_choice"], "none");
    }

    #[test]
    fn test_tool_choice_required_serialization() {
        let req = TextRequest {
            messages: vec![Message::user("hi")],
            max_output_tokens: None,
            temperature: None,
            tools: vec![ToolDefinition::new("my_tool", "desc", json!({}))],
            tool_choice: Some(crate::core::tool::ToolChoice::Required {
                name: "my_tool".to_string(),
            }),
        };
        let body = text_request_to_openai("gpt-4", &req, false);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["function"]["name"], "my_tool");
    }

    #[test]
    fn test_message_tool_call_part_to_openai() {
        let msg = Message::assistant_parts(vec![MessagePart::ToolCall(
            crate::core::message::ToolCall::new(
                "call_123",
                "weather",
                json!({"location": "Paris"}),
            ),
        )]);

        let msgs = message_to_chat_messages(&msg);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        let tool_calls = msgs[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "weather");
    }

    #[test]
    fn test_message_tool_result_part_to_openai() {
        let msg = Message::tool_result("call_123", "sunny and warm");

        let msgs = message_to_chat_messages(&msg);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "tool");
        assert_eq!(msgs[0].content.as_deref(), Some("sunny and warm"));
        assert_eq!(msgs[0].tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_chat_response_to_chat_result_with_tool_calls() {
        let resp = ChatCompletionResponse {
            id: Some("req_1".to_string()),
            model: Some("gpt-4".to_string()),
            choices: vec![Choice {
                message: AssistantMessage {
                    content: None,
                    tool_calls: Some(vec![OaiToolCallIn {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: OaiFunctionCallIn {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"Paris"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let result = chat_response_to_chat_result(resp).unwrap();
        assert!(result.has_tool_calls());
        let calls = result.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "Paris");
        assert!(matches!(result.finish_reason, FinishReason::ToolUse));
    }
}

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

#[derive(Debug, Serialize)]
pub(super) struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Serialize)]
pub(super) struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Serialize)]
pub(super) struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(super) enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(super) enum AnthropicMessageContent {
    Text(String),
    Parts(Vec<AnthropicContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(super) enum AnthropicContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub(super) struct AnthropicUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicErrorResponse {
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub error_type: String,
}

// ---------------------------------------------------------------------------
// Stream event types — envelope-first, forward-compatible
// ---------------------------------------------------------------------------

/// Minimal envelope used to read the `type` field before full deserialization.
#[derive(Debug, Deserialize)]
pub(super) struct EventEnvelope {
    #[serde(rename = "type")]
    pub event_type: String,
}

/// `message_start` — carries message metadata and initial usage.
#[derive(Debug, Deserialize)]
pub(super) struct MessageStartEvent {
    pub message: MessageStartBody,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageStartBody {
    pub id: Option<String>,
    pub model: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

/// `content_block_delta` — carries a text (or other) delta.
#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentDelta,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentDelta {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub partial_json: Option<String>,
}

/// `content_block_start` - carries text or tool-use block metadata.
#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockStartEvent {
    pub index: u32,
    pub content_block: ContentBlockStart,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(super) enum ContentBlockStart {
    #[serde(rename = "text")]
    Text { text: Option<String> },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

/// `content_block_stop` - marks the end of a text or tool-use block.
#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockStopEvent {
    pub index: u32,
}

/// `message_delta` — carries stop_reason and output usage.
#[derive(Debug, Deserialize)]
pub(super) struct MessageDeltaEvent {
    pub delta: MessageDeltaBody,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageDeltaBody {
    pub stop_reason: Option<String>,
}

/// `error` — carries an Anthropic error payload.
#[derive(Debug, Deserialize)]
pub(super) struct ErrorEvent {
    pub error: AnthropicErrorDetail,
}

pub(super) fn text_request_to_anthropic(
    model: &str,
    request: &TextRequest,
    stream_mode: bool,
) -> AnthropicRequest {
    let mut system_strings = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => system_strings.push(system_text(msg)),
            Role::User | Role::Tool => messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: message_content(msg),
            }),
            Role::Assistant => messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: message_content(msg),
            }),
        }
    }

    let system = if system_strings.is_empty() {
        None
    } else {
        Some(system_strings.join("\n"))
    };

    let mut tools: Vec<AnthropicTool> = request
        .tools
        .iter()
        .map(|tool| AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect();

    let tool_choice = request
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            ToolChoice::Auto => Some(AnthropicToolChoice::Auto),
            ToolChoice::Required { name } => Some(AnthropicToolChoice::Tool { name: name.clone() }),
            ToolChoice::None => {
                tools.clear();
                None
            }
        });

    AnthropicRequest {
        model: model.to_string(),
        messages,
        max_tokens: request.max_output_tokens.unwrap_or(4096),
        system,
        temperature: request.temperature,
        stream: if stream_mode { Some(true) } else { None },
        tools,
        tool_choice,
    }
}

fn system_text(msg: &Message) -> String {
    msg.effective_parts()
        .into_iter()
        .filter_map(|part| {
            if let MessagePart::Text(text) = part {
                Some(text)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn message_content(msg: &Message) -> AnthropicMessageContent {
    if msg.parts.is_empty() {
        return AnthropicMessageContent::Text(msg.content.clone());
    }

    let parts = msg
        .parts
        .iter()
        .map(|part| match part {
            MessagePart::Text(text) => AnthropicContentPart::Text { text: text.clone() },
            MessagePart::ToolCall(call) => AnthropicContentPart::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.input.clone(),
            },
            MessagePart::ToolResult(result) => AnthropicContentPart::ToolResult {
                tool_use_id: result.tool_call_id.clone(),
                content: result.content.clone(),
            },
        })
        .collect();

    AnthropicMessageContent::Parts(parts)
}

pub(super) fn map_stop_reason(stop_reason: &str) -> FinishReason {
    match stop_reason {
        "end_turn" => FinishReason::Stop,
        "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(super) fn anthropic_response_to_text_result(
    resp: AnthropicResponse,
) -> Result<TextResult, SdkError> {
    let mut text = String::new();
    for block in resp.content {
        if block.block_type == "text"
            && let Some(t) = block.text
        {
            text.push_str(&t);
        }
    }

    let finish_reason = resp
        .stop_reason
        .as_deref()
        .map(map_stop_reason)
        .unwrap_or_else(|| FinishReason::Other("unknown".to_string()));

    let usage = resp.usage.map(|u| {
        let input = u.input_tokens.unwrap_or(0);
        let output = u.output_tokens.unwrap_or(0);
        TokenUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: Some(input + output),
        }
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

pub(super) fn anthropic_response_to_chat_result(
    resp: AnthropicResponse,
) -> Result<ChatResult, SdkError> {
    let mut parts = Vec::new();
    for block in &resp.content {
        match block.block_type.as_str() {
            "text" => {
                if let Some(text) = &block.text
                    && !text.is_empty()
                {
                    parts.push(MessagePart::Text(text.clone()));
                }
            }
            "tool_use" => {
                if let (Some(id), Some(name)) = (&block.id, &block.name) {
                    parts.push(MessagePart::ToolCall(ToolCall::new(
                        id.clone(),
                        name.clone(),
                        block.input.clone().unwrap_or(Value::Null),
                    )));
                }
            }
            _ => {}
        }
    }

    let finish_reason = resp
        .stop_reason
        .as_deref()
        .map(map_stop_reason)
        .unwrap_or_else(|| FinishReason::Other("unknown".to_string()));

    let usage = resp.usage.map(|u| {
        let input = u.input_tokens.unwrap_or(0);
        let output = u.output_tokens.unwrap_or(0);
        TokenUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: Some(input + output),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_stop_reason() {
        assert!(matches!(map_stop_reason("end_turn"), FinishReason::Stop));
        assert!(matches!(
            map_stop_reason("max_tokens"),
            FinishReason::Length
        ));

        let other = map_stop_reason("unknown_reason");
        match other {
            FinishReason::Other(s) => assert_eq!(s, "unknown_reason"),
            _ => panic!("Expected Other"),
        }
    }

    #[test]
    fn test_anthropic_response_to_text_result_success() {
        let resp = AnthropicResponse {
            id: Some("msg_123".to_string()),
            model: Some("claude-3-5-sonnet".to_string()),
            content: vec![AnthropicContentBlock {
                block_type: "text".to_string(),
                text: Some("Hello world".to_string()),
                id: None,
                name: None,
                input: None,
            }],
            stop_reason: Some("end_turn".to_string()),
            usage: Some(AnthropicUsage {
                input_tokens: Some(10),
                output_tokens: Some(2),
            }),
        };

        let res = anthropic_response_to_text_result(resp).expect("Should succeed");
        assert_eq!(res.text, "Hello world");
        assert!(matches!(res.finish_reason, FinishReason::Stop));
        assert_eq!(res.response.id.as_deref(), Some("msg_123"));
        assert_eq!(res.response.model.as_deref(), Some("claude-3-5-sonnet"));

        let usg = res.usage.expect("Usage expected");
        assert_eq!(usg.input_tokens, Some(10));
        assert_eq!(usg.output_tokens, Some(2));
        assert_eq!(usg.total_tokens, Some(12));
    }

    #[test]
    fn test_tool_def_serialization() {
        let req =
            TextRequest::prompt("hello").with_tools(vec![crate::core::tool::ToolDefinition::new(
                "get_weather",
                "Get weather",
                serde_json::json!({"type": "object"}),
            )]);

        let body = text_request_to_anthropic("claude", &req, false);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"][0]["name"], "get_weather");
        assert_eq!(json["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn test_message_tool_result_to_anthropic() {
        let req = TextRequest {
            messages: vec![Message::tool_result("toolu_1", "sunny")],
            max_output_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
        };

        let body = text_request_to_anthropic("claude", &req, false);
        let json = serde_json::to_value(&body.messages[0]).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "tool_result");
        assert_eq!(json["content"][0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn test_message_tool_call_to_anthropic() {
        let req = TextRequest {
            messages: vec![Message::assistant_parts(vec![MessagePart::ToolCall(
                ToolCall::new(
                    "toolu_1",
                    "get_weather",
                    serde_json::json!({"location": "Paris"}),
                ),
            )])],
            max_output_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
        };

        let body = text_request_to_anthropic("claude", &req, false);
        let json = serde_json::to_value(&body.messages[0]).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "tool_use");
        assert_eq!(json["content"][0]["name"], "get_weather");
    }

    #[test]
    fn test_anthropic_response_to_chat_result_with_tool_use() {
        let resp = AnthropicResponse {
            id: Some("msg_1".to_string()),
            model: Some("claude".to_string()),
            content: vec![AnthropicContentBlock {
                block_type: "tool_use".to_string(),
                text: None,
                id: Some("toolu_1".to_string()),
                name: Some("get_weather".to_string()),
                input: Some(serde_json::json!({"location": "Paris"})),
            }],
            stop_reason: Some("tool_use".to_string()),
            usage: None,
        };

        let result = anthropic_response_to_chat_result(resp).unwrap();
        assert!(result.has_tool_calls());
        let calls = result.tool_calls();
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "Paris");
        assert!(matches!(result.finish_reason, FinishReason::ToolUse));
    }
}

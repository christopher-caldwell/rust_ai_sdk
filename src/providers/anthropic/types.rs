use serde::{Deserialize, Serialize};

use crate::core::{
    error::SdkError,
    message::Role,
    request::TextRequest,
    result::TextResult,
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
}

#[derive(Debug, Serialize)]
pub(super) struct AnthropicMessage {
    pub role: String,
    pub content: String,
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

// Stream chunk types
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(super) enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartBody },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { #[allow(dead_code)] index: u32, #[allow(dead_code)] content_block: AnthropicContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { #[allow(dead_code)] index: u32, delta: AnthropicContentDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { #[allow(dead_code)] index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaBody, usage: Option<AnthropicUsage> },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicErrorDetail },
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageStartBody {
    pub id: Option<String>,
    pub model: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicContentDelta {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub delta_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageDeltaBody {
    pub stop_reason: Option<String>,
}

pub(super) fn text_request_to_anthropic(model: &str, request: &TextRequest, stream_mode: bool) -> AnthropicRequest {
    let mut system_strings = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => system_strings.push(msg.content.clone()),
            Role::User => messages.push(AnthropicMessage { role: "user".to_string(), content: msg.content.clone() }),
            Role::Assistant => messages.push(AnthropicMessage { role: "assistant".to_string(), content: msg.content.clone() }),
        }
    }

    let system = if system_strings.is_empty() {
        None
    } else {
        Some(system_strings.join("\n"))
    };

    AnthropicRequest {
        model: model.to_string(),
        messages,
        max_tokens: request.max_output_tokens.unwrap_or(4096),
        system,
        temperature: request.temperature,
        stream: if stream_mode { Some(true) } else { None },
    }
}

pub(super) fn map_stop_reason(stop_reason: &str) -> FinishReason {
    match stop_reason {
        "end_turn" => FinishReason::Stop,
        "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::Other("tool_use".to_string()),
        other => FinishReason::Other(other.to_string()),
    }
}

pub(super) fn anthropic_response_to_text_result(
    resp: AnthropicResponse,
) -> Result<TextResult, SdkError> {
    let mut text = String::new();
    for block in resp.content {
        if block.block_type == "text" {
            if let Some(t) = block.text {
                text.push_str(&t);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_stop_reason() {
        assert!(matches!(map_stop_reason("end_turn"), FinishReason::Stop));
        assert!(matches!(map_stop_reason("max_tokens"), FinishReason::Length));
        
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
}

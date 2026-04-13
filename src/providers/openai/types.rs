use serde::{Deserialize, Serialize};

use crate::core::{
    error::SdkError,
    message::{Message, Role},
    request::TextRequest,
    result::TextResult,
    types::{FinishReason, ResponseMetadata, Usage as TokenUsage},
};

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatMessage {
    pub role: String,
    pub content: String,
}

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

pub(super) fn text_request_to_openai(model: &str, request: &TextRequest) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: request.messages.iter().map(message_to_chat).collect(),
        max_tokens: request.max_output_tokens,
        temperature: request.temperature,
    }
}

fn message_to_chat(msg: &Message) -> ChatMessage {
    ChatMessage {
        role: match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .to_string(),
        content: msg.content.clone(),
    }
}

pub(super) fn map_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_finish_reason() {
        assert!(matches!(map_finish_reason("stop"), FinishReason::Stop));
        assert!(matches!(map_finish_reason("length"), FinishReason::Length));
        assert!(matches!(map_finish_reason("content_filter"), FinishReason::ContentFilter));
        
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
        assert_eq!(res.response.model.as_deref(), Some("gpt-4"));
        
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
}


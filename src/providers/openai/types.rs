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

pub(super) fn text_request_to_openai(model: &str, request: TextRequest) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: request.messages.into_iter().map(message_to_chat).collect(),
        max_tokens: request.max_output_tokens,
        temperature: request.temperature,
    }
}

fn message_to_chat(msg: Message) -> ChatMessage {
    ChatMessage {
        role: match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .to_string(),
        content: msg.content,
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
        .unwrap_or_else(|| FinishReason::Other("missing".to_string()));

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

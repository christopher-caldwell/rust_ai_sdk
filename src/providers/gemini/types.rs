use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::core::{
    error::SdkError,
    message::{Message, MessagePart, Role, ToolCall},
    request::TextRequest,
    result::{ChatResult, TextResult},
    tool::ToolChoice,
    types::{FinishReason, ResponseMetadata, Usage as TokenUsage},
};

const GEMINI_METADATA_KEY: &str = "gemini";
const THOUGHT_SIGNATURE_KEY: &str = "thoughtSignature";
const FUNCTION_CALL_ID_KEY: &str = "functionCallId";
const FUNCTION_CALL_NAME_KEY: &str = "functionCallName";

#[derive(Debug, Clone)]
struct ToolCallRef {
    id: String,
    name: String,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(super) struct GenerateContentRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<GeminiTool>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GeminiToolConfig>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiSystemInstruction>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiFunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiFunctionCallingConfig {
    pub mode: &'static str,
    #[serde(rename = "allowedFunctionNames", skip_serializing_if = "Vec::is_empty")]
    pub allowed_function_names: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct GeminiGenerationConfig {
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(rename = "responseId")]
    pub response_id: Option<String>,
    #[serde(rename = "modelVersion")]
    pub model_version: Option<String>,
    #[serde(rename = "promptFeedback")]
    pub prompt_feedback: Option<GeminiPromptFeedback>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiCandidate {
    pub content: Option<GeminiResponseContent>,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiResponseContent {
    #[serde(default)]
    pub parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiResponsePart {
    pub text: Option<String>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<GeminiFunctionCallIn>,
    #[serde(rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiFunctionCallIn {
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Deserialize, Clone)]
pub(super) struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiPromptFeedback {
    #[serde(rename = "blockReason")]
    pub block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiErrorResponse {
    pub error: GeminiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(super) struct GeminiErrorDetail {
    pub message: String,
}

pub(super) fn text_request_to_gemini(
    request: &TextRequest,
) -> Result<GenerateContentRequest, SdkError> {
    let mut tool_call_refs = HashMap::new();
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    for msg in &request.messages {
        if matches!(msg.role, Role::System) {
            system_parts.extend(text_parts(msg));
            continue;
        }

        let parts = message_parts_to_gemini(msg, &mut tool_call_refs)?;
        if !parts.is_empty() {
            contents.push(GeminiContent {
                role: gemini_role(&msg.role).to_string(),
                parts,
            });
        }
    }

    let tools = if request.tools.is_empty() {
        vec![]
    } else {
        vec![GeminiTool {
            function_declarations: request
                .tools
                .iter()
                .map(|tool| GeminiFunctionDeclaration {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: gemini_schema(tool.input_schema.clone()),
                })
                .collect(),
        }]
    };

    let tool_config = request.tool_choice.as_ref().map(|choice| {
        let function_calling_config = match choice {
            ToolChoice::Auto => GeminiFunctionCallingConfig {
                mode: "AUTO",
                allowed_function_names: vec![],
            },
            ToolChoice::None => GeminiFunctionCallingConfig {
                mode: "NONE",
                allowed_function_names: vec![],
            },
            ToolChoice::Required { name } => GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: vec![name.clone()],
            },
        };
        GeminiToolConfig {
            function_calling_config,
        }
    });

    let generation_config = if request.max_output_tokens.is_some() || request.temperature.is_some()
    {
        Some(GeminiGenerationConfig {
            max_output_tokens: request.max_output_tokens,
            temperature: request.temperature,
        })
    } else {
        None
    };

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiSystemInstruction {
            parts: system_parts,
        })
    };

    Ok(GenerateContentRequest {
        contents,
        tools,
        tool_config,
        generation_config,
        system_instruction,
    })
}

fn gemini_role(role: &Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        Role::System | Role::User | Role::Tool => "user",
    }
}

fn text_parts(msg: &Message) -> Vec<GeminiPart> {
    msg.effective_parts()
        .into_iter()
        .filter_map(|part| match part {
            MessagePart::Text(text) => Some(GeminiPart {
                text: Some(text),
                function_call: None,
                function_response: None,
                thought_signature: None,
            }),
            MessagePart::ToolCall(_) | MessagePart::ToolResult(_) => None,
        })
        .collect()
}

fn message_parts_to_gemini(
    msg: &Message,
    tool_call_refs: &mut HashMap<String, ToolCallRef>,
) -> Result<Vec<GeminiPart>, SdkError> {
    let mut parts = Vec::new();

    for part in msg.effective_parts() {
        match part {
            MessagePart::Text(text) => parts.push(GeminiPart {
                text: Some(text),
                function_call: None,
                function_response: None,
                thought_signature: None,
            }),
            MessagePart::ToolCall(call) => {
                let provider_id = gemini_function_call_id(&call);
                let call_ref = ToolCallRef {
                    id: provider_id.clone(),
                    name: call.name.clone(),
                };
                tool_call_refs.insert(call.id.clone(), call_ref.clone());
                tool_call_refs.insert(provider_id, call_ref);
                parts.push(tool_call_part(&call));
            }
            MessagePart::ToolResult(result) => {
                let Some(call_ref) = tool_call_refs.get(&result.tool_call_id) else {
                    return Err(SdkError::Api(format!(
                        "Gemini tool result references unknown tool call id: {}",
                        result.tool_call_id
                    )));
                };
                parts.push(GeminiPart {
                    text: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        id: Some(call_ref.id.clone()),
                        name: call_ref.name.clone(),
                        response: tool_result_response_value(&result.content),
                    }),
                    thought_signature: None,
                });
            }
        }
    }

    Ok(parts)
}

fn tool_call_part(call: &ToolCall) -> GeminiPart {
    GeminiPart {
        text: None,
        function_call: Some(GeminiFunctionCall {
            id: Some(gemini_function_call_id(call)),
            name: call.name.clone(),
            args: call.input.clone(),
        }),
        function_response: None,
        thought_signature: gemini_thought_signature(call),
    }
}

fn tool_result_response_value(content: &str) -> Value {
    match serde_json::from_str::<Value>(content) {
        Ok(Value::Object(map)) => Value::Object(map),
        Ok(value) => json!({ "result": value }),
        Err(_) => json!({ "result": content }),
    }
}

pub(super) fn gemini_function_call_id(call: &ToolCall) -> String {
    gemini_metadata_string(call, FUNCTION_CALL_ID_KEY).unwrap_or_else(|| call.id.clone())
}

fn gemini_thought_signature(call: &ToolCall) -> Option<String> {
    gemini_metadata_string(call, THOUGHT_SIGNATURE_KEY)
}

fn gemini_metadata_string(call: &ToolCall, key: &str) -> Option<String> {
    let metadata = call.provider_metadata.as_ref()?;
    let gemini = metadata.get(GEMINI_METADATA_KEY).unwrap_or(metadata);
    gemini.get(key)?.as_str().map(ToString::to_string)
}

pub(super) fn gemini_tool_metadata(
    id: Option<&str>,
    name: &str,
    thought_signature: Option<&str>,
) -> Value {
    let mut gemini = Map::new();
    if let Some(id) = id {
        gemini.insert(
            FUNCTION_CALL_ID_KEY.to_string(),
            Value::String(id.to_string()),
        );
    }
    gemini.insert(
        FUNCTION_CALL_NAME_KEY.to_string(),
        Value::String(name.to_string()),
    );
    if let Some(signature) = thought_signature {
        gemini.insert(
            THOUGHT_SIGNATURE_KEY.to_string(),
            Value::String(signature.to_string()),
        );
    }

    let mut root = Map::new();
    root.insert(GEMINI_METADATA_KEY.to_string(), Value::Object(gemini));
    Value::Object(root)
}

fn gemini_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut map) => {
            map.remove("additionalProperties");
            Value::Object(
                map.into_iter()
                    .map(|(key, value)| (key, gemini_schema(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(gemini_schema).collect()),
        other => other,
    }
}

pub(super) fn map_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "RECITATION"
        | "LANGUAGE"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

fn usage_from_metadata(usage: Option<GeminiUsageMetadata>) -> Option<TokenUsage> {
    usage.map(|u| TokenUsage {
        input_tokens: u.prompt_token_count,
        output_tokens: u.candidates_token_count,
        total_tokens: u.total_token_count,
    })
}

fn response_metadata(resp: &GenerateContentResponse) -> ResponseMetadata {
    ResponseMetadata {
        id: resp.response_id.clone(),
        model: resp.model_version.clone(),
    }
}

pub(super) fn gemini_response_to_text_result(
    resp: GenerateContentResponse,
) -> Result<TextResult, SdkError> {
    let Some(choice) = resp.candidates.first() else {
        let blocked = resp
            .prompt_feedback
            .as_ref()
            .and_then(|feedback| feedback.block_reason.as_deref())
            .unwrap_or("no candidates");
        return Err(SdkError::Api(format!(
            "Gemini response contained no candidates: {}",
            blocked
        )));
    };

    let text = choice
        .content
        .as_ref()
        .map(|content| {
            content
                .parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(map_finish_reason)
        .unwrap_or_else(|| FinishReason::Other("unknown".to_string()));

    Ok(TextResult {
        text,
        finish_reason,
        usage: usage_from_metadata(resp.usage_metadata.clone()),
        response: response_metadata(&resp),
    })
}

pub(super) fn gemini_response_to_chat_result(
    resp: GenerateContentResponse,
) -> Result<ChatResult, SdkError> {
    let Some(choice) = resp.candidates.first() else {
        let blocked = resp
            .prompt_feedback
            .as_ref()
            .and_then(|feedback| feedback.block_reason.as_deref())
            .unwrap_or("no candidates");
        return Err(SdkError::Api(format!(
            "Gemini response contained no candidates: {}",
            blocked
        )));
    };

    let mut parts = Vec::new();
    if let Some(content) = &choice.content {
        for part in &content.parts {
            if let Some(text) = &part.text {
                if !text.is_empty() {
                    parts.push(MessagePart::Text(text.clone()));
                }
            }
            if let Some(call) = &part.function_call {
                let id = call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("gemini_call_{}", parts.len()));
                let metadata = gemini_tool_metadata(
                    call.id.as_deref(),
                    &call.name,
                    part.thought_signature.as_deref(),
                );
                parts.push(MessagePart::ToolCall(
                    ToolCall::new(id, call.name.clone(), call.args.clone())
                        .with_provider_metadata(metadata),
                ));
            }
        }
    }

    let has_tool_calls = parts
        .iter()
        .any(|part| matches!(part, MessagePart::ToolCall(_)));
    let finish_reason = if has_tool_calls {
        FinishReason::ToolUse
    } else {
        choice
            .finish_reason
            .as_deref()
            .map(map_finish_reason)
            .unwrap_or_else(|| FinishReason::Other("unknown".to_string()))
    };

    Ok(ChatResult {
        parts,
        finish_reason,
        usage: usage_from_metadata(resp.usage_metadata.clone()),
        response: response_metadata(&resp),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        message::{Message, ToolResult},
        tool::ToolDefinition,
    };

    #[test]
    fn request_maps_prompt() {
        let request = TextRequest::prompt("hello");
        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(json["contents"][0]["role"], "user");
        assert_eq!(json["contents"][0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn request_extracts_system_instruction() {
        let request = TextRequest::builder()
            .system("be brief")
            .prompt("hello")
            .build();
        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(json["systemInstruction"]["parts"][0]["text"], "be brief");
        assert_eq!(json["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn request_maps_multiturn_roles() {
        let request = TextRequest::new(vec![
            Message::user("hello"),
            Message::assistant("hi"),
            Message::user("again"),
        ]);
        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(json["contents"][0]["role"], "user");
        assert_eq!(json["contents"][1]["role"], "model");
        assert_eq!(json["contents"][2]["role"], "user");
    }

    #[test]
    fn request_maps_generation_config() {
        let request = TextRequest::prompt("hello")
            .with_max_output_tokens(128)
            .with_temperature(0.2);
        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(json["generationConfig"]["maxOutputTokens"], 128);
        let temperature = json["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((temperature - 0.2).abs() < 0.000001);
    }

    #[test]
    fn request_maps_function_declarations() {
        let request = TextRequest::prompt("weather").with_tools(vec![ToolDefinition::new(
            "get_weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            }),
        )]);
        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(
            json["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert!(
            json["tools"][0]["functionDeclarations"][0]["parameters"]
                .get("additionalProperties")
                .is_none()
        );
        assert!(
            json["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["location"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn request_maps_tool_choice_modes() {
        let auto =
            text_request_to_gemini(&TextRequest::prompt("x").with_tool_choice(ToolChoice::Auto))
                .unwrap();
        let none =
            text_request_to_gemini(&TextRequest::prompt("x").with_tool_choice(ToolChoice::None))
                .unwrap();
        let required = text_request_to_gemini(&TextRequest::prompt("x").with_tool_choice(
            ToolChoice::Required {
                name: "get_weather".to_string(),
            },
        ))
        .unwrap();

        let auto = serde_json::to_value(auto).unwrap();
        let none = serde_json::to_value(none).unwrap();
        let required = serde_json::to_value(required).unwrap();

        assert_eq!(auto["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
        assert_eq!(none["toolConfig"]["functionCallingConfig"]["mode"], "NONE");
        assert_eq!(
            required["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );
        assert_eq!(
            required["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "get_weather"
        );
    }

    #[test]
    fn request_maps_tool_call_and_result_with_metadata() {
        let call = ToolCall::new("sdk_call_1", "get_weather", json!({"location": "Paris"}))
            .with_provider_metadata(gemini_tool_metadata(
                Some("provider_call_1"),
                "get_weather",
                Some("sig_123"),
            ));
        let request = TextRequest::new(vec![
            Message::user("weather"),
            Message::assistant_parts(vec![MessagePart::ToolCall(call)]),
            Message {
                role: Role::User,
                content: String::new(),
                parts: vec![MessagePart::ToolResult(ToolResult {
                    tool_call_id: "sdk_call_1".to_string(),
                    content: json!({"forecast": "cloudy"}).to_string(),
                })],
            },
        ]);

        let body = text_request_to_gemini(&request).unwrap();
        let json = serde_json::to_value(body).unwrap();

        assert_eq!(
            json["contents"][1]["parts"][0]["functionCall"]["id"],
            "provider_call_1"
        );
        assert_eq!(
            json["contents"][1]["parts"][0]["functionCall"]["name"],
            "get_weather"
        );
        assert_eq!(
            json["contents"][1]["parts"][0]["thoughtSignature"],
            "sig_123"
        );
        assert_eq!(
            json["contents"][2]["parts"][0]["functionResponse"]["id"],
            "provider_call_1"
        );
        assert_eq!(
            json["contents"][2]["parts"][0]["functionResponse"]["name"],
            "get_weather"
        );
        assert_eq!(
            json["contents"][2]["parts"][0]["functionResponse"]["response"]["forecast"],
            "cloudy"
        );
    }

    #[test]
    fn request_rejects_tool_result_before_prior_tool_call() {
        let request = TextRequest::new(vec![
            Message {
                role: Role::User,
                content: String::new(),
                parts: vec![MessagePart::ToolResult(ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: json!({"forecast": "cloudy"}).to_string(),
                })],
            },
            Message::assistant_parts(vec![MessagePart::ToolCall(ToolCall::new(
                "call_1",
                "get_weather",
                json!({"location": "Paris"}),
            ))]),
        ]);

        let err = text_request_to_gemini(&request).unwrap_err();
        assert!(matches!(err, SdkError::Api(message) if message.contains("call_1")));
    }

    #[test]
    fn text_response_maps_usage_and_metadata() {
        let resp: GenerateContentResponse = serde_json::from_value(json!({
            "responseId": "resp_1",
            "modelVersion": "gemini-2.5-flash",
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5
            }
        }))
        .unwrap();

        let result = gemini_response_to_text_result(resp).unwrap();
        assert_eq!(result.text, "Hello");
        assert!(matches!(result.finish_reason, FinishReason::Stop));
        assert_eq!(result.response.id.as_deref(), Some("resp_1"));
        assert_eq!(result.response.model.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(result.usage.unwrap().total_tokens, Some(5));
    }

    #[test]
    fn chat_response_maps_tool_call_with_thought_signature() {
        let resp: GenerateContentResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"location": "Paris"}
                    },
                    "thoughtSignature": "sig_123"
                }]},
                "finishReason": "STOP"
            }]
        }))
        .unwrap();

        let result = gemini_response_to_chat_result(resp).unwrap();
        assert!(matches!(result.finish_reason, FinishReason::ToolUse));
        let calls = result.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "Paris");
        assert_eq!(
            calls[0].provider_metadata.as_ref().unwrap()["gemini"]["thoughtSignature"],
            "sig_123"
        );
    }

    #[test]
    fn no_candidates_returns_api_error() {
        let resp: GenerateContentResponse = serde_json::from_value(json!({
            "promptFeedback": {"blockReason": "SAFETY"}
        }))
        .unwrap();

        let err = gemini_response_to_chat_result(resp).unwrap_err();
        assert!(matches!(err, SdkError::Api(message) if message.contains("SAFETY")));
    }
}

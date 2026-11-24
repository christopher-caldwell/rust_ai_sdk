#[cfg(any(test, feature = "gemini"))]
use serde_json::Value;

use crate::core::{tool::ToolChoice, types::FinishReason};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoicePolicy<'a> {
    /// The request makes no provider-neutral assertion about tool use.
    Default,
    /// The model may decide whether to call a tool.
    Auto,
    /// The model must not call tools for this request.
    None,
    /// The model must call the named tool.
    Required { name: &'a str },
}

pub fn tool_choice_policy(choice: Option<&ToolChoice>) -> ToolChoicePolicy<'_> {
    match choice {
        Some(ToolChoice::Auto) => ToolChoicePolicy::Auto,
        Some(ToolChoice::None) => ToolChoicePolicy::None,
        Some(ToolChoice::Required { name }) => ToolChoicePolicy::Required { name },
        None => ToolChoicePolicy::Default,
    }
}

/// Normalize the provider's finish reason for a response.
///
/// Tool calls normally report [`FinishReason::ToolUse`], whatever the provider
/// said. The exception is [`FinishReason::Length`]: a response cut off by the
/// output-token limit may end inside a tool call whose input is incomplete, so
/// the length stop is kept to let callers detect the truncation and retry with
/// a higher limit.
pub fn finish_reason_with_tool_override(
    provider_finish_reason: Option<FinishReason>,
    has_tool_calls: bool,
) -> FinishReason {
    if has_tool_calls && provider_finish_reason != Some(FinishReason::Length) {
        return FinishReason::ToolUse;
    }

    provider_finish_reason.unwrap_or_else(unknown_finish_reason)
}

pub fn unknown_finish_reason() -> FinishReason {
    FinishReason::Other("unknown".to_string())
}

#[cfg(any(test, feature = "gemini"))]
pub fn remove_additional_properties(schema: Value) -> Value {
    match schema {
        Value::Object(mut map) => {
            map.remove("additionalProperties");
            Value::Object(
                map.into_iter()
                    .map(|(key, value)| (key, remove_additional_properties(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(remove_additional_properties)
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_choice_policy_documents_provider_neutral_meaning() {
        assert_eq!(tool_choice_policy(None), ToolChoicePolicy::Default);
        assert_eq!(
            tool_choice_policy(Some(&ToolChoice::Auto)),
            ToolChoicePolicy::Auto
        );
        assert_eq!(
            tool_choice_policy(Some(&ToolChoice::None)),
            ToolChoicePolicy::None
        );
        assert_eq!(
            tool_choice_policy(Some(&ToolChoice::Required {
                name: "get_weather".to_string(),
            })),
            ToolChoicePolicy::Required {
                name: "get_weather"
            }
        );
    }

    #[test]
    fn tool_calls_take_precedence_over_provider_finish_reason() {
        let finish_reason = finish_reason_with_tool_override(Some(FinishReason::Stop), true);

        assert_eq!(finish_reason, FinishReason::ToolUse);
    }

    #[test]
    fn length_stop_is_kept_when_tool_calls_are_present() {
        let finish_reason = finish_reason_with_tool_override(Some(FinishReason::Length), true);

        assert_eq!(finish_reason, FinishReason::Length);
    }

    #[test]
    fn missing_provider_finish_reason_with_tool_calls_becomes_tool_use() {
        let finish_reason = finish_reason_with_tool_override(None, true);

        assert_eq!(finish_reason, FinishReason::ToolUse);
    }

    #[test]
    fn missing_provider_finish_reason_becomes_unknown() {
        let finish_reason = finish_reason_with_tool_override(None, false);

        assert_eq!(finish_reason, FinishReason::Other("unknown".to_string()));
    }

    #[test]
    fn additional_properties_are_removed_recursively() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "location": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "city": {
                            "type": "string",
                            "additionalProperties": false
                        }
                    }
                }
            }
        });

        let adapted = remove_additional_properties(schema);

        assert!(adapted.get("additionalProperties").is_none());
        assert!(
            adapted["properties"]["location"]
                .get("additionalProperties")
                .is_none()
        );
        assert!(
            adapted["properties"]["location"]["properties"]["city"]
                .get("additionalProperties")
                .is_none()
        );
    }
}

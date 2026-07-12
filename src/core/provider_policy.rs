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

pub fn finish_reason_with_tool_override(
    provider_finish_reason: Option<FinishReason>,
    has_tool_calls: bool,
) -> FinishReason {
    if has_tool_calls {
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

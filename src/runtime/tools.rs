use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

use serde_json::Value;

use crate::core::{error::SdkError, message::ToolCall, tool::ToolDefinition};

type ToolFuture = Pin<Box<dyn Future<Output = Result<Value, SdkError>> + Send>>;
type ToolHandler = Arc<dyn Fn(ToolCall) -> ToolFuture + Send + Sync>;

#[derive(Clone)]
struct RegisteredTool {
    definition: ToolDefinition,
    handler: ToolHandler,
}

/// A small application-side registry for tool definitions and execution.
///
/// The SDK stays provider-neutral: callers decide which tools are registered
/// and what each tool is allowed to do.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, rejecting invalid definitions or duplicate names.
    pub fn register<F, Fut>(
        mut self,
        definition: ToolDefinition,
        handler: F,
    ) -> Result<Self, SdkError>
    where
        F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, SdkError>> + Send + 'static,
    {
        definition.validate()?;
        let name = definition.name.clone();
        if self.tools.contains_key(&name) {
            return Err(SdkError::Validation(format!(
                "tool registry already contains a tool named '{name}'",
            )));
        }
        let handler = Arc::new(move |call: ToolCall| -> ToolFuture { Box::pin(handler(call)) });
        self.tools.insert(
            name,
            RegisteredTool {
                definition,
                handler,
            },
        );
        Ok(self)
    }

    /// Return registered definitions in deterministic name order.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    /// Execute a validated call through its registered handler.
    pub async fn execute(&self, call: &ToolCall) -> Result<Value, SdkError> {
        if call.has_malformed_input() {
            let parse_error = call
                .malformed_input_error()
                .unwrap_or("unknown JSON parse error");

            return Err(SdkError::Validation(format!(
                "tool call '{}' for tool '{}' has malformed JSON input and will not be executed: {}",
                call.id, call.name, parse_error,
            )));
        }

        let Some(tool) = self.tools.get(&call.name) else {
            return Err(SdkError::Unknown(format!("unknown tool: {}", call.name)));
        };

        validate_input_against_schema(&tool.definition, &call.input)?;

        (tool.handler)(call.clone()).await
    }

    /// Return whether a tool name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

fn validate_input_against_schema(
    definition: &ToolDefinition,
    input: &Value,
) -> Result<(), SdkError> {
    let input_object = input.as_object().ok_or_else(|| {
        SdkError::Validation(format!(
            "tool '{}' input must be a JSON object",
            definition.name,
        ))
    })?;

    if let Some(required) = definition
        .input_schema
        .get("required")
        .and_then(Value::as_array)
    {
        for field in required.iter().filter_map(Value::as_str) {
            if !input_object.contains_key(field) {
                return Err(SdkError::Validation(format!(
                    "tool '{}' input is missing required field '{field}'",
                    definition.name,
                )));
            }
        }
    }

    if let Some(properties) = definition
        .input_schema
        .get("properties")
        .and_then(Value::as_object)
    {
        for (field, value) in input_object {
            let Some(property_schema) = properties.get(field) else {
                if definition.input_schema.get("additionalProperties") == Some(&Value::Bool(false))
                {
                    return Err(SdkError::Validation(format!(
                        "tool '{}' input contains unknown field '{field}'",
                        definition.name,
                    )));
                }
                continue;
            };

            if let Some(expected_type) = property_schema.get("type").and_then(Value::as_str)
                && !json_type_matches(value, expected_type)
            {
                return Err(SdkError::Validation(format!(
                    "tool '{}' field '{field}' must have JSON type {expected_type}",
                    definition.name,
                )));
            }
        }
    }

    Ok(())
}

fn json_type_matches(value: &Value, expected_type: &str) -> bool {
    match expected_type {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn registry_returns_definitions_and_executes_tools() {
        let registry = ToolRegistry::new()
            .register(
                ToolDefinition::new("echo", "Echo input", json!({"type": "object"})),
                |call| async move { Ok(json!({ "input": call.input })) },
            )
            .unwrap();

        assert!(registry.contains("echo"));
        assert_eq!(registry.definitions()[0].name, "echo");

        let output = registry
            .execute(&ToolCall::new("call_1", "echo", json!({ "value": 42 })))
            .await
            .unwrap();

        assert_eq!(output["input"]["value"], 42);
    }

    #[tokio::test]
    async fn registry_errors_for_unknown_tools() {
        let registry = ToolRegistry::new();

        let error = registry
            .execute(&ToolCall::new("call_1", "missing", Value::Null))
            .await
            .unwrap_err();

        assert!(matches!(error, SdkError::Unknown(message) if message.contains("missing")));
    }

    #[tokio::test]
    async fn registry_rejects_malformed_tool_input_before_handler_runs() {
        let registry = ToolRegistry::new()
            .register(
                ToolDefinition::new("echo", "Echo input", json!({"type": "object"})),
                |_call| async move { Ok(json!({ "handler": "ran" })) },
            )
            .unwrap();
        let call = ToolCall::malformed_json_input("call_1", "echo", "{broken", "expected value");

        let error = registry.execute(&call).await.unwrap_err();

        assert!(
            matches!(error, SdkError::Validation(message) if message.contains("malformed JSON input"))
        );
    }

    #[tokio::test]
    async fn registry_validates_input_schema_before_handler_runs() {
        let registry = ToolRegistry::new()
            .register(
                ToolDefinition::new(
                    "weather",
                    "Get weather",
                    json!({
                        "type": "object",
                        "properties": { "location": { "type": "string" } },
                        "required": ["location"],
                        "additionalProperties": false
                    }),
                ),
                |_call| async move { Ok(json!({"handler": "ran"})) },
            )
            .unwrap();

        let error = registry
            .execute(&ToolCall::new("call_1", "weather", json!({"location": 42})))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("must have JSON type string")
        ));
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let registry = ToolRegistry::new()
            .register(
                ToolDefinition::new("echo", "Echo input", json!({"type": "object"})),
                |_call| async move { Ok(Value::Null) },
            )
            .unwrap();

        let error = registry
            .register(
                ToolDefinition::new("echo", "Another echo", json!({"type": "object"})),
                |_call| async move { Ok(Value::Null) },
            )
            .err()
            .expect("duplicate registration should fail");

        assert!(matches!(
            error,
            SdkError::Validation(message) if message.contains("already contains")
        ));
    }
}

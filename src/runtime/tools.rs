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
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn register<F, Fut>(mut self, definition: ToolDefinition, handler: F) -> Self
    where
        F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, SdkError>> + Send + 'static,
    {
        let name = definition.name.clone();
        let handler = Arc::new(move |call: ToolCall| -> ToolFuture { Box::pin(handler(call)) });
        self.tools.insert(
            name,
            RegisteredTool {
                definition,
                handler,
            },
        );
        self
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    pub async fn execute(&self, call: &ToolCall) -> Result<Value, SdkError> {
        let Some(tool) = self.tools.get(&call.name) else {
            return Err(SdkError::Unknown(format!("unknown tool: {}", call.name)));
        };

        (tool.handler)(call.clone()).await
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn registry_returns_definitions_and_executes_tools() {
        let registry = ToolRegistry::new().register(
            ToolDefinition::new("echo", "Echo input", json!({"type": "object"})),
            |call| async move { Ok(json!({ "input": call.input })) },
        );

        assert!(registry.contains("echo"));
        assert_eq!(registry.definitions()[0].name, "echo");

        let output = registry
            .execute(&ToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: json!({ "value": 42 }),
            })
            .await
            .unwrap();

        assert_eq!(output["input"]["value"], 42);
    }

    #[tokio::test]
    async fn registry_errors_for_unknown_tools() {
        let registry = ToolRegistry::new();

        let error = registry
            .execute(&ToolCall {
                id: "call_1".to_string(),
                name: "missing".to_string(),
                input: Value::Null,
            })
            .await
            .unwrap_err();

        assert!(matches!(error, SdkError::Unknown(message) if message.contains("missing")));
    }
}

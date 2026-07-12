use serde_json::Value;

use super::error::SdkError;

/// Provider-neutral tool definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Model-visible tool name and registry key.
    pub name: String,
    /// Model-visible description of when and how to use the tool.
    pub description: String,
    /// JSON Schema object describing the tool's input parameters.
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a provider-neutral tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// Validate the definition before advertising or registering the tool.
    pub fn validate(&self) -> Result<(), SdkError> {
        if self.name.trim().is_empty() {
            return Err(SdkError::Validation(
                "tool definition must have a non-empty name".to_string(),
            ));
        }
        if self.description.trim().is_empty() {
            return Err(SdkError::Validation(format!(
                "tool definition '{}' must have a non-empty description",
                self.name,
            )));
        }
        if !self.input_schema.is_object() {
            return Err(SdkError::Validation(format!(
                "tool definition '{}' input_schema must be a JSON object",
                self.name,
            )));
        }

        Ok(())
    }
}

/// Controls which tool(s) the model may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model decides whether to call a tool (default).
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call exactly this named tool.
    Required {
        /// Required registered tool name.
        name: String,
    },
}

impl ToolChoice {
    /// Require the model to call the named tool.
    pub fn required(name: impl Into<String>) -> Self {
        Self::Required { name: name.into() }
    }
}

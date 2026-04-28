use serde_json::Value;

/// Provider-neutral tool definition.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the tool's input parameters.
    pub input_schema: Value,
}

impl ToolDefinition {
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
}

/// Controls which tool(s) the model may call.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// Model decides whether to call a tool (default).
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call exactly this named tool.
    Required { name: String },
}

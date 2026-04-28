use std::collections::HashMap;

use futures_util::StreamExt;
use serde_json::Value;

use crate::core::{
    error::SdkError,
    message::{Message, MessagePart, ToolCall},
    model::LanguageModel,
    request::TextRequest,
    result::ChatResult,
    stream::StreamEvent,
    types::{FinishReason, ResponseMetadata, Usage},
};

/// The outcome of one model turn.
#[derive(Debug)]
pub enum TurnOutcome {
    /// Model finished with no tool calls.
    Completed(ChatResult),
    /// Model emitted one or more tool calls; the caller must execute them.
    ToolsRequired {
        tool_calls: Vec<ToolCall>,
        /// All parts of the assistant's turn — needed to build the continuation request.
        assistant_parts: Vec<MessagePart>,
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    },
}

/// Run one model turn, accumulate the stream into a `TurnOutcome`.
pub async fn run_turn<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TurnOutcome, SdkError> {
    let mut stream = model.stream(request).await?;
    let mut acc = TurnAccumulator::default();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(t) => acc.push_text(t),
            StreamEvent::ToolCallStarted { id, name, index } => {
                acc.start_tool_call(index, id, name);
            }
            StreamEvent::ToolCallDelta {
                index, input_delta, ..
            } => {
                acc.append_tool_delta(index, input_delta);
            }
            StreamEvent::ToolCallReady { .. } => {
                // Informational — accumulator already has the full input from deltas.
            }
            StreamEvent::Finished {
                finish_reason,
                usage,
                response,
            } => {
                acc.set_finish(finish_reason, usage, response);
            }
        }
    }

    acc.into_outcome()
}

/// Build a continuation request by appending the assistant's turn and tool results.
pub struct ContinuationBuilder {
    request: TextRequest,
}

impl ContinuationBuilder {
    pub fn from_request(request: TextRequest) -> Self {
        Self { request }
    }

    pub fn with_assistant_turn(mut self, parts: Vec<MessagePart>) -> Self {
        self.request.messages.push(Message::assistant_parts(parts));
        self
    }

    pub fn with_tool_result(
        mut self,
        tool_call_id: impl Into<String>,
        result_content: impl Into<String>,
    ) -> Self {
        self.request
            .messages
            .push(Message::tool_result(tool_call_id, result_content));
        self
    }

    pub fn with_tool_results(
        mut self,
        results: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        for (id, content) in results {
            self.request
                .messages
                .push(Message::tool_result(id, content));
        }
        self
    }

    pub fn build(self) -> TextRequest {
        self.request
    }
}

// ---------------------------------------------------------------------------
// Internal accumulator
// ---------------------------------------------------------------------------

struct ToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct TurnAccumulator {
    parts_order: Vec<PartSlot>,
    tool_buffers: HashMap<u32, ToolCallBuffer>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    response: Option<ResponseMetadata>,
}

enum PartSlot {
    Text(String),
    ToolIndex(u32),
}

impl TurnAccumulator {
    fn push_text(&mut self, text: String) {
        if let Some(PartSlot::Text(existing)) = self.parts_order.last_mut() {
            existing.push_str(&text);
        } else {
            self.parts_order.push(PartSlot::Text(text));
        }
    }

    fn start_tool_call(&mut self, index: u32, id: String, name: String) {
        self.tool_buffers.insert(
            index,
            ToolCallBuffer {
                id,
                name,
                arguments: String::new(),
            },
        );
        self.parts_order.push(PartSlot::ToolIndex(index));
    }

    fn append_tool_delta(&mut self, index: u32, delta: String) {
        if let Some(buf) = self.tool_buffers.get_mut(&index) {
            buf.arguments.push_str(&delta);
        }
    }

    fn set_finish(
        &mut self,
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    ) {
        self.finish_reason = Some(finish_reason);
        self.usage = usage;
        self.response = Some(response);
    }

    fn into_outcome(mut self) -> Result<TurnOutcome, SdkError> {
        let finish_reason = self
            .finish_reason
            .unwrap_or(FinishReason::Other("unknown".to_string()));
        let usage = self.usage;
        let response = self.response.unwrap_or(ResponseMetadata {
            id: None,
            model: None,
        });

        let mut parts: Vec<MessagePart> = Vec::new();
        for slot in self.parts_order {
            match slot {
                PartSlot::Text(t) => {
                    if !t.is_empty() {
                        parts.push(MessagePart::Text(t));
                    }
                }
                PartSlot::ToolIndex(idx) => {
                    if let Some(buf) = self.tool_buffers.remove(&idx) {
                        let input: Value = serde_json::from_str(&buf.arguments)
                            .unwrap_or(Value::String(buf.arguments.clone()));
                        parts.push(MessagePart::ToolCall(ToolCall {
                            id: buf.id,
                            name: buf.name,
                            input,
                        }));
                    }
                }
            }
        }

        let tool_calls: Vec<ToolCall> = parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::ToolCall(tc) = p {
                    Some(tc.clone())
                } else {
                    None
                }
            })
            .collect();

        if tool_calls.is_empty() {
            Ok(TurnOutcome::Completed(ChatResult {
                parts,
                finish_reason,
                usage,
                response,
            }))
        } else {
            Ok(TurnOutcome::ToolsRequired {
                tool_calls,
                assistant_parts: parts,
                finish_reason,
                usage,
                response,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::ResponseMetadata;
    use futures_util::stream;

    fn make_stream(events: Vec<StreamEvent>) -> crate::core::stream::TextEventStream {
        Box::pin(stream::iter(events.into_iter().map(Ok)))
    }

    struct MockModel {
        events: std::sync::Mutex<Option<Vec<StreamEvent>>>,
    }

    impl MockModel {
        fn new(events: Vec<StreamEvent>) -> Self {
            Self {
                events: std::sync::Mutex::new(Some(events)),
            }
        }
    }

    #[async_trait::async_trait]
    impl LanguageModel for MockModel {
        async fn generate(
            &self,
            _request: TextRequest,
        ) -> Result<crate::core::result::TextResult, SdkError> {
            unimplemented!()
        }

        async fn stream(
            &self,
            _request: TextRequest,
        ) -> Result<crate::core::stream::TextEventStream, SdkError> {
            let events = self.events.lock().unwrap().take().unwrap_or_default();
            Ok(make_stream(events))
        }

        fn model_id(&self) -> &str {
            "mock"
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    fn meta() -> ResponseMetadata {
        ResponseMetadata {
            id: Some("r1".to_string()),
            model: Some("m".to_string()),
        }
    }

    #[tokio::test]
    async fn test_run_turn_text_only() {
        let model = MockModel::new(vec![
            StreamEvent::TextDelta("Hello ".to_string()),
            StreamEvent::TextDelta("world".to_string()),
            StreamEvent::Finished {
                finish_reason: FinishReason::Stop,
                usage: None,
                response: meta(),
            },
        ]);

        let request = TextRequest::prompt("hi");
        let outcome = run_turn(&model, request).await.unwrap();

        match outcome {
            TurnOutcome::Completed(result) => {
                assert_eq!(result.text(), "Hello world");
                assert!(matches!(result.finish_reason, FinishReason::Stop));
            }
            TurnOutcome::ToolsRequired { .. } => panic!("Expected Completed"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_single_tool_call() {
        let model = MockModel::new(vec![
            StreamEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                index: 0,
            },
            StreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                index: 0,
                input_delta: r#"{"location":"Paris"}"#.to_string(),
            },
            StreamEvent::ToolCallReady {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                index: 0,
                input: serde_json::json!({"location": "Paris"}),
            },
            StreamEvent::Finished {
                finish_reason: FinishReason::ToolUse,
                usage: None,
                response: meta(),
            },
        ]);

        let request = TextRequest::prompt("weather?");
        let outcome = run_turn(&model, request).await.unwrap();

        match outcome {
            TurnOutcome::ToolsRequired { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_1");
                assert_eq!(tool_calls[0].name, "get_weather");
                assert_eq!(tool_calls[0].input["location"], "Paris");
            }
            TurnOutcome::Completed(_) => panic!("Expected ToolsRequired"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_parallel_tool_calls() {
        let model = MockModel::new(vec![
            StreamEvent::ToolCallStarted {
                id: "call_a".to_string(),
                name: "tool_a".to_string(),
                index: 0,
            },
            StreamEvent::ToolCallStarted {
                id: "call_b".to_string(),
                name: "tool_b".to_string(),
                index: 1,
            },
            StreamEvent::ToolCallDelta {
                id: "call_a".to_string(),
                index: 0,
                input_delta: r#"{"x":1}"#.to_string(),
            },
            StreamEvent::ToolCallDelta {
                id: "call_b".to_string(),
                index: 1,
                input_delta: r#"{"y":2}"#.to_string(),
            },
            StreamEvent::Finished {
                finish_reason: FinishReason::ToolUse,
                usage: None,
                response: meta(),
            },
        ]);

        let request = TextRequest::prompt("parallel tools");
        let outcome = run_turn(&model, request).await.unwrap();

        match outcome {
            TurnOutcome::ToolsRequired { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 2);
                let names: Vec<&str> = tool_calls.iter().map(|tc| tc.name.as_str()).collect();
                assert!(names.contains(&"tool_a"));
                assert!(names.contains(&"tool_b"));
            }
            TurnOutcome::Completed(_) => panic!("Expected ToolsRequired"),
        }
    }

    #[test]
    fn test_continuation_builder_message_order() {
        let request = TextRequest::prompt("original");
        let parts = vec![
            MessagePart::Text("thinking...".to_string()),
            MessagePart::ToolCall(ToolCall {
                id: "c1".to_string(),
                name: "tool".to_string(),
                input: serde_json::json!({}),
            }),
        ];

        let continuation = ContinuationBuilder::from_request(request)
            .with_assistant_turn(parts)
            .with_tool_result("c1", "42 degrees")
            .build();

        assert_eq!(continuation.messages.len(), 3);
        assert!(matches!(
            continuation.messages[0].role,
            crate::core::message::Role::User
        ));
        assert!(matches!(
            continuation.messages[1].role,
            crate::core::message::Role::Assistant
        ));
        assert!(matches!(
            continuation.messages[2].role,
            crate::core::message::Role::User
        ));
    }
}

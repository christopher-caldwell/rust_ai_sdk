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
        acc.push_event(event?);
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
// Stream turn accumulation
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
    input: Option<Value>,
}

/// Accumulates a provider-neutral stream into one assistant turn.
///
/// This is useful for HTTP adapters that need to forward individual stream
/// events while still reconstructing the assistant message for continuation.
#[derive(Debug, Default)]
pub struct TurnAccumulator {
    parts_order: Vec<PartSlot>,
    tool_buffers: HashMap<u32, ToolCallBuffer>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    response: Option<ResponseMetadata>,
}

#[derive(Debug)]
enum PartSlot {
    Text(String),
    ToolIndex(u32),
}

impl TurnAccumulator {
    pub fn push_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::TextDelta(text) => self.push_text(text),
            StreamEvent::ToolCallStarted { id, name, index } => {
                self.start_tool_call(index, id, name);
            }
            StreamEvent::ToolCallDelta {
                id,
                index,
                input_delta,
            } => {
                self.append_tool_delta(index, id, input_delta);
            }
            StreamEvent::ToolCallReady {
                id,
                name,
                index,
                input,
            } => {
                self.ready_tool_call(index, id, name, input);
            }
            StreamEvent::Finished {
                finish_reason,
                usage,
                response,
            } => {
                self.set_finish(finish_reason, usage, response);
            }
        }
    }

    pub fn into_accumulated(mut self) -> AccumulatedTurn {
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
                        let input = buf.input.unwrap_or_else(|| {
                            serde_json::from_str(&buf.arguments)
                                .unwrap_or(Value::String(buf.arguments.clone()))
                        });
                        parts.push(MessagePart::ToolCall(ToolCall {
                            id: buf.id,
                            name: buf.name,
                            input,
                        }));
                    }
                }
            }
        }

        AccumulatedTurn {
            parts,
            finish_reason,
            usage,
            response,
        }
    }

    pub fn into_outcome(self) -> Result<TurnOutcome, SdkError> {
        let turn = self.into_accumulated();
        let tool_calls = turn.tool_calls_cloned();

        if tool_calls.is_empty() {
            Ok(TurnOutcome::Completed(ChatResult {
                parts: turn.parts,
                finish_reason: turn.finish_reason,
                usage: turn.usage,
                response: turn.response,
            }))
        } else {
            Ok(TurnOutcome::ToolsRequired {
                tool_calls,
                assistant_parts: turn.parts,
                finish_reason: turn.finish_reason,
                usage: turn.usage,
                response: turn.response,
            })
        }
    }

    fn push_text(&mut self, text: String) {
        if let Some(PartSlot::Text(existing)) = self.parts_order.last_mut() {
            existing.push_str(&text);
        } else {
            self.parts_order.push(PartSlot::Text(text));
        }
    }

    fn start_tool_call(&mut self, index: u32, id: String, name: String) {
        self.ensure_tool_slot(index);
        let buffer = self.tool_buffers.entry(index).or_insert(ToolCallBuffer {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
            input: None,
        });
        buffer.id = id;
        buffer.name = name;
    }

    fn append_tool_delta(&mut self, index: u32, id: String, delta: String) {
        self.ensure_tool_slot(index);
        let buffer = self.tool_buffers.entry(index).or_insert(ToolCallBuffer {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
            input: None,
        });
        if !id.is_empty() {
            buffer.id = id;
        }
        buffer.arguments.push_str(&delta);
    }

    fn ready_tool_call(&mut self, index: u32, id: String, name: String, input: Value) {
        self.ensure_tool_slot(index);
        let buffer = self.tool_buffers.entry(index).or_insert(ToolCallBuffer {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
            input: None,
        });
        buffer.id = id;
        buffer.name = name;
        buffer.input = Some(input);
    }

    fn ensure_tool_slot(&mut self, index: u32) {
        if !self
            .parts_order
            .iter()
            .any(|slot| matches!(slot, PartSlot::ToolIndex(existing) if *existing == index))
        {
            self.parts_order.push(PartSlot::ToolIndex(index));
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
}

/// A completed assistant turn reconstructed from stream events.
#[derive(Debug, Clone)]
pub struct AccumulatedTurn {
    pub parts: Vec<MessagePart>,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub response: ResponseMetadata,
}

impl AccumulatedTurn {
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::Text(text) = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.parts
            .iter()
            .filter_map(|p| {
                if let MessagePart::ToolCall(tool_call) = p {
                    Some(tool_call)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn tool_calls_cloned(&self) -> Vec<ToolCall> {
        self.tool_calls().into_iter().cloned().collect()
    }

    pub fn has_tool_calls(&self) -> bool {
        self.parts
            .iter()
            .any(|part| matches!(part, MessagePart::ToolCall(_)))
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

        async fn generate_chat(
            &self,
            _request: TextRequest,
        ) -> Result<crate::core::result::ChatResult, SdkError> {
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
    async fn test_run_turn_uses_tool_call_ready_input_without_deltas() {
        let model = MockModel::new(vec![
            StreamEvent::ToolCallReady {
                id: "call_ready".to_string(),
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

        let outcome = run_turn(&model, TextRequest::prompt("weather?"))
            .await
            .unwrap();

        match outcome {
            TurnOutcome::ToolsRequired { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_ready");
                assert_eq!(tool_calls[0].name, "get_weather");
                assert_eq!(tool_calls[0].input["location"], "Paris");
            }
            TurnOutcome::Completed(_) => panic!("Expected ToolsRequired"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_ready_input_overrides_partial_delta_buffer() {
        let model = MockModel::new(vec![
            StreamEvent::ToolCallStarted {
                id: "call_override".to_string(),
                name: "get_weather".to_string(),
                index: 0,
            },
            StreamEvent::ToolCallDelta {
                id: "call_override".to_string(),
                index: 0,
                input_delta: r#"{"location":"Par"#.to_string(),
            },
            StreamEvent::ToolCallReady {
                id: "call_override".to_string(),
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

        let outcome = run_turn(&model, TextRequest::prompt("weather?"))
            .await
            .unwrap();

        match outcome {
            TurnOutcome::ToolsRequired { tool_calls, .. } => {
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

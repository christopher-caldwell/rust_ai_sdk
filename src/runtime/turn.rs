use std::collections::HashMap;

use futures_util::StreamExt;
use serde_json::Value;

use crate::core::{
    error::SdkError,
    message::{Message, MessagePart, ProviderMetadata, ToolCall, ToolOutput},
    model::StreamingLanguageModel,
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
        /// Tool calls ready for application execution.
        tool_calls: Vec<ToolCall>,
        /// All parts of the assistant's turn — needed to build the continuation request.
        assistant_parts: Vec<MessagePart>,
        /// Provider-neutral finish reason for the assistant turn.
        finish_reason: FinishReason,
        /// Provider-reported token usage.
        usage: Option<Usage>,
        /// Provider response metadata.
        response: ResponseMetadata,
    },
}

/// Run one model turn, accumulate the stream into a `TurnOutcome`.
pub async fn run_turn<M: StreamingLanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TurnOutcome, SdkError> {
    let mut stream = model.stream(request).await?;
    let mut acc = TurnAccumulator::default();

    while let Some(event) = stream.next().await {
        let event = event?;
        let is_finished = matches!(event, StreamEvent::Finished { .. });
        acc.push_event(event);
        if is_finished {
            return acc.into_outcome();
        }
    }

    Err(SdkError::stream_terminated(
        Some(model.provider_name()),
        "stream ended before a terminal Finished event",
    ))
}

/// Build a continuation request by appending the assistant's turn and tool results.
pub struct ContinuationBuilder {
    request: TextRequest,
}

impl ContinuationBuilder {
    /// Start a continuation from the request that produced the tool calls.
    pub fn from_request(request: TextRequest) -> Self {
        Self { request }
    }

    /// Append the structured assistant turn containing tool calls.
    pub fn with_assistant_turn(mut self, parts: Vec<MessagePart>) -> Self {
        self.request.messages.push(Message::assistant_parts(parts));
        self
    }

    /// Append one tool result.
    pub fn with_tool_result(
        mut self,
        tool_call_id: impl Into<String>,
        output: impl Into<ToolOutput>,
    ) -> Self {
        self.request
            .messages
            .push(Message::tool_result(tool_call_id, output));
        self
    }

    /// Append several tool results in iteration order.
    pub fn with_tool_results<O>(mut self, results: impl IntoIterator<Item = (String, O)>) -> Self
    where
        O: Into<ToolOutput>,
    {
        for (id, output) in results {
            self.request.messages.push(Message::tool_result(id, output));
        }
        self
    }

    /// Finish the continuation request.
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
    provider_metadata: Option<ProviderMetadata>,
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
    /// Apply one provider-neutral event to the accumulated assistant turn.
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
                provider_metadata,
            } => {
                self.ready_tool_call(index, id, name, input, provider_metadata);
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

    /// Finish accumulation and return the reconstructed assistant turn.
    pub fn into_accumulated(mut self) -> Result<AccumulatedTurn, SdkError> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            SdkError::stream_terminated(None, "turn has no terminal Finished event")
        })?;
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
                        let call = tool_call_from_buffer(buf);
                        parts.push(MessagePart::ToolCall(call));
                    }
                }
            }
        }

        Ok(AccumulatedTurn {
            parts,
            finish_reason,
            usage,
            response,
        })
    }

    /// Convert the accumulated turn into a completed or tools-required outcome.
    pub fn into_outcome(self) -> Result<TurnOutcome, SdkError> {
        let turn = self.into_accumulated()?;
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
            provider_metadata: None,
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
            provider_metadata: None,
        });
        if !id.is_empty() {
            buffer.id = id;
        }
        buffer.arguments.push_str(&delta);
    }

    fn ready_tool_call(
        &mut self,
        index: u32,
        id: String,
        name: String,
        input: Value,
        provider_metadata: Option<ProviderMetadata>,
    ) {
        self.ensure_tool_slot(index);
        let buffer = self.tool_buffers.entry(index).or_insert(ToolCallBuffer {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
            input: None,
            provider_metadata: None,
        });
        buffer.id = id;
        buffer.name = name;
        buffer.input = Some(input);
        buffer.provider_metadata = provider_metadata;
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

fn tool_call_from_buffer(buf: ToolCallBuffer) -> ToolCall {
    if let Some(input) = buf.input {
        let mut call = ToolCall::new(buf.id, buf.name, input);
        call.provider_metadata = buf.provider_metadata;
        return call;
    }

    match serde_json::from_str(&buf.arguments) {
        Ok(input) => {
            let mut call = ToolCall::new(buf.id, buf.name, input);
            call.provider_metadata = buf.provider_metadata;
            call
        }
        Err(error) => {
            let metadata = ToolCall::metadata_with_malformed_input(
                buf.provider_metadata,
                buf.arguments.clone(),
                error.to_string(),
            );
            ToolCall::new(buf.id, buf.name, Value::Null).with_provider_metadata(metadata)
        }
    }
}

/// A completed assistant turn reconstructed from stream events.
#[derive(Debug, Clone)]
pub struct AccumulatedTurn {
    /// Reconstructed assistant text and tool-call parts in stream order.
    pub parts: Vec<MessagePart>,
    /// Provider-neutral finish reason.
    pub finish_reason: FinishReason,
    /// Provider-reported token usage.
    pub usage: Option<Usage>,
    /// Provider response metadata.
    pub response: ResponseMetadata,
}

impl AccumulatedTurn {
    /// Concatenate all text parts.
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

    /// Borrow all tool-call parts.
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

    /// Clone all tool calls for application execution.
    pub fn tool_calls_cloned(&self) -> Vec<ToolCall> {
        self.tool_calls().into_iter().cloned().collect()
    }

    /// Return whether the turn contains at least one tool call.
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
    use crate::core::{model::LanguageModel, types::ResponseMetadata};
    use futures_util::{StreamExt, stream};

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

        fn model_id(&self) -> &str {
            "mock"
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    #[async_trait::async_trait]
    impl StreamingLanguageModel for MockModel {
        async fn stream(
            &self,
            _request: TextRequest,
        ) -> Result<crate::core::stream::TextEventStream, SdkError> {
            let events = self.events.lock().unwrap().take().unwrap_or_default();
            Ok(make_stream(events))
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
    async fn test_run_turn_returns_after_finished_even_if_stream_stays_open() {
        struct NeverEndingAfterFinished;

        #[async_trait::async_trait]
        impl LanguageModel for NeverEndingAfterFinished {
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

            fn model_id(&self) -> &str {
                "never-ending"
            }

            fn provider_name(&self) -> &str {
                "test"
            }
        }

        #[async_trait::async_trait]
        impl StreamingLanguageModel for NeverEndingAfterFinished {
            async fn stream(
                &self,
                _request: TextRequest,
            ) -> Result<crate::core::stream::TextEventStream, SdkError> {
                Ok(Box::pin(
                    stream::once(async {
                        Ok(StreamEvent::Finished {
                            finish_reason: FinishReason::Stop,
                            usage: None,
                            response: meta(),
                        })
                    })
                    .chain(stream::pending()),
                ))
            }
        }

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            run_turn(&NeverEndingAfterFinished, TextRequest::prompt("hi")),
        )
        .await;

        assert!(result.is_ok(), "run_turn should stop after Finished");
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
                provider_metadata: None,
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
                provider_metadata: None,
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
    async fn test_run_turn_preserves_tool_call_provider_metadata() {
        let metadata = serde_json::json!({"provider": {"signature": "sig_1"}});
        let model = MockModel::new(vec![
            StreamEvent::ToolCallReady {
                id: "call_ready".to_string(),
                name: "get_weather".to_string(),
                index: 0,
                input: serde_json::json!({"location": "Paris"}),
                provider_metadata: Some(metadata.clone().into()),
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
                assert_eq!(
                    tool_calls[0].provider_metadata.as_ref().map(|m| m.as_raw()),
                    Some(&metadata),
                );
            }
            TurnOutcome::Completed(_) => panic!("Expected ToolsRequired"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_preserves_malformed_tool_json_from_deltas() {
        let model = MockModel::new(vec![
            StreamEvent::ToolCallStarted {
                id: "call_bad".to_string(),
                name: "get_weather".to_string(),
                index: 0,
            },
            StreamEvent::ToolCallDelta {
                id: "call_bad".to_string(),
                index: 0,
                input_delta: r#"{"location":"Paris""#.to_string(),
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
                assert_eq!(tool_calls[0].input, serde_json::Value::Null);
                assert!(tool_calls[0].has_malformed_input());
                assert_eq!(
                    tool_calls[0].malformed_input_raw(),
                    Some(r#"{"location":"Paris""#),
                );
            }
            TurnOutcome::Completed(_) => panic!("Expected ToolsRequired"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_without_finished_returns_error() {
        let model = MockModel::new(vec![StreamEvent::TextDelta("hello".to_string())]);

        let error = run_turn(&model, TextRequest::prompt("hi"))
            .await
            .unwrap_err();

        assert!(matches!(error, SdkError::StreamTerminated { .. }));
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
                provider_metadata: None,
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
            MessagePart::ToolCall(ToolCall::new("c1", "tool", serde_json::json!({}))),
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
            crate::core::message::Role::Tool
        ));
    }
}

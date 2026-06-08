use futures_core::Stream;
use serde_json::Value;
use std::pin::Pin;

use super::{
    error::SdkError,
    message::ProviderMetadata,
    types::{FinishReason, ResponseMetadata, Usage},
};

/// Provider-neutral streaming event emitted by [`LanguageModel`](crate::core::model::LanguageModel)
/// implementations.
///
/// High-level applications usually consume these through
/// [`stream_text`](crate::runtime::stream::stream_text), [`run_turn`](crate::runtime::turn::run_turn),
/// or the `message-stream` feature. Low-level adapters can inspect each event directly.
///
/// One model turn is a sequence of zero or more content/tool events followed by
/// at most one terminal [`StreamEvent::Finished`] event. Runtime consumers
/// should stop processing the current turn after `Finished`, even if a provider
/// transport keeps yielding data.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A text delta that should be appended to the current assistant text.
    ///
    /// Providers may split text at arbitrary byte-safe boundaries. Consumers
    /// append deltas in arrival order to reconstruct the assistant text part.
    /// Providers should avoid emitting empty text deltas.
    TextDelta(String),
    /// The provider finished the current turn.
    ///
    /// This event should be emitted once after all text and tool-call events
    /// for the turn. `usage` may be `None` when the provider did not report
    /// token counts. `response` should carry the best available provider
    /// response id and model name, even when those fields arrived on earlier
    /// chunks.
    Finished {
        finish_reason: FinishReason,
        usage: Option<Usage>,
        response: ResponseMetadata,
    },
    /// The provider started a streamed tool call.
    ///
    /// `index` is the stable per-turn tool-call index used to join subsequent
    /// deltas and the final ready event. `id` should be the provider id when
    /// available. If a provider does not supply one, it may be empty or
    /// synthetic; runtime adapters normalize empty ids to `tool_call_{index}`
    /// before building continuation messages.
    ToolCallStarted {
        id: String,
        name: String,
        index: u32,
    },
    /// A streamed tool-call argument delta.
    ///
    /// `input_delta` is raw provider argument text, normally a partial JSON
    /// document fragment. It is not guaranteed to be valid JSON until the
    /// matching [`StreamEvent::ToolCallReady`] event. Providers should emit the
    /// same `index` as the matching start/ready event and the best known `id`.
    ToolCallDelta {
        id: String,
        index: u32,
        input_delta: String,
    },
    /// A complete tool call is ready for runtime accumulation.
    ///
    /// `input` is the parsed JSON input when parsing succeeded. If the provider
    /// produced malformed JSON, providers must not coerce the raw text into an
    /// ordinary JSON string. Instead, use `Value::Null` and set
    /// `provider_metadata.another_ai_sdk.tool_input.state` to
    /// `"malformed_json"` with the raw input and parse error. Runtime tool
    /// execution rejects those malformed calls by default.
    ToolCallReady {
        id: String,
        name: String,
        index: u32,
        input: Value,
        provider_metadata: Option<ProviderMetadata>,
    },
}

/// Boxed provider-neutral stream type used by model implementations.
pub type TextEventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, SdkError>> + Send>>;
/// Backwards-compatible alias for [`TextEventStream`].
pub type EventStream = TextEventStream;

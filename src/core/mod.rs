/// Provider-neutral SDK errors.
pub mod error;
/// Provider-neutral messages and tool-call parts.
pub mod message;
/// The shared language-model trait.
pub mod model;
#[cfg(any(test, feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(crate) mod provider_policy;
/// Provider-neutral generation requests and validation.
pub mod request;
/// Structured generation results.
pub mod result;
/// Provider-neutral streaming events.
pub mod stream;
/// Tool definitions and selection policy.
pub mod tool;
/// Usage, finish-reason, and response metadata types.
pub mod types;

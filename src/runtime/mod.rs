/// Convenience helpers for non-streaming generation.
pub mod generate;
#[cfg(feature = "message-stream")]
/// Vercel AI SDK UI-message stream adapter.
pub mod message_stream;
/// Convenience helper for provider-neutral streaming.
pub mod stream;
/// Application-owned tool registry.
pub mod tools;
/// Stream accumulation and tool-continuation helpers.
pub mod turn;

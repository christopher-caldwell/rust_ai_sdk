#[cfg(feature = "anthropic")]
/// Anthropic model wrapper and model identifiers.
pub mod anthropic;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod config;
#[cfg(feature = "gemini")]
/// Gemini model wrapper and model identifiers.
pub mod gemini;
#[cfg(feature = "openai")]
/// OpenAI model wrapper and model identifiers.
pub mod openai;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod transport;

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
/// Shared HTTP transport configuration for built-in providers.
pub use config::ProviderHttpConfig;

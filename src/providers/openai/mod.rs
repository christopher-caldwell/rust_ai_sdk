mod model;
/// OpenAI model identifiers.
pub mod models;

pub(crate) mod client;
pub(crate) mod error;
pub(crate) mod types;

pub use model::OpenAiChatModel;
pub use models::OpenAiModel;

use async_trait::async_trait;

use super::{
    error::SdkError,
    message::MessagePart,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

/// Provider-neutral model interface used by runtime helpers.
///
/// Provider wrappers implement this trait so application code can call
/// [`generate`](LanguageModel::generate) or
/// [`generate_chat`](LanguageModel::generate_chat) without depending on provider wire formats.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Generate a simple text result from a request.
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError>;

    /// Generate a structured chat result from a request.
    ///
    /// Providers may override this when they can return structured assistant parts
    /// directly. The default wraps [`generate`](LanguageModel::generate) text into a
    /// single assistant text part.
    async fn generate_chat(&self, request: TextRequest) -> Result<ChatResult, SdkError> {
        let result = self.generate(request).await?;
        Ok(ChatResult {
            parts: vec![MessagePart::Text(result.text)],
            finish_reason: result.finish_reason,
            usage: result.usage,
            response: result.response,
        })
    }

    /// Return the provider model identifier configured for this wrapper.
    fn model_id(&self) -> &str;

    /// Return the provider name, such as `openai`, `anthropic`, or `gemini`.
    fn provider_name(&self) -> &str;
}

/// Streaming capability implemented only by models that can produce events.
///
/// Separating this from [`LanguageModel`] makes unsupported streaming a
/// compile-time error instead of a runtime feature error.
#[async_trait]
pub trait StreamingLanguageModel: LanguageModel {
    /// Stream provider-neutral text and tool-call events for a request.
    async fn stream(
        &self,
        request: TextRequest,
    ) -> Result<super::stream::TextEventStream, SdkError>;
}

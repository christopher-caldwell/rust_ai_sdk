use async_trait::async_trait;

use super::{
    error::SdkError,
    message::MessagePart,
    request::TextRequest,
    result::{ChatResult, TextResult},
    stream::TextEventStream,
};

#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError>;

    async fn generate_chat(&self, request: TextRequest) -> Result<ChatResult, SdkError> {
        let result = self.generate(request).await?;
        Ok(ChatResult {
            parts: vec![MessagePart::Text(result.text)],
            finish_reason: result.finish_reason,
            usage: result.usage,
            response: result.response,
        })
    }

    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError>;

    fn model_id(&self) -> &str;

    fn provider_name(&self) -> &str;
}

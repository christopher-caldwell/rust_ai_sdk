use async_trait::async_trait;

use super::{error::SdkError, request::TextRequest, result::TextResult, stream::TextEventStream};

#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError>;

    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError>;

    fn model_id(&self) -> &str;

    fn provider_name(&self) -> &str;
}

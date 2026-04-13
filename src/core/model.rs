use async_trait::async_trait;

use super::{error::SdkError, request::TextRequest, result::TextResult};

#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError>;

    fn model_id(&self) -> &str;

    fn provider_name(&self) -> &str;
}

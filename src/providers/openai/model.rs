use async_trait::async_trait;

use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::TextResult,
};

use super::client::OpenAiClient;

pub struct OpenAiChatModel {
    client: OpenAiClient,
    model: String,
}

impl OpenAiChatModel {
    pub fn new(api_key: String, model: impl Into<String>) -> Self {
        Self {
            client: OpenAiClient::new(api_key),
            model: model.into(),
        }
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatModel {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError> {
        self.client.generate(&self.model, request).await
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

use async_trait::async_trait;

use crate::core::{
    error::SdkError, model::LanguageModel, request::TextRequest, result::TextResult,
    stream::TextEventStream,
};

use super::client::AnthropicClient;

pub struct AnthropicChatModel {
    client: AnthropicClient,
    model: String,
}

impl AnthropicChatModel {
    pub fn new(api_key: String, model: impl Into<String>) -> Self {
        Self {
            client: AnthropicClient::new(api_key),
            model: model.into(),
        }
    }
}

#[async_trait]
impl LanguageModel for AnthropicChatModel {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError> {
        self.client.generate(&self.model, &request).await
    }

    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError> {
        self.client.stream(&self.model, &request).await
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

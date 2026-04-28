use async_trait::async_trait;

use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::{ChatResult, TextResult},
    stream::TextEventStream,
};

use super::client::OpenAiClient;

#[derive(Clone)]
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

    pub async fn generate_chat(&self, request: TextRequest) -> Result<ChatResult, SdkError> {
        self.client.generate_chat(&self.model, &request).await
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatModel {
    async fn generate(&self, request: TextRequest) -> Result<TextResult, SdkError> {
        self.client.generate(&self.model, &request).await
    }

    async fn generate_chat(&self, request: TextRequest) -> Result<ChatResult, SdkError> {
        self.client.generate_chat(&self.model, &request).await
    }

    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError> {
        self.client.stream(&self.model, &request).await
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

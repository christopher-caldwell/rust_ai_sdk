use async_trait::async_trait;

use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::{ChatResult, TextResult},
};
#[cfg(feature = "streaming")]
use crate::core::{model::StreamingLanguageModel, stream::TextEventStream};

use super::super::ProviderHttpConfig;
use super::client::OpenAiClient;

#[derive(Clone)]
/// OpenAI implementation of the provider-neutral language-model interface.
pub struct OpenAiChatModel {
    client: OpenAiClient,
    model: String,
}

impl OpenAiChatModel {
    /// Construct an OpenAI model with the default provider endpoint and HTTP client.
    pub fn new(api_key: String, model: impl Into<String>) -> Result<Self, SdkError> {
        Ok(Self {
            client: OpenAiClient::new(api_key)?,
            model: model.into(),
        })
    }

    /// Construct an OpenAI model with application-defined HTTP configuration.
    pub fn with_config(
        api_key: String,
        model: impl Into<String>,
        config: ProviderHttpConfig,
    ) -> Result<Self, SdkError> {
        Ok(Self {
            client: OpenAiClient::with_config(api_key, config)?,
            model: model.into(),
        })
    }

    /// Generate a structured assistant result that can contain tool calls.
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

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

#[cfg(feature = "streaming")]
#[async_trait]
impl StreamingLanguageModel for OpenAiChatModel {
    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError> {
        self.client.stream(&self.model, &request).await
    }
}

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
use super::client::GeminiClient;

#[derive(Clone)]
/// Gemini implementation of the provider-neutral language-model interface.
pub struct GeminiChatModel {
    client: GeminiClient,
    model: String,
}

impl GeminiChatModel {
    /// Construct a Gemini model with the default provider endpoint and HTTP client.
    pub fn new(api_key: String, model: impl Into<String>) -> Result<Self, SdkError> {
        Ok(Self {
            client: GeminiClient::new(api_key)?,
            model: model.into(),
        })
    }

    /// Construct a Gemini model with application-defined HTTP configuration.
    pub fn with_config(
        api_key: String,
        model: impl Into<String>,
        config: ProviderHttpConfig,
    ) -> Result<Self, SdkError> {
        Ok(Self {
            client: GeminiClient::with_config(api_key, config)?,
            model: model.into(),
        })
    }

    /// Generate a structured assistant result that can contain tool calls.
    pub async fn generate_chat(&self, request: TextRequest) -> Result<ChatResult, SdkError> {
        self.client.generate_chat(&self.model, &request).await
    }
}

#[async_trait]
impl LanguageModel for GeminiChatModel {
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
        "gemini"
    }
}

#[cfg(feature = "streaming")]
#[async_trait]
impl StreamingLanguageModel for GeminiChatModel {
    async fn stream(&self, request: TextRequest) -> Result<TextEventStream, SdkError> {
        self.client.stream(&self.model, &request).await
    }
}

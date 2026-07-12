use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

/// Generate a text-only result through a provider-neutral model.
pub async fn generate_text<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TextResult, SdkError> {
    model.generate(request).await
}

/// Generate a structured result that may contain tool calls.
pub async fn generate_chat<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<ChatResult, SdkError> {
    model.generate_chat(request).await
}

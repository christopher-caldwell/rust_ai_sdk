use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::{ChatResult, TextResult},
};

pub async fn generate_text<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TextResult, SdkError> {
    model.generate(request).await
}

pub async fn generate_chat<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<ChatResult, SdkError> {
    model.generate_chat(request).await
}

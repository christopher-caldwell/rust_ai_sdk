use crate::core::{
    error::SdkError,
    model::LanguageModel,
    request::TextRequest,
    result::TextResult,
};

pub async fn generate_text<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TextResult, SdkError> {
    model.generate(request).await
}

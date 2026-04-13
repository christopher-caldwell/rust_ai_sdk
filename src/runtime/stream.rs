use crate::core::{
    error::SdkError, model::LanguageModel, request::TextRequest, stream::TextEventStream,
};

pub async fn stream_text<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TextEventStream, SdkError> {
    model.stream(request).await
}

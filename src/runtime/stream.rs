use crate::core::{
    error::SdkError, model::StreamingLanguageModel, request::TextRequest, stream::TextEventStream,
};

/// Start a provider-neutral stream of text, tool-call, and terminal events.
pub async fn stream_text<M: StreamingLanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
) -> Result<TextEventStream, SdkError> {
    model.stream(request).await
}

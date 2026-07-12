use crate::core::error::{SdkError, TransportErrorKind};
use futures_util::StreamExt;

pub(super) fn reqwest_error(provider: &str, error: reqwest::Error) -> SdkError {
    let kind = if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connect
    } else if error.is_request() {
        TransportErrorKind::Request
    } else if error.is_body() {
        TransportErrorKind::Body
    } else if error.is_decode() {
        TransportErrorKind::Decode
    } else {
        TransportErrorKind::Other
    };

    SdkError::provider_transport(
        provider,
        error.status().map(|status| status.as_u16()),
        kind,
        error.to_string(),
    )
}

pub(super) fn truncate_body(body: &str, max_bytes: usize) -> String {
    if body.len() <= max_bytes {
        return body.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &body[..end])
}

pub(super) async fn read_bounded_body(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, reqwest::Error> {
    let mut body = Vec::with_capacity(max_bytes.min(4096));
    let mut chunks = response.bytes_stream();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk?;
        let remaining = max_bytes.saturating_sub(body.len());
        if remaining == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }

    Ok(body)
}

use std::{error::Error, fmt, sync::Arc};

/// Shared handle to the underlying error that caused an [`SdkError`].
///
/// It is reference-counted so that [`SdkError`] stays [`Clone`].
pub type ErrorSource = Arc<dyn Error + Send + Sync + 'static>;

/// Stable category for HTTP transport failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    /// A request exceeded its configured timeout.
    Timeout,
    /// A connection could not be established.
    Connect,
    /// A request could not be constructed or sent.
    Request,
    /// A response body could not be read.
    Body,
    /// A response could not be decoded.
    Decode,
    /// Another transport failure.
    Other,
}

/// Crate-wide error type returned by provider wrappers and runtime helpers.
///
/// The variants are intentionally provider-neutral. Provider translators map
/// HTTP, API, serialization, validation, and unexpected failures into this type
/// so application code can handle errors consistently across providers.
///
/// When a failure came from an underlying error (for example a `reqwest` or
/// `serde_json` error), it is kept and returned by [`Error::source`], so the
/// root cause (DNS, TLS, proxy, refused connection, malformed JSON) can be
/// walked through the source chain. [`fmt::Display`] does not print the
/// source. Equality ignores the source.
///
/// [`fmt::Debug`] is not redacted: it prints every field, including any
/// provider response `body` and the underlying source error.
#[derive(Debug, Clone)]
pub enum SdkError {
    /// An HTTP transport or response failure.
    Http {
        /// Provider that produced the failure, when known.
        provider: Option<String>,
        /// HTTP status code, when a response was received.
        status: Option<u16>,
        /// Human-readable failure summary.
        message: String,
        /// Bounded response-body excerpt for diagnostics.
        body: Option<String>,
        /// Stable transport category, when this was a client-side failure.
        transport_kind: Option<TransportErrorKind>,
        /// Underlying error, when one caused this failure.
        source: Option<ErrorSource>,
    },

    /// A structured error response returned by a provider API.
    Api {
        /// Provider that returned the error, when known.
        provider: Option<String>,
        /// HTTP status code, when available.
        status: Option<u16>,
        /// Provider-specific error code.
        code: Option<String>,
        /// Provider-specific error type.
        error_type: Option<String>,
        /// Human-readable provider message.
        message: String,
        /// Bounded response-body excerpt for diagnostics.
        body: Option<String>,
    },

    /// A provider payload could not be serialized or deserialized.
    Serialization {
        /// Provider associated with the payload, when known.
        provider: Option<String>,
        /// Serialization failure summary.
        message: String,
        /// Underlying error, when one caused this failure.
        source: Option<ErrorSource>,
    },

    /// A stream ended without the provider's terminal event.
    StreamTerminated {
        /// Provider whose stream ended, when known.
        provider: Option<String>,
        /// Description of the missing terminal condition.
        message: String,
    },

    /// A request or application input violated an SDK invariant.
    Validation(String),

    /// A failure that does not fit a more specific category.
    Unknown(String),
}

impl SdkError {
    /// Build an HTTP error without provider metadata.
    pub fn http(message: impl Into<String>) -> Self {
        Self::Http {
            provider: None,
            status: None,
            message: message.into(),
            body: None,
            transport_kind: None,
            source: None,
        }
    }

    /// Build an HTTP error with provider, status, and bounded body context.
    pub fn provider_http(
        provider: impl Into<String>,
        status: Option<u16>,
        message: impl Into<String>,
        body: Option<String>,
    ) -> Self {
        Self::Http {
            provider: Some(provider.into()),
            status,
            message: message.into(),
            body,
            transport_kind: None,
            source: None,
        }
    }

    /// Build a categorized provider transport error.
    pub fn provider_transport(
        provider: impl Into<String>,
        status: Option<u16>,
        kind: TransportErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self::Http {
            provider: Some(provider.into()),
            status,
            message: message.into(),
            body: None,
            transport_kind: Some(kind),
            source: None,
        }
    }

    /// Build an API error without provider metadata.
    pub fn api(message: impl Into<String>) -> Self {
        Self::Api {
            provider: None,
            status: None,
            code: None,
            error_type: None,
            message: message.into(),
            body: None,
        }
    }

    /// Build a structured provider API error.
    pub fn provider_api(
        provider: impl Into<String>,
        status: Option<u16>,
        code: Option<String>,
        error_type: Option<String>,
        message: impl Into<String>,
        body: Option<String>,
    ) -> Self {
        Self::Api {
            provider: Some(provider.into()),
            status,
            code,
            error_type,
            message: message.into(),
            body,
        }
    }

    /// Build a serialization error with optional provider context.
    pub fn serialization(provider: Option<&str>, message: impl Into<String>) -> Self {
        Self::Serialization {
            provider: provider.map(ToString::to_string),
            message: message.into(),
            source: None,
        }
    }

    /// Attach the underlying error that caused an HTTP or serialization
    /// failure. Other variants carry no source and are returned unchanged.
    #[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
    pub(crate) fn with_source(mut self, error: impl Error + Send + Sync + 'static) -> Self {
        match &mut self {
            Self::Http { source, .. } | Self::Serialization { source, .. } => {
                *source = Some(Arc::new(error));
            }
            Self::Api { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => {}
        }
        self
    }

    /// Build an error for a stream that ended before a terminal event.
    pub fn stream_terminated(provider: Option<&str>, message: impl Into<String>) -> Self {
        Self::StreamTerminated {
            provider: provider.map(ToString::to_string),
            message: message.into(),
        }
    }

    /// Return the provider associated with this error, when available.
    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::Http { provider, .. }
            | Self::Api { provider, .. }
            | Self::Serialization { provider, .. }
            | Self::StreamTerminated { provider, .. } => provider.as_deref(),
            Self::Validation(_) | Self::Unknown(_) => None,
        }
    }

    /// Return the HTTP status code associated with this error, when available.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } | Self::Api { status, .. } => *status,
            Self::Serialization { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    /// Return the provider-specific error code, when available.
    pub fn provider_code(&self) -> Option<&str> {
        match self {
            Self::Api { code, .. } => code.as_deref(),
            Self::Http { .. }
            | Self::Serialization { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    /// Return the provider-specific error type, when available.
    pub fn provider_error_type(&self) -> Option<&str> {
        match self {
            Self::Api { error_type, .. } => error_type.as_deref(),
            Self::Http { .. }
            | Self::Serialization { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    /// Return the bounded provider response body, when one was captured.
    pub fn body_snippet(&self) -> Option<&str> {
        match self {
            Self::Http { body, .. } | Self::Api { body, .. } => body.as_deref(),
            Self::Serialization { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    /// Return a stable transport category, when available.
    pub fn transport_kind(&self) -> Option<TransportErrorKind> {
        match self {
            Self::Http { transport_kind, .. } => *transport_kind,
            _ => None,
        }
    }

    /// Return whether this is a categorized timeout failure.
    pub fn is_timeout(&self) -> bool {
        self.transport_kind() == Some(TransportErrorKind::Timeout)
    }

    /// Return whether this is a categorized connection failure.
    pub fn is_connect(&self) -> bool {
        self.transport_kind() == Some(TransportErrorKind::Connect)
    }

    /// Return the human-readable message without category formatting.
    pub fn message(&self) -> &str {
        match self {
            Self::Http { message, .. }
            | Self::Api { message, .. }
            | Self::Serialization { message, .. }
            | Self::StreamTerminated { message, .. }
            | Self::Validation(message)
            | Self::Unknown(message) => message,
        }
    }
}

impl fmt::Display for SdkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http {
                provider,
                status,
                message,
                ..
            } => write_error(
                formatter,
                "HTTP error",
                provider.as_deref(),
                *status,
                message,
            ),
            Self::Api {
                provider,
                status,
                code,
                error_type,
                message,
                ..
            } => {
                write_error(
                    formatter,
                    "API error",
                    provider.as_deref(),
                    *status,
                    message,
                )?;
                write_provider_detail(formatter, "code", code.as_deref())?;
                write_provider_detail(formatter, "type", error_type.as_deref())
            }
            Self::Serialization {
                provider, message, ..
            } => write_error(
                formatter,
                "Serialization error",
                provider.as_deref(),
                None,
                message,
            ),
            Self::StreamTerminated { provider, message } => write_error(
                formatter,
                "Stream terminated",
                provider.as_deref(),
                None,
                message,
            ),
            Self::Validation(message) => write!(formatter, "Validation error: {message}"),
            Self::Unknown(message) => write!(formatter, "Unknown error: {message}"),
        }
    }
}

impl Error for SdkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http { source, .. } | Self::Serialization { source, .. } => source
                .as_deref()
                .map(|source| source as &(dyn Error + 'static)),
            Self::Api { .. }
            | Self::StreamTerminated { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }
}

/// Compares every field except the underlying error source, which has no
/// meaningful equality.
///
/// The match on `self` is exhaustive and every pattern names all fields, so
/// adding a variant or a field fails to compile until it is handled here.
impl PartialEq for SdkError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Http {
                provider,
                status,
                message,
                body,
                transport_kind,
                source: _,
            } => {
                let Self::Http {
                    provider: other_provider,
                    status: other_status,
                    message: other_message,
                    body: other_body,
                    transport_kind: other_transport_kind,
                    source: _,
                } = other
                else {
                    return false;
                };
                provider == other_provider
                    && status == other_status
                    && message == other_message
                    && body == other_body
                    && transport_kind == other_transport_kind
            }
            Self::Api {
                provider,
                status,
                code,
                error_type,
                message,
                body,
            } => {
                let Self::Api {
                    provider: other_provider,
                    status: other_status,
                    code: other_code,
                    error_type: other_error_type,
                    message: other_message,
                    body: other_body,
                } = other
                else {
                    return false;
                };
                provider == other_provider
                    && status == other_status
                    && code == other_code
                    && error_type == other_error_type
                    && message == other_message
                    && body == other_body
            }
            Self::Serialization {
                provider,
                message,
                source: _,
            } => {
                let Self::Serialization {
                    provider: other_provider,
                    message: other_message,
                    source: _,
                } = other
                else {
                    return false;
                };
                provider == other_provider && message == other_message
            }
            Self::StreamTerminated { provider, message } => {
                let Self::StreamTerminated {
                    provider: other_provider,
                    message: other_message,
                } = other
                else {
                    return false;
                };
                provider == other_provider && message == other_message
            }
            Self::Validation(message) => {
                let Self::Validation(other_message) = other else {
                    return false;
                };
                message == other_message
            }
            Self::Unknown(message) => {
                let Self::Unknown(other_message) = other else {
                    return false;
                };
                message == other_message
            }
        }
    }
}

impl Eq for SdkError {}

fn write_error(
    formatter: &mut fmt::Formatter<'_>,
    label: &str,
    provider: Option<&str>,
    status: Option<u16>,
    message: &str,
) -> fmt::Result {
    write!(formatter, "{label}")?;
    if let Some(provider) = provider {
        write!(formatter, " ({provider}")?;
        if let Some(status) = status {
            write!(formatter, ", HTTP {status}")?;
        }
        write!(formatter, ")")?;
    } else if let Some(status) = status {
        write!(formatter, " (HTTP {status})")?;
    }
    write!(formatter, ": {message}")
}

fn write_provider_detail(
    formatter: &mut fmt::Formatter<'_>,
    label: &str,
    value: Option<&str>,
) -> fmt::Result {
    if let Some(value) = value {
        write!(formatter, " [{label}: {value}]")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_http_error_exposes_provider_and_status() {
        let error = SdkError::provider_http(
            "OpenAI",
            Some(429),
            "rate limited",
            Some("too many requests".to_string()),
        );

        assert_eq!(error.provider(), Some("OpenAI"));
        assert_eq!(error.status_code(), Some(429));
        assert_eq!(error.body_snippet(), Some("too many requests"));
        assert_eq!(
            error.to_string(),
            "HTTP error (OpenAI, HTTP 429): rate limited"
        );
    }

    #[test]
    fn transport_errors_expose_stable_categories() {
        let error = SdkError::provider_transport(
            "test",
            None,
            TransportErrorKind::Timeout,
            "request timed out",
        );

        assert!(error.is_timeout());
        assert!(!error.is_connect());
        assert_eq!(error.transport_kind(), Some(TransportErrorKind::Timeout));
    }

    #[test]
    fn structured_api_error_exposes_provider_code_and_type() {
        let error = SdkError::provider_api(
            "Anthropic",
            Some(401),
            Some("invalid_api_key".to_string()),
            Some("authentication_error".to_string()),
            "Invalid API key.",
            None,
        );

        assert_eq!(error.provider(), Some("Anthropic"));
        assert_eq!(error.status_code(), Some(401));
        assert_eq!(error.provider_code(), Some("invalid_api_key"));
        assert_eq!(error.provider_error_type(), Some("authentication_error"));
        assert_eq!(
            error.to_string(),
            "API error (Anthropic, HTTP 401): Invalid API key. [code: invalid_api_key] [type: authentication_error]"
        );
    }

    // Regression (audit F5): the underlying error is reachable through
    // `Error::source()`, is not printed by Display, and is ignored by equality.
    #[test]
    fn underlying_error_is_exposed_as_source() {
        use std::io;

        let error = SdkError::Http {
            provider: Some("OpenAI".to_string()),
            status: None,
            message: "error sending request".to_string(),
            body: None,
            transport_kind: Some(TransportErrorKind::Connect),
            source: Some(Arc::new(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "connection refused",
            ))),
        };

        let source = error.source().expect("source is kept");
        assert_eq!(
            source.downcast_ref::<io::Error>().map(io::Error::kind),
            Some(io::ErrorKind::ConnectionRefused)
        );
        assert!(error.clone().source().is_some());
        assert_eq!(
            error.to_string(),
            "HTTP error (OpenAI): error sending request"
        );

        let without_source = SdkError::provider_transport(
            "OpenAI",
            None,
            TransportErrorKind::Connect,
            "error sending request",
        );
        assert!(without_source.source().is_none());
        assert_eq!(error, without_source);

        let serialization = SdkError::Serialization {
            provider: Some("Gemini".to_string()),
            message: "expected value".to_string(),
            source: Some(Arc::new(io::Error::other("expected value"))),
        };
        assert!(serialization.source().is_some());
        assert_eq!(
            serialization,
            SdkError::serialization(Some("Gemini"), "expected value")
        );

        // Equality still distinguishes variants with the same payload.
        assert_ne!(
            SdkError::Validation("expected value".to_string()),
            SdkError::Unknown("expected value".to_string())
        );
        assert_ne!(
            serialization,
            SdkError::Validation("expected value".to_string())
        );
    }

    #[test]
    fn serialization_error_keeps_provider_context() {
        let error = SdkError::serialization(Some("Gemini"), "missing field");

        assert_eq!(error.provider(), Some("Gemini"));
        assert_eq!(
            error.to_string(),
            "Serialization error (Gemini): missing field"
        );
    }
}

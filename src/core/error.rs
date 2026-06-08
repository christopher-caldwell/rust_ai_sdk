use std::{error::Error, fmt};

/// Crate-wide error type returned by provider wrappers and runtime helpers.
///
/// The variants are intentionally provider-neutral. Provider translators map
/// HTTP, API, serialization, validation, and unexpected failures into this type
/// so application code can handle errors consistently across providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdkError {
    Http {
        provider: Option<String>,
        status: Option<u16>,
        message: String,
        body: Option<String>,
    },

    Api {
        provider: Option<String>,
        status: Option<u16>,
        code: Option<String>,
        error_type: Option<String>,
        message: String,
        body: Option<String>,
    },

    Serialization {
        provider: Option<String>,
        message: String,
    },

    Validation(String),

    Unknown(String),
}

impl SdkError {
    pub fn http(message: impl Into<String>) -> Self {
        Self::Http {
            provider: None,
            status: None,
            message: message.into(),
            body: None,
        }
    }

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
        }
    }

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

    pub fn serialization(provider: Option<&str>, message: impl Into<String>) -> Self {
        Self::Serialization {
            provider: provider.map(ToString::to_string),
            message: message.into(),
        }
    }

    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::Http { provider, .. }
            | Self::Api { provider, .. }
            | Self::Serialization { provider, .. } => provider.as_deref(),
            Self::Validation(_) | Self::Unknown(_) => None,
        }
    }

    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } | Self::Api { status, .. } => *status,
            Self::Serialization { .. } | Self::Validation(_) | Self::Unknown(_) => None,
        }
    }

    pub fn provider_code(&self) -> Option<&str> {
        match self {
            Self::Api { code, .. } => code.as_deref(),
            Self::Http { .. }
            | Self::Serialization { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    pub fn provider_error_type(&self) -> Option<&str> {
        match self {
            Self::Api { error_type, .. } => error_type.as_deref(),
            Self::Http { .. }
            | Self::Serialization { .. }
            | Self::Validation(_)
            | Self::Unknown(_) => None,
        }
    }

    pub fn body_snippet(&self) -> Option<&str> {
        match self {
            Self::Http { body, .. } | Self::Api { body, .. } => body.as_deref(),
            Self::Serialization { .. } | Self::Validation(_) | Self::Unknown(_) => None,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Http { message, .. }
            | Self::Api { message, .. }
            | Self::Serialization { message, .. }
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
            Self::Serialization { provider, message } => write_error(
                formatter,
                "Serialization error",
                provider.as_deref(),
                None,
                message,
            ),
            Self::Validation(message) => write!(formatter, "Validation error: {message}"),
            Self::Unknown(message) => write!(formatter, "Unknown error: {message}"),
        }
    }
}

impl Error for SdkError {}

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

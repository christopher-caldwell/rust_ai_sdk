use crate::core::error::SdkError;

/// Shared HTTP transport configuration for built-in provider models.
///
/// A caller-supplied [`reqwest::Client`] can centralize timeouts, proxies,
/// certificate policy, connection pooling, and middleware. A custom base URL
/// is useful for compatible gateways and integration tests.
#[derive(Clone, Default)]
pub struct ProviderHttpConfig {
    client: Option<reqwest::Client>,
    base_url: Option<String>,
}

impl ProviderHttpConfig {
    /// Create configuration that uses the SDK's default HTTP client and endpoint.
    ///
    /// The default client uses a 10-second connect timeout and a 120-second
    /// overall request timeout. Supply a client to choose different policies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use an application-configured HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Use a provider-compatible API base URL.
    ///
    /// The URL is validated when the provider model is constructed.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub(crate) fn resolve(
        self,
        provider: &str,
        default_base_url: &str,
    ) -> Result<(reqwest::Client, String), SdkError> {
        let base_url = self
            .base_url
            .unwrap_or_else(|| default_base_url.to_string());
        let parsed = reqwest::Url::parse(&base_url).map_err(|error| {
            SdkError::Validation(format!("invalid {provider} base URL '{base_url}': {error}"))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(SdkError::Validation(format!(
                "invalid {provider} base URL '{base_url}': expected an absolute HTTP(S) URL",
            )));
        }

        let client = match self.client {
            Some(client) => client,
            None => reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .map_err(|error| super::transport::reqwest_error(provider, error))?,
        };

        Ok((client, base_url.trim_end_matches('/').to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_and_normalizes_custom_base_url() {
        let (_, base_url) = ProviderHttpConfig::new()
            .with_base_url("http://localhost:8080/v1/")
            .resolve("test", "https://example.com")
            .unwrap();

        assert_eq!(base_url, "http://localhost:8080/v1");
    }

    #[test]
    fn rejects_non_http_base_url() {
        let error = ProviderHttpConfig::new()
            .with_base_url("file:///tmp/provider")
            .resolve("test", "https://example.com")
            .unwrap_err();

        assert!(matches!(error, SdkError::Validation(_)));
    }
}

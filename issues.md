# Library Quality Issues

This file enumerates the main quality gaps found during a review of the crate
against common expectations from mature open source Rust libraries such as
`serde`, `tokio`, `reqwest`, `clap`, `axum`, and `sqlx`.

## 1. Error handling is too string-oriented

`SdkError` currently stores most failures as plain strings:

- `Http(String)`
- `Api(String)`
- `Serialization(String)`
- `Unknown(String)`

This is a problem because callers cannot reliably inspect failures by status
code, provider error type, provider error code, retryability, or original source
error. Mature Rust libraries usually preserve structured error data so
applications can decide whether to retry, display a user-facing message, log a
provider-specific diagnostic, or branch on a specific failure mode without
parsing text.

## 2. Provider clients do not expose enough configuration

The OpenAI, Anthropic, and Gemini clients construct their own `reqwest::Client`
internally and expose only basic constructors that accept an API key and model.
Base URL overrides exist only as crate-private test helpers.

This is a problem because production users often need to configure timeouts,
proxies, custom headers, user agents, retries, alternate endpoints, and shared
HTTP clients. Without a public builder or configurable client layer, users may
have to fork the crate or wrap it awkwardly to meet normal deployment
requirements.

## 3. Package contents include local analysis artifacts

`cargo package --list` currently includes `.understand-anything/*` and
`Cargo.toml.orig`. The package exclude list only filters `examples/**` and
`target/**`.

This is a problem because crates.io packages should contain only intentional
source, metadata, docs, and build inputs. Shipping local analysis artifacts or
temporary files makes the published crate noisier, increases package size, and
signals weaker release hygiene to users evaluating the crate.

## 4. Project release hygiene is incomplete

The repository has a publish script and useful metadata in `Cargo.toml`, but no
CI configuration, changelog, or license file was found during the review.

This is a problem because mature open source users look for repeatable release
signals: automated checks, visible release history, and clearly available
license terms. The package has `license = "MIT"`, which is valid metadata, but
a checked-in license file still makes the legal terms easier to inspect in the
repository and source package.

## 5. Rustdoc coverage is light for the public API

The crate builds documentation successfully, but many public types and methods
have minimal or no rustdoc examples. The README covers common workflows, while
the API docs themselves are thinner.

This is a problem because Rust users frequently discover and evaluate crates
through docs.rs. Mature crates usually make the generated API docs useful on
their own, with examples, invariants, error behavior, and feature-gated API
notes close to the types and functions being used.

## 6. API stability is still early-stage

The crate is currently versioned as `0.0.3`, which accurately communicates that
the API is young.

This is a problem for adopters who want low-risk dependencies. A `0.0.x` crate
can still be useful, but users will reasonably assume that public APIs,
provider behavior, error types, and feature flags may change. That makes it
harder to adopt in production without pinning versions tightly.

## 7. Runtime and provider behavior need broader integration coverage

The crate has a strong unit test baseline, including mocked provider HTTP tests
and streaming/tool-call tests. However, the review did not find CI-backed live
provider smoke tests or documented compatibility checks against real provider
behavior.

This is a problem because AI provider APIs evolve quickly, especially streaming
and tool-calling protocols. Mocked tests protect internal translation logic, but
they cannot catch all drift in real response shapes, SSE behavior, model
capabilities, or provider-specific edge cases.


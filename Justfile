default: check

# Check if the project compiles
check:
    cargo check --all-targets

# Build the project
build:
    cargo build

# Run formatting checks
fmt:
    cargo fmt --all

# Run clippy for linting
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
test:
    cargo test
    cargo test --all-features

# Check Rust examples that are intentionally outside the published package
check-examples:
    cargo check --manifest-path examples/standalone/Cargo.toml
    cargo check --manifest-path examples/chatbot/server/Cargo.toml
    cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml

# Check important crate feature combinations
check-features:
    cargo check --no-default-features
    cargo check --no-default-features --features openai
    cargo check --no-default-features --features anthropic
    cargo check --no-default-features --features gemini
    cargo check --no-default-features --features providers-all
    cargo check --no-default-features --features providers-all,streaming
    cargo check --no-default-features --features message-stream
    cargo check --all-features

# Build the chatbot web example
check-chatbot-web:
    cd examples/chatbot/web && npm run build

# Build public rustdoc without dependencies
doc:
    cargo doc --all-features --no-deps

# Validate the public API surface and examples from the repository root
check-public: test check-examples doc

# Format and lint the codebase
lint: fmt clippy

# Publish the package
publish-dry:
    sh scripts/publish-crate.sh

publish:
    sh scripts/publish-crate.sh --publish

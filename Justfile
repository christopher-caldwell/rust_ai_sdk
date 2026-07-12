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

fmt-check:
    cargo fmt --all -- --check

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
    cargo test --no-default-features
    cargo test --no-default-features --features openai
    cargo test --no-default-features --features anthropic
    cargo test --no-default-features --features gemini
    cargo test --no-default-features --features providers-all
    cargo test --no-default-features --features providers-all,streaming
    cargo test --no-default-features --features message-stream
    cargo test --all-features
    RUSTDOCFLAGS="-D warnings" cargo doc --no-default-features --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Build the chatbot web example
check-chatbot-web:
    cd examples/chatbot/web && npm run build

# Build public rustdoc without dependencies
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Verify crate contents and release metadata without requiring a clean worktree
package-check:
    ./scripts/publish-crate.sh --check-only --allow-dirty

# Validate the public API surface and examples from the repository root
check-public: test check-examples doc

# Format and lint the codebase
lint: fmt-check clippy

# Prepare, validate, commit, publish, and tag a release from a clean branch
release:
    ./scripts/publish-crate.sh --release

# Prepare a local release by generating version and changelog updates
release-prepare:
    ./scripts/publish-crate.sh

# Run publish validation without generating release changes
publish-dry:
    ./scripts/publish-crate.sh --check-only

# Publish the package from the committed release changes
publish:
    ./scripts/publish-crate.sh --publish

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

# Format and lint the codebase
lint: fmt clippy

# Run the generate example
example-generate:
    set -a && . examples/.env && set +a && cargo run --example generate

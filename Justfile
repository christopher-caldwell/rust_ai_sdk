set dotenv-load
set dotenv-path := "examples/.env"

# Run the generate example
example-generate:
    cargo run --example generate

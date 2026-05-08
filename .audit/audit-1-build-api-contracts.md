# Audit 1: build, features, and public API contracts

## Commands/evidence collected

- `git rev-parse HEAD | tee .audit/head.txt`
- `git status --short | tee .audit/git-status-before.txt`
- `cargo metadata --format-version=1 --no-deps > .audit/cargo-metadata.json`
- `cargo tree -e features > .audit/cargo-tree-features.txt`
- `cargo fmt --all -- --check 2>&1 | tee .audit/fmt.txt`
- `cargo check --all-targets 2>&1 | tee .audit/check-default.txt`
- `cargo check --all-targets --all-features 2>&1 | tee .audit/check-all-features.txt`
- `cargo check --all-targets --no-default-features 2>&1 | tee .audit/check-no-default-features.txt`
- `cargo test --all-targets 2>&1 | tee .audit/test-default.txt`
- `cargo test --all-targets --all-features 2>&1 | tee .audit/test-all-features.txt`
- `cargo test --doc --all-features 2>&1 | tee .audit/test-doc-all-features.txt`
- `cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tee .audit/clippy-all-features.txt`
- `cargo doc --all-features --no-deps 2>&1 | tee .audit/doc-all-features.txt`
- `cargo audit 2>&1 | tee .audit/cargo-audit.txt || true`
- `cargo deny check 2>&1 | tee .audit/cargo-deny.txt || true`
- Source inspected with `nl -ba` and `rg`: `Cargo.toml`, `README.md`, `src/lib.rs`, `src/core/request.rs`, `src/core/message.rs`, `src/core/model.rs`, `src/core/tool.rs`, `src/runtime/turn.rs`, `src/runtime/tools.rs`, and `src/core/stream.rs`.

## Confirmed issues from this loop

### A1-1: Public model enum docs contain broken intra-doc links

- Status: confirmed
- Severity: low
- Evidence type: failing-command
- Files/lines: `src/providers/anthropic/models.rs:4`, `src/providers/gemini/models.rs:4`, `src/providers/openai/models.rs:4`
- Trigger: Run `cargo doc --all-features --no-deps`.
- Observed behavior: Rustdoc emits `broken_intra_doc_links` warnings for `AnthropicChatModel::new`, `GeminiChatModel::new`, and `OpenAiChatModel::new`; the output is captured in `.audit/doc-all-features.txt`.
- Expected behavior: Public rustdoc links should resolve when documentation is generated for the crate.
- User impact: Published docs contain broken links from provider model enums to the constructors users are told to call for custom model IDs.
- Root cause: The links are written from the model enum modules without a resolvable path to the chat model types in sibling modules.
- Minimal fix: Qualify the links, e.g. link to `crate::providers::anthropic::model::AnthropicChatModel::new`, `crate::providers::gemini::model::GeminiChatModel::new`, and `crate::providers::openai::model::OpenAiChatModel::new`, or import the target types for docs.
- Test to add: Add a docs CI command with `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps`.
- What would prove this false: A clean `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` run at the audited commit, or cited lines no longer containing unresolved links.

## Unproven / rejected candidates

### Candidate: Feature combinations do not compile

- Why rejected/unproven: `cargo check --all-targets`, `cargo check --all-targets --all-features`, and `cargo check --all-targets --no-default-features` all completed successfully in `.audit/check-default.txt`, `.audit/check-all-features.txt`, and `.audit/check-no-default-features.txt`.

### Candidate: Unit tests or doctests fail under the audited feature sets

- Why rejected/unproven: `cargo test --all-targets`, `cargo test --all-targets --all-features`, and `cargo test --doc --all-features` all completed successfully. Doctests reported zero doctests rather than failures.

### Candidate: Clippy blocks release under all features

- Why rejected/unproven: `cargo clippy --all-targets --all-features -- -D warnings` completed successfully in `.audit/clippy-all-features.txt`.

### Candidate: Optional dependency-audit tools found dependency issues

- Why rejected/unproven: `cargo audit` and `cargo deny` were not installed in this environment, captured in `.audit/cargo-audit.txt` and `.audit/cargo-deny.txt`; tool unavailability is not a code issue.

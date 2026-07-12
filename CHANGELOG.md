# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/christopher-caldwell/rust_ai_sdk/compare/v0.0.6...v0.0.7) - 2026-07-12

### Added

- Strengthen foundation

### Changed

- Stream UI-message deltas immediately instead of buffering complete model turns.
- Validate request/tool transactions and tool definitions before provider calls.
- Reject browser-supplied system roles and redact public stream/tool failures by default.
- Treat premature stream termination as an error.
- Add configurable provider HTTP clients and base URLs.
- Make request building and tool registration validated operations.

### Fixed

- Exclude internal audit/agent artifacts from crate packages and require license packaging.
- Make core-only doctests and rustdoc feature-aware.
- Prevent dirty source from being published and tagged as a different commit.
- Add `just release` as the guarded end-to-end release command.
- Keep release-file validation compatible with the system Bash on macOS.
- Update the locked TLS certificate validator to a release without known vulnerabilities.
- Use process-unique UI message identifiers and bounded concurrent tool execution.

## [0.0.6](https://github.com/christopher-caldwell/rust_ai_sdk/compare/v0.0.5...v0.0.6) - 2026-07-01

### Changed

- Advance the crate and example lockfiles to version 0.0.6.

## [0.0.5](https://github.com/christopher-caldwell/rust_ai_sdk/compare/v0.0.4...v0.0.5) - 2026-06-30

### Changed

- Improve release preparation, publishing safeguards, and project documentation.

## [0.0.4](https://github.com/christopher-caldwell/rust_ai_sdk/compare/v0.0.3...v0.0.4) - 2026-06-08

### Added

- Adding agents to always prompt the graph

### Fixed

- Deleting ua
- Adding back ua, with right ignore

### Other

- Fix terminal event handling and provider finish normalization ([#4](https://github.com/christopher-caldwell/rust_ai_sdk/pull/4))

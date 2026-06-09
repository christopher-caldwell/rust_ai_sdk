#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/publish-crate.sh              # generate release changes + dry-run publish
  scripts/publish-crate.sh --check-only # run preflight + cargo publish --dry-run
  scripts/publish-crate.sh --publish    # run preflight + publish to crates.io

Options:
  --publish      Actually upload to crates.io. Default prepares a dry run only.
  --check-only   Skip release-plz update and only run validation/dry-run publish.
  --allow-dirty  Allow publishing checks from a dirty git worktree.
  -h, --help     Show this help.

Before first publish:
  1. Create a crates.io account and verify your email.
  2. Create an API token at https://crates.io/me.
  3. Run: cargo login
  4. Choose a license and add either `license = "..."` or `license-file = "..."`.

Local release flow:
  1. Run: scripts/publish-crate.sh
  2. Review the generated Cargo.toml, Cargo.lock, and CHANGELOG.md changes.
  3. Commit those release changes.
  4. Run: scripts/publish-crate.sh --publish
USAGE
}

mode="prepare"
allow_dirty=false

for arg in "$@"; do
  case "$arg" in
    --publish)
      mode="publish"
      ;;
    --check-only)
      mode="check-only"
      ;;
    --allow-dirty)
      allow_dirty=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

manifest="Cargo.toml"
example_manifests=(
  "examples/standalone/Cargo.toml"
  "examples/chatbot/server/Cargo.toml"
  "examples/chatbot/server-explicit/Cargo.toml"
)

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_clean_worktree() {
  if [[ "$allow_dirty" == false ]] && [[ -n "$(git status --porcelain)" ]]; then
    cat >&2 <<'MSG'
Git worktree is dirty.

Commit or stash changes before preparing or publishing a release so the
generated changelog, version, and published crate map to a specific commit.
If you intentionally want to test with local changes, rerun:
  scripts/publish-crate.sh --allow-dirty
MSG
    exit 1
  fi
}

require_manifest_field() {
  local field="$1"
  if ! grep -Eq "^[[:space:]]*$field[[:space:]]*=" "$manifest"; then
    echo "Cargo.toml is missing required/recommended package field: $field" >&2
    exit 1
  fi
}

package_value() {
  local field="$1"
  sed -nE "s/^[[:space:]]*$field[[:space:]]*=[[:space:]]*\"([^\"]*)\".*/\1/p" "$manifest" | head -n 1
}

refresh_example_lockfiles() {
  local example_manifest

  for example_manifest in "${example_manifests[@]}"; do
    echo "==> Updating lockfile for $example_manifest"
    cargo update --manifest-path "$example_manifest" -p "$name"
  done
}

require_command cargo
require_command git

if [[ "$mode" == "prepare" ]]; then
  require_command release-plz
fi

if [[ ! -s README.md ]]; then
  echo "README.md is missing or empty." >&2
  exit 1
fi

require_manifest_field "description"
require_manifest_field "repository"
require_manifest_field "readme"

if ! grep -Eq "^[[:space:]]*(license|license-file)[[:space:]]*=" "$manifest"; then
  cat >&2 <<'MSG'
Cargo.toml is missing a license.

Choose the license you want for this project, then add one of:
  license = "MIT"
  license = "MIT OR Apache-2.0"
  license-file = "LICENSE"

Do not publish until the license is intentional.
MSG
  exit 1
fi

ensure_clean_worktree

name="$(package_value name)"
version="$(package_value version)"
tag="v$version"

if [[ -z "$name" || -z "$version" ]]; then
  echo "Could not read package name/version from Cargo.toml." >&2
  exit 1
fi

cargo_dirty_args=()
if [[ "$allow_dirty" == true || "$mode" == "prepare" ]]; then
  cargo_dirty_args=(--allow-dirty)
fi

echo "==> Package: $name $version"

if [[ "$mode" == "prepare" ]]; then
  echo "==> Updating version and changelog with release-plz"
  release-plz update

  version="$(package_value version)"
  tag="v$version"
  echo "==> Prepared package version: $version"

  refresh_example_lockfiles
fi

echo "==> Verifying cargo metadata"
cargo metadata --no-deps --format-version=1 >/dev/null

echo "==> Checking formatting"
cargo fmt --check

echo "==> Running tests"
cargo test

echo "==> Running all-feature tests"
cargo test --all-features

echo "==> Checking standalone examples"
cargo check --manifest-path examples/standalone/Cargo.toml

echo "==> Checking chatbot server example"
cargo check --manifest-path examples/chatbot/server/Cargo.toml

echo "==> Checking explicit chatbot server example"
cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml

echo "==> Building public rustdoc"
cargo doc --all-features --no-deps

echo "==> Listing packaged files"
cargo package --list "${cargo_dirty_args[@]}"

echo "==> Running cargo publish --dry-run"
cargo publish --dry-run "${cargo_dirty_args[@]}"

if [[ "$mode" != "publish" ]]; then
  if [[ "$mode" == "prepare" ]]; then
    cat <<MSG

Release preparation passed. Nothing was uploaded.

Review the generated release changes, then commit them:
  git diff -- Cargo.toml Cargo.lock CHANGELOG.md examples
  git add Cargo.toml Cargo.lock CHANGELOG.md examples
  git commit -m "chore: prepare $name $version release"

When you are ready to publish the committed release:
  scripts/publish-crate.sh --publish

After publishing, tag the released commit:
  git tag $tag
  git push origin $tag
MSG
    exit 0
  fi

  cat <<MSG

Dry run passed. Nothing was uploaded.

When you are ready:
  1. Ensure Cargo.toml version is correct: $version
  2. Ensure you are logged in: cargo login
  3. Commit these exact files.
  4. Run: scripts/publish-crate.sh --publish

After publishing, tag the released commit:
  git tag $tag
  git push origin $tag
MSG
  exit 0
fi

cat <<MSG

About to publish $name $version to crates.io.

Publishing is permanent:
  - This exact version cannot be overwritten.
  - The uploaded source cannot be deleted.
  - A bad version can only be yanked later.

MSG

read -r -p "Type '$name $version' to publish: " confirmation
if [[ "$confirmation" != "$name $version" ]]; then
  echo "Confirmation did not match. Aborting."
  exit 1
fi

echo "==> Publishing"
cargo publish "${cargo_dirty_args[@]}"

cat <<MSG

Published $name $version.

Recommended next step:
  git tag $tag
  git push origin $tag
MSG

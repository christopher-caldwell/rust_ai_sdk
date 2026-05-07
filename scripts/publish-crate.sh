#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/publish-crate.sh            # run preflight + cargo publish --dry-run
  scripts/publish-crate.sh --publish  # run preflight + publish to crates.io

Options:
  --publish     Actually upload to crates.io. Default is dry-run only.
  --allow-dirty Allow publishing checks from a dirty git worktree.
  -h, --help    Show this help.

Before first publish:
  1. Create a crates.io account and verify your email.
  2. Create an API token at https://crates.io/me.
  3. Run: cargo login
  4. Choose a license and add either `license = "..."` or `license-file = "..."`.
USAGE
}

publish=false
allow_dirty=false

for arg in "$@"; do
  case "$arg" in
    --publish)
      publish=true
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

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
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

require_command cargo
require_command git

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

if [[ "$allow_dirty" == false ]] && [[ -n "$(git status --porcelain)" ]]; then
  cat >&2 <<'MSG'
Git worktree is dirty.

Commit or stash changes before publishing so the published crate maps to a
specific commit. If you intentionally want to test with local changes, rerun:
  scripts/publish-crate.sh --allow-dirty
MSG
  exit 1
fi

name="$(package_value name)"
version="$(package_value version)"
tag="v$version"

if [[ -z "$name" || -z "$version" ]]; then
  echo "Could not read package name/version from Cargo.toml." >&2
  exit 1
fi

echo "==> Package: $name $version"
echo "==> Verifying cargo metadata"
cargo metadata --no-deps --format-version=1 >/dev/null

echo "==> Checking formatting"
cargo fmt --check

echo "==> Running tests"
cargo test

echo "==> Running all-feature tests"
cargo test --all-features

echo "==> Listing packaged files"
cargo package --list

echo "==> Running cargo publish --dry-run"
cargo publish --dry-run

if [[ "$publish" == false ]]; then
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
cargo publish

cat <<MSG

Published $name $version.

Recommended next step:
  git tag $tag
  git push origin $tag
MSG

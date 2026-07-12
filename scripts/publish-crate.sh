#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/publish-crate.sh --release      # prepare, commit, publish, and tag
  scripts/publish-crate.sh              # generate release changes + dry-run publish
  scripts/publish-crate.sh --check-only # run preflight + cargo publish --dry-run
  scripts/publish-crate.sh --publish    # run preflight + publish to crates.io

Options:
  --release      Run the complete release workflow from a clean worktree.
  --publish      Actually upload to crates.io. Default prepares a dry run only.
  --check-only   Skip release-plz update and only run validation/dry-run publish.
  --yes          Skip the publish confirmation prompt. Valid with --publish or --release.
  --allow-dirty  Allow publishing checks from a dirty git worktree.
  -h, --help     Show this help.

Before first publish:
  1. Create a crates.io account and verify your email.
  2. Create an API token at https://crates.io/me.
  3. Run: cargo login
  4. Choose a license and add either `license = "..."` or `license-file = "..."`.

Local release flow:
  1. Start from a clean branch that can be pushed to origin.
  2. Run: scripts/publish-crate.sh --release
  3. Confirm the permanent crates.io publish when prompted.

The release command prepares and validates the release, commits and pushes the
release metadata, publishes the crate, then creates and pushes the Git tag.
USAGE
}

mode="prepare"
allow_dirty=false
skip_publish_confirmation=false

for arg in "$@"; do
  case "$arg" in
    --release)
      mode="release"
      ;;
    --publish)
      mode="publish"
      ;;
    --check-only)
      mode="check-only"
      ;;
    --yes)
      skip_publish_confirmation=true
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

if [[ "$skip_publish_confirmation" == true && "$mode" != "publish" && "$mode" != "release" ]]; then
  echo "--yes is only valid with --publish or --release." >&2
  usage >&2
  exit 2
fi

if [[ "$allow_dirty" == true && ( "$mode" == "publish" || "$mode" == "release" ) ]]; then
  echo "--allow-dirty is not permitted with --publish or --release because the crate must match its Git tag." >&2
  exit 2
fi

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
MSG
    if [[ "$mode" != "publish" && "$mode" != "release" ]]; then
      cat >&2 <<'MSG'
If you intentionally want to test with local changes, rerun:
  scripts/publish-crate.sh --allow-dirty
MSG
    fi
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

sync_readme_version() {
  local release_version="$1"

  RELEASE_VERSION="$release_version" perl -0pi -e '
    s/(another-ai-sdk\s*=\s*(?:\{\s*version\s*=\s*)?")\d+\.\d+\.\d+(")/$1$ENV{RELEASE_VERSION}$2/g
  ' README.md
}

assert_only_release_files_changed() {
  local path

  while IFS= read -r path; do
    case "$path" in
      Cargo.toml|Cargo.lock|CHANGELOG.md|README.md|examples/standalone/Cargo.lock|examples/chatbot/server/Cargo.lock|examples/chatbot/server-explicit/Cargo.lock)
        ;;
      "")
        ;;
      *)
        echo "Release preparation changed an unexpected file: $path" >&2
        return 1
        ;;
    esac
  done < <({
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | sort -u)
}

commit_and_push_release() {
  local branch
  branch="$(git branch --show-current)"
  if [[ -z "$branch" ]]; then
    echo "A release must be created from a named Git branch." >&2
    exit 1
  fi

  assert_only_release_files_changed
  git add -- \
    Cargo.toml \
    Cargo.lock \
    CHANGELOG.md \
    README.md \
    examples/standalone/Cargo.lock \
    examples/chatbot/server/Cargo.lock \
    examples/chatbot/server-explicit/Cargo.lock

  if git diff --cached --quiet; then
    echo "Release preparation produced no changes to commit." >&2
    exit 1
  fi

  echo "==> Committing release metadata"
  git commit -m "chore: prepare $name $version release"

  echo "==> Pushing release commit to origin/$branch"
  git push origin "HEAD:refs/heads/$branch"
}

ensure_tag_is_available() {
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "Local git tag already exists: $tag" >&2
    exit 1
  fi

  if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
    echo "Remote git tag already exists on origin: $tag" >&2
    exit 1
  fi
}

tag_and_push_release() {
  echo "==> Creating git tag $tag"
  git tag "$tag"

  echo "==> Pushing git tag $tag"
  git push origin "$tag"
}

require_command cargo
require_command git

if [[ "$mode" == "prepare" || "$mode" == "release" ]]; then
  require_command release-plz
  require_command perl
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
starting_version="$version"

if [[ -z "$name" || -z "$version" ]]; then
  echo "Could not read package name/version from Cargo.toml." >&2
  exit 1
fi

cargo_allow_dirty=false
if [[ "$allow_dirty" == true || "$mode" == "prepare" || "$mode" == "release" ]]; then
  cargo_allow_dirty=true
fi

cargo_package_list() {
  if [[ "$cargo_allow_dirty" == true ]]; then
    cargo package --list --allow-dirty
  else
    cargo package --list
  fi
}

verify_release_metadata() {
  if ! grep -Fq "another-ai-sdk = \"$version\"" README.md; then
    echo "README.md does not contain the current dependency version $version." >&2
    exit 1
  fi
  if ! grep -Fq "## [$version]" CHANGELOG.md; then
    echo "CHANGELOG.md does not contain a release section for $version." >&2
    exit 1
  fi
}

verify_package_contents() {
  local packaged_files
  packaged_files="$(cargo_package_list)"
  if grep -Eq '^(audits/|AGENTS\.md|\.understand-anything/|target/)' <<<"$packaged_files"; then
    echo "Packaged files contain internal repository artifacts." >&2
    grep -E '^(audits/|AGENTS\.md|\.understand-anything/|target/)' <<<"$packaged_files" >&2
    exit 1
  fi
  if ! grep -Eq '^LICENSE($|[-.])' <<<"$packaged_files"; then
    echo "Packaged files do not contain a license text." >&2
    exit 1
  fi
  printf '%s\n' "$packaged_files"
}

cargo_publish_dry_run() {
  if [[ "$cargo_allow_dirty" == true ]]; then
    cargo publish --dry-run --allow-dirty
  else
    cargo publish --dry-run
  fi
}

cargo_publish_release() {
  if [[ "$cargo_allow_dirty" == true ]]; then
    cargo publish --allow-dirty
  else
    cargo publish
  fi
}

echo "==> Package: $name $version"

if [[ "$mode" == "prepare" || "$mode" == "release" ]]; then
  echo "==> Updating version and changelog with release-plz"
  release-plz update

  version="$(package_value version)"
  tag="v$version"
  echo "==> Prepared package version: $version"

  if [[ "$mode" == "release" && "$version" == "$starting_version" ]]; then
    echo "release-plz did not advance the package version; there is no release to publish." >&2
    exit 1
  fi

  sync_readme_version "$version"
  refresh_example_lockfiles
fi

verify_release_metadata

echo "==> Verifying cargo metadata"
cargo metadata --no-deps --format-version=1 >/dev/null

echo "==> Checking formatting"
cargo fmt --all -- --check

echo "==> Running strict Clippy"
cargo clippy --all-targets --all-features -- -D warnings

echo "==> Running tests"
cargo test

echo "==> Running all-feature tests"
cargo test --all-features

echo "==> Testing supported feature combinations"
cargo test --no-default-features
cargo test --no-default-features --features openai
cargo test --no-default-features --features anthropic
cargo test --no-default-features --features gemini
cargo test --no-default-features --features providers-all,streaming
cargo test --no-default-features --features message-stream

echo "==> Checking standalone examples"
cargo check --manifest-path examples/standalone/Cargo.toml

echo "==> Checking chatbot server example"
cargo check --manifest-path examples/chatbot/server/Cargo.toml

echo "==> Checking explicit chatbot server example"
cargo check --manifest-path examples/chatbot/server-explicit/Cargo.toml

echo "==> Building public rustdoc"
RUSTDOCFLAGS="-D warnings" cargo doc --no-default-features --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "==> Listing packaged files"
verify_package_contents

echo "==> Running cargo publish --dry-run"
cargo_publish_dry_run

if [[ "$mode" != "publish" && "$mode" != "release" ]]; then
  if [[ "$mode" == "prepare" ]]; then
    cat <<MSG

Release preparation passed. Nothing was uploaded.

Review the generated release changes, then commit them:
  git diff -- Cargo.toml Cargo.lock CHANGELOG.md examples
  git add Cargo.toml Cargo.lock CHANGELOG.md examples
  git commit -m "chore: prepare $name $version release"

When you are ready to publish the committed release:
  scripts/publish-crate.sh --publish

The publish command creates and pushes tag $tag after crates.io accepts the package.
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

The publish command creates and pushes tag $tag after crates.io accepts the package.
MSG
  exit 0
fi

ensure_tag_is_available

cat <<MSG

About to release $name $version.

The command will commit and push release metadata when needed, then publish to
crates.io. Publishing is permanent:
  - This exact version cannot be overwritten.
  - The uploaded source cannot be deleted.
  - A bad version can only be yanked later.

MSG

if [[ "$skip_publish_confirmation" == false ]]; then
  read -r -p "Type '$name $version' to publish: " confirmation
  if [[ "$confirmation" != "$name $version" ]]; then
    echo "Confirmation did not match. Aborting."
    exit 1
  fi
fi

if [[ "$mode" == "release" ]]; then
  commit_and_push_release
  cargo_allow_dirty=false
  ensure_clean_worktree
fi

echo "==> Publishing"
cargo_publish_release

tag_and_push_release

cat <<MSG

Published $name $version.
Tagged and pushed $tag.
MSG

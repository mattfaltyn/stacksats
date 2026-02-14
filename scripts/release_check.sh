#!/usr/bin/env bash

set -euo pipefail

log() {
  echo "[release_check] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_packaging_tools() {
  log "Ensuring packaging tools are available"
  if python -m pip install --upgrade build twine >/dev/null 2>&1; then
    return 0
  fi

  log "Could not refresh build/twine via pip (possibly constrained SSL trust roots)."
  log "Falling back to currently installed local tooling."

  if ! python -m build --version >/dev/null 2>&1; then
    echo "python -m build is not available. Install it before releasing." >&2
    exit 1
  fi

  if ! python -m twine --version >/dev/null 2>&1; then
    echo "python -m twine is not available. Install it before releasing." >&2
    exit 1
  fi
}

is_ssl_constraint_failure() {
  local error_file="$1"
  grep -Eqi "SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|SSL: CERTIFICATE_VERIFY_FAILED|OSStatus -26276" "$error_file"
}

build_with_ssl_fallback() {
  local build_err
  build_err="$(mktemp)"

  if python -m build 2> >(tee "$build_err" >&2); then
    rm -f "$build_err"
    return 0
  fi

  if is_ssl_constraint_failure "$build_err"; then
    log "Isolated build failed due SSL trust constraints; retrying without isolation."
    rm -rf dist/ build/ .eggs/ ./*.egg-info
    python -m build --no-isolation
    rm -f "$build_err"
    return 0
  fi

  rm -f "$build_err"
  echo "Build failed for a non-SSL reason." >&2
  return 1
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Must run inside a git repository." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit/stash changes before release." >&2
  git status --short
  exit 1
fi

require_cmd python
require_cmd git

ensure_packaging_tools

log "Running lint"
python -m ruff check .

log "Running tests"
python -m pytest tests/ -v -n auto

log "Cleaning build artifacts"
rm -rf dist/ build/ .eggs/ ./*.egg-info

log "Building source and wheel distributions"
build_with_ssl_fallback

log "Validating package metadata/rendering"
python -m twine check dist/*

log "Release preflight passed"

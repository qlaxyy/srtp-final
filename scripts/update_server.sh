#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  echo "Repository not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

git -C "${PROJECT_ROOT}" status --short
git -C "${PROJECT_ROOT}" pull --ff-only

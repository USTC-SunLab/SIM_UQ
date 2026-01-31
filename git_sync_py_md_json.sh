#!/usr/bin/env bash
set -euo pipefail

# Sync only .py/.md/.json changes (no repo init). Usage:
#   ./git_sync_py_md_json.sh "your commit message"

cd "$(dirname "$0")"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository. Aborting." >&2
  exit 1
fi

# Stage modifications/deletions and new files for the allowed extensions
PATTERNS=("*.py" "*.md" "*.json")

git add -u -- "${PATTERNS[@]}"
git add -- "${PATTERNS[@]}"

# If nothing staged, exit
if git diff --cached --quiet; then
  echo "No .py/.md/.json changes to commit."
  exit 0
fi

MSG=${1:-"sync py/md/json $(date +%Y-%m-%d_%H-%M-%S)"}

git commit -m "$MSG"

git push origin HEAD

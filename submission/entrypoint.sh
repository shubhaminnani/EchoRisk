#!/usr/bin/env bash
# entrypoint.sh — runs once, no arguments (harness contract §1).
set -euo pipefail

if [[ ! -f /input/manifest.csv ]]; then
  echo "error: /input/manifest.csv not found. Check the input mount." >&2
  exit 1
fi

mkdir -p /output

set +e
python3 /app/infer.py
EXIT_CODE=$?
set -e

if [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "error: infer.py exited with code ${EXIT_CODE}." >&2
  exit "${EXIT_CODE}"
fi

if [[ ! -s /output/predictions.csv ]]; then
  echo "error: /output/predictions.csv was not written or is empty." >&2
  exit 1
fi

exit 0

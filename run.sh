#!/usr/bin/env bash
# Launch the marimo app in edit mode.
#
# This points uv at an external venv (~/.uv_envs/day3b_embeddings) because
# pCloud Drive doesn't support symlinks, which the default uv .venv layout needs.
#
# First time only: bootstrap the venv with `./run.sh --setup`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export UV_PROJECT_ENVIRONMENT="$HOME/.uv_envs/day3b_embeddings"
export UV_LINK_MODE=copy

if [[ "${1:-}" == "--setup" ]]; then
  echo "Creating venv at $UV_PROJECT_ENVIRONMENT and installing dependencies..."
  uv venv "$UV_PROJECT_ENVIRONMENT"
  uv sync
  echo "Done."
  exit 0
fi

if [[ ! -d "$UV_PROJECT_ENVIRONMENT" ]]; then
  echo "Venv missing. Run ./run.sh --setup first." >&2
  exit 1
fi

exec uv run python -m marimo run app.py

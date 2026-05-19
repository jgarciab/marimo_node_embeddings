#!/usr/bin/env bash
# Build a static WASM bundle of the app for hosting (e.g. on GitHub Pages).
#
# Output: ./build/index.html and any sibling assets marimo produces.
# Open the HTML in a browser or upload the build/ directory to a static host.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export UV_PROJECT_ENVIRONMENT="$HOME/.uv_envs/day3b_embeddings"
export UV_LINK_MODE=copy

OUT_DIR="${1:-build}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Exporting WASM bundle to $OUT_DIR/index.html ..."
uv run marimo export html-wasm app.py -o "$OUT_DIR/index.html" --mode run "$@"

echo
echo "Done. To preview locally:"
echo "  cd $OUT_DIR && python -m http.server 8000"
echo "  then open http://localhost:8000/"

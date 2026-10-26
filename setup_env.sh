#!/usr/bin/env bash
# Create the isolated Python 3.12 environment for the GraphRAG notebooks.
#
# Why: the default Codespaces Python is 3.14, but graspologic (hierarchical Leiden)
# only supports Python < 3.13, so `pip install graspologic` fails / conflicts there.
#
# Usage:  bash setup_env.sh        → creates .venv and the "GraphRAG (Python 3.12)" kernel
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (fast Python/package manager)..."
  python3 -m pip install --quiet --user uv
fi

# uv downloads a standalone Python 3.12 if the system one lacks venv/ensurepip.
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt -c requirements.lock.txt

.venv/bin/python -m ipykernel install --user --name graphrag-py312 --display-name "GraphRAG (Python 3.12)"

echo
echo "✅ Environment ready."
echo "   Notebook: open graphrag_ai_copyright_v2.ipynb and select the kernel 'GraphRAG (Python 3.12)'."
echo "   CLI:      .venv/bin/python run_pipeline.py --config topics/ai_copyright.json --dry-run"

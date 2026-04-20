#!/bin/bash

set -e

echo "Initializing environment..."
cd proj2/vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v
pip install nixl[cu13] 
#!/usr/bin/env bash
# Create or refresh the TensorFlow GPU venv for stock-prediction (WSL / RTX 5090).
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

# Recreate venv if it was copied/moved from another path (breaks shebangs and pip).
if [[ -d .venv ]] && grep -q "stock-tf/.venv" .venv/pyvenv.cfg 2>/dev/null; then
  echo "Detected migrated .venv (old stock-tf path). Recreating..."
  rm -rf .venv
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# Never use --user installs inside a virtualenv.
mkdir -p .venv
cat > .venv/pip.conf <<'EOF'
[install]
user = false
EOF

# shellcheck disable=SC1091
source .venv/bin/activate
unset PIP_USER

pip install --upgrade pip
pip install -r requirements.txt

bash "$(dirname "$0")/cleanup-jupyter-kernels.sh"
bash "$(dirname "$0")/install-jupyter-kernel.sh"
python "$(dirname "$0")/fix_notebook_kernels.py"

python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow:', tf.__version__); print('GPU available:', bool(gpus)); print(gpus)"

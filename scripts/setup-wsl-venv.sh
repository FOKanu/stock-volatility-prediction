#!/usr/bin/env bash
# Create or refresh the TensorFlow GPU venv for stock-prediction (WSL / RTX 5090).
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install \
  "tensorflow[and-cuda]==2.21.0" \
  jupyterlab ipykernel \
  numpy pandas matplotlib seaborn scikit-learn scipy \
  "nvidia-cudnn-cu12>=9.3.0" \
  "nvidia-cuda-nvcc-cu12>=12.5.0"

python -m ipykernel install --user \
  --name stock-prediction-wsl \
  --display-name "Python (stock-prediction-wsl)"

bash "$(dirname "$0")/install-jupyter-kernel.sh"

python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow:', tf.__version__); print('GPU available:', bool(gpus)); print(gpus)"

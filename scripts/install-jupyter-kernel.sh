#!/usr/bin/env bash
# Write Jupyter kernel.json with TensorFlow GPU env for RTX 5090 (WSL).
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT}/.venv"
KERNEL_NAME="stock-prediction-wsl"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
SP="${VENV}/lib/python3.12/site-packages/nvidia"

LD_PATHS=(
  "${SP}/cudnn/lib"
  "${SP}/cublas/lib"
  "${SP}/cuda_runtime/lib"
  "${SP}/cusolver/lib"
  "${SP}/cusparse/lib"
  "${SP}/curand/lib"
  "${SP}/nvjitlink/lib"
  "${SP}/cufft/lib"
  "${SP}/nccl/lib"
  "/usr/lib/wsl/lib"
)
LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PATHS[*]}"):"

mkdir -p "$KERNEL_DIR"
cat > "${KERNEL_DIR}/kernel.json" <<EOF
{
  "argv": [
    "${VENV}/bin/python",
    "-Xfrozen_modules=off",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (${KERNEL_NAME})",
  "language": "python",
  "metadata": {
    "debugger": true,
    "supported_encryption": "curve"
  },
  "kernel_protocol_version": "5.5",
  "env": {
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=${SP}/cuda_nvcc",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true"
  }
}
EOF
echo "Wrote ${KERNEL_DIR}/kernel.json"

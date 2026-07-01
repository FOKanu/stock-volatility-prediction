#!/usr/bin/env bash
# Activate stock-prediction .venv with GPU env vars. Use this in every shell session.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT}/.venv/bin/activate"
cd "$PROJECT"
echo "Activated: ${VIRTUAL_ENV}"
echo "Python:    $(which python)"
echo "Project:   ${PROJECT}"

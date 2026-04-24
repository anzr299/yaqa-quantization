#!/usr/bin/env bash
set -euo pipefail

# Setup script for YAQA quantization environment
#
# Usage:
#   ./setup_env.sh [ENV_DIR]
#
# Example:
#   ./setup_env.sh              # creates ../.env-yaqa
#   ./setup_env.sh /path/to/env # creates env at specified path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$(dirname "$SCRIPT_DIR")/.env-yaqa}"
FHT_DIR="$(dirname "$SCRIPT_DIR")/fast-hadamard-transform"

echo "[1/7] Creating virtual environment at $ENV_DIR"
if [[ -d "$ENV_DIR" ]]; then
    echo "  Environment already exists, skipping creation."
else
    python3.10 -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "[2/7] Upgrading pip"
"$PIP" install --upgrade pip setuptools wheel

echo "[3/7] Installing PyTorch 2.7.0 (CUDA 12.6)"
"$PIP" install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

echo "[4/7] Installing Python dependencies"
"$PIP" install \
    accelerate==1.6.0 \
    datasets==3.5.0 \
    glog==0.3.1 \
    huggingface-hub==0.30.1 \
    lm_eval==0.4.4 \
    numpy==2.2.4 \
    scipy==1.15.2 \
    tqdm==4.67.1 \
    transformers==4.51.0 \
    safetensors==0.5.3 \
    sentencepiece==0.2.0 \
    protobuf \
    peft==0.15.1 \
    pytorch-lightning==2.5.1.post0

echo "[5/7] Installing fast_hadamard_transform from source"
if [[ -d "$FHT_DIR" ]]; then
    echo "  Found existing source at $FHT_DIR"
else
    echo "  Cloning fast-hadamard-transform..."
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "$FHT_DIR"
fi
"$PIP" install --no-build-isolation -e "$FHT_DIR"

echo "[6/7] Installing QTIP CUDA kernels"
"$PIP" install --no-build-isolation -e "$SCRIPT_DIR/qtip-kernels"

echo "[7/7] Installing remaining utilities"
"$PIP" install \
    matplotlib \
    seaborn \
    pandas \
    ninja \
    pybind11

echo ""
echo "===== Setup complete ====="
echo "Activate with: source $ENV_DIR/bin/activate"
echo "Run pipeline:  ./run_yaqa_pipeline.sh <MODEL_ID> <RUN_TAG>"

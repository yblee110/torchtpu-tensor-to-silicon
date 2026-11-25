#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/3] Checking curl & installing uv for Python 3.12..."
if ! command -v curl >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y curl
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> [2/3] Creating Python 3.12 virtual environment (.venv)..."
uv venv .venv --python 3.12 --seed
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [3/3] Installing TorchTPU v0.2.0, XProf, and Jupyter tools..."
uv pip install "setuptools<81"
uv pip install torch_tpu jax xprof tensorboard tensorboard-plugin-profile matplotlib jupyter jupyter_http_over_ws nbconvert

echo "==> Verifying TPU v6e device via TorchTPU..."
TPU_VISIBLE_DEVICES=0 python -c '
import torch
device = torch.device("tpu")
props = torch.tpu.get_device_properties(device)
print(f"🎉 TorchTPU Ready! Device: {props.name} | Visible Devices: {torch.accelerator.device_count()}")
'

echo "✅ Setup complete! Next, run: source .venv/bin/activate"

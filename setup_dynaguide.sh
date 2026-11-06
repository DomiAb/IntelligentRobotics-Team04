#!/bin/bash
# One-command install of the DynaGuide env on Ubuntu (Windows: setup_dynaguide.bat).
# Usage: bash setup_dynaguide.sh
set -euo pipefail

ENV_NAME="dynaguide_env"
ENV_FILE="dynaguide_env.yml"
cd "$(cd "$(dirname "$0")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda and open a new terminal." >&2
    exit 1
fi

# System libraries conda doesn't provide: EGL rendering in pybullet, and libGL/glib for opencv-python.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi
$SUDO apt-get update -y
$SUDO apt-get install -y libegl1 libgl1 libglib2.0-0

EXISTING_ENVS="$(conda env list)"
if grep -q "^$ENV_NAME " <<< "$EXISTING_ENVS"; then
    echo "Updating existing env $ENV_NAME..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
    echo "Creating env $ENV_NAME..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

mkdir -p DynaGuide-main/results

conda run -n "$ENV_NAME" python -c "import torch, pybullet, cv2, gym, hydra, robomimic, calvin_env, quaternion, git, einops, sklearn, diffusers; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo
echo "Done. Next: conda activate $ENV_NAME && cd DynaGuide-main"

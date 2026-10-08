#!/bin/bash
set -euo pipefail

echo "--- Starting Setup ---"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

PIP_CMD="python -m pip"

# ----------------------------
# 1. System deps
# ----------------------------
echo "[1/4] System dependencies..."

sudo apt-get update -y
sudo apt-get install -y cmake build-essential libosmesa6-dev libegl1-mesa-dev libgl1-mesa-dev ffmpeg pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev


# ----------------------------
# 2. Submodules
# ----------------------------
echo "[2/4] Submodules..."
git submodule update --init --recursive --remote


# ----------------------------
# 3. Robotics packages (IMPORTANT FIX)
# ----------------------------
echo "[3/4] Installing robotics packages..."


if [ -d "./calvin" ]; then
    echo "Installing calvin..."
    (cd ./calvin && bash install.sh)
    $PIP_CMD install -e ./calvin
fi

if [ -d "./robomimic" ]; then
    echo "Installing robomimic..."
    $PIP_CMD install -e ./robomimic
fi

if [ -d "./mimicgen" ]; then
    echo "Installing mimicgen..."
    $PIP_CMD install -e ./mimicgen
fi

if [ -d "./robosuite" ]; then
    echo "Installing robosuite..."
    $PIP_CMD install -e ./robosuite
fi

if [ -d "./dsrl" ]; then
    echo "Installing dsrl..."
    $PIP_CMD install -e ./dsrl
fi

if [ -d "./dsrl/dppo" ]; then
    echo "Installing dsrl/dppo..."
    $PIP_CMD install -e ./dsrl/dppo
fi

if [ -d "./dsrl/stable-baselines3" ]; then
    echo "Installing dsrl/stable-baselines3..."
    $PIP_CMD install -e ./dsrl/stable-baselines3
fi

# ----------------------------
# 4. Macros setup (safe)
# ----------------------------
echo "[4/4] Macros setup..."

if [ -f "./robomimic/robomimic/scripts/setup_macros.py" ]; then
    python ./robomimic/robomimic/scripts/setup_macros.py || true
fi


if [ -f "robosuite/robosuite/scripts/setup_macros.py" ]; then
    python robosuite/robosuite/scripts/setup_macros.py || true
fi

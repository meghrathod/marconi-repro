#!/bin/bash
# ==============================================================================
# setup_local_env.sh — Provision a Chameleon Cloud instance for mlsys-marconi
#
# Usage:
#   ./setup_local_env.sh <SSH_HOST>
#
# Where <SSH_HOST> is either:
#   - An SSH config alias (e.g. "chameleon")
#   - A full user@host address (e.g. "cc@192.5.86.151")
#
# Prerequisites:
#   - SSH access to the remote host (key-based auth configured)
#   - The remote instance should be a Chameleon Cloud Ubuntu 22.04 image
#     (e.g. CC-Ubuntu22.04-CUDA on bare metal or KVM)
# ==============================================================================

set -euo pipefail

# --------------- Argument Parsing ---------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <SSH_HOST>"
    echo "  SSH_HOST: SSH alias or user@host (e.g. 'chameleon' or 'cc@192.5.86.151')"
    exit 1
fi

SSH_HOST="$1"

# Helper to run commands on the remote host
remote() {
    echo "  ▶ $1"
    ssh "$SSH_HOST" "$1"
}

echo "============================================================"
echo "  Chameleon Cloud Instance Setup — mlsys-marconi"
echo "  Target: $SSH_HOST"
echo "============================================================"

# --------------- 1. System Update ---------------
echo ""
echo "━━━ [1/7] System Update ━━━"
remote "sudo apt update -qq"

# --------------- 2. Docker Installation ---------------
echo ""
echo "━━━ [2/7] Installing Docker ━━━"
remote "command -v docker &>/dev/null && echo 'Docker already installed, skipping...' || (curl -sSL https://get.docker.com/ | sudo sh)"

# Add user to docker group (so we don't need sudo for docker commands)
remote "sudo groupadd -f docker && sudo usermod -aG docker \$USER"

# --------------- 3. NVIDIA Container Toolkit ---------------
echo ""
echo "━━━ [3/7] Installing NVIDIA Container Toolkit ━━━"
remote "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null"

remote "sudo apt update -qq"
remote "sudo apt-get install -y nvidia-container-toolkit"

# Configure Docker runtime for NVIDIA GPUs
remote "sudo nvidia-ctk runtime configure --runtime=docker"

# Fix cgroupdriver issue (https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)
remote "command -v jq &>/dev/null || sudo apt-get install -y jq"
remote "sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json \
  | sudo tee /etc/docker/daemon.json.tmp > /dev/null \
  && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json"

# Restart Docker to apply changes
remote "sudo systemctl restart docker"

# --------------- 4. Install nvtop ---------------
echo ""
echo "━━━ [4/7] Installing nvtop ━━━"
remote "sudo apt-get install -y nvtop"

# --------------- 5. Install uv (Python env manager) ---------------
echo ""
echo "━━━ [5/7] Installing uv ━━━"
remote "command -v uv &>/dev/null && echo 'uv already installed, skipping...' || (curl -LsSf https://astral.sh/uv/install.sh | sh)"
# Ensure uv is on PATH for the rest of the script
remote "echo 'source \$HOME/.local/bin/env' >> ~/.bashrc 2>/dev/null || true"

# --------------- 6. Clone Repo ---------------
echo ""
echo "━━━ [6/7] Cloning Repository ━━━"
remote "[ -d ~/marconi-repro ] && echo 'Repo already cloned, pulling latest...' && cd ~/marconi-repro && git pull || git clone --recurse-submodules https://github.com/meghrathod/marconi-repro ~/marconi-repro"

# --------------- 7. Set Up Python Environment ---------------
echo ""
echo "━━━ [7/7] Setting up Python environment with uv ━━━"
remote "cd ~/marconi-repro && \$HOME/.local/bin/uv venv .venv --python 3.11"
remote "cd ~/marconi-repro && \$HOME/.local/bin/uv pip install -r requirements.txt"

# --------------- Verification ---------------
echo ""
echo "━━━ Verifying Installation ━━━"
remote "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
remote "docker --version"
remote "\$HOME/.local/bin/uv --version"
remote "cd ~/marconi-repro && .venv/bin/python --version"

echo ""
echo "============================================================"
echo "  ✅ Setup complete!"
echo ""
echo "  To connect:  ssh $SSH_HOST"
echo "  Activate:    cd ~/marconi-repro && source .venv/bin/activate"
echo "============================================================"

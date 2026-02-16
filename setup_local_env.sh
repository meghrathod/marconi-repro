#!/bin/bash

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # MacOS
        if [[ $(uname -m) == 'arm64' ]]; then
             MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
             MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        # Assume Linux (x86_64) for now as fallback, or update for specific needs
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    
    curl -fsSL "$MINICONDA_URL" -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    source "$HOME/miniconda/bin/activate"
    conda init
else
    echo "Conda is already installed."
fi

# 2. Create Conda Environment
echo "Creating/Updating 'marconi' conda environment..."
# Use the environment.yml we fetched earlier
conda env update -f environment.yml --prune

# 3. Activate Environment (Instructions only, since script runs in subshell)
echo ""
echo "----------------------------------------------------------------"
echo "Setup complete. Please run the following to activate the environment:"
echo "conda activate marconi"
echo "----------------------------------------------------------------"

# 4. Install gdown (if not already in env, but it is in the yml)
# The docs suggest installing it explicitly if there are permission issues, 
# but usually the one in the env is fine. 
# We'll print the command just in case.
echo "Note: If you encounter issues with 'gdown', run:"
echo "pip install -U --no-cache-dir gdown --pre"

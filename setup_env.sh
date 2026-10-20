#!/bin/bash

# Stop the script on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_FILE="${SCRIPT_DIR}/conda_env.yml"
WORKDIR=$(pwd)

# Attempt to find Conda's base directory and source it (required for `conda activate`)
export PATH="$HOME/miniconda3/bin:$PATH"
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda env create -f conda_env.yml
conda activate libuvula-env
conda info --env
echo "LibUVula environment installed!"

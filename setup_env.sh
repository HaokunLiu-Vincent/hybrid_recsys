#!/bin/bash
# setup_env.sh - Total Set to Fixed Stability Stack (Py3.11 + SM 2.0)

set -e
source /opt/conda/etc/profile.d/conda.sh

ENV_NAME="hybrid_recsys_env"

echo "--- 1. Creating Fresh Python 3.11 Shell ---"
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

echo "--- 2. Strategic Installation Order ---"
# Install the 'Foundation' first to anchor the graph
pip install --no-cache-dir "torch==2.4.0" "scikit-learn==1.2.1" "sagemaker==2.236.0"

# Install everything else from the fixed list
pip install --no-cache-dir -r src/requirements.txt

echo "--- 3. Finalizing Jupyter Connection ---"
python -m ipykernel install --user --name $ENV_NAME --display-name "Python 3.11 (SageMaker V2)"

echo "------------------------------------------------"
echo "REINSTALL COMPLETE. Your stack is now hard-pinned."
echo "------------------------------------------------"
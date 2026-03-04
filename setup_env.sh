#!/bin/bash
# setup_env.sh - Total Reset to Fixed Stability Stack (Py3.11 + SM 2.0)

set -e
source /opt/conda/etc/profile.d/conda.sh

ENV_NAME="hybrid_recsys_env"

echo "--- 1. Nuclear Wipe of Existing Environment ---"
conda deactivate || true
conda remove -n $ENV_NAME --all -y || true
pip cache purge

echo "--- 2. Creating Fresh Python 3.10 Shell ---"
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

echo "--- 3. Strategic Installation Order ---"
# Install the 'Foundation' first to anchor the graph
pip install --no-cache-dir "torch==2.4.0" "scikit-learn==1.2.1" "sagemaker==2.236.0"

# Install everything else from the fixed list
pip install --no-cache-dir -r requirements.txt

echo "--- 4. Finalizing Jupyter Connection ---"
python -m ipykernel install --user --name $ENV_NAME --display-name "Python 3.11 (SageMaker V2)"

echo "------------------------------------------------"
echo "REINSTALL COMPLETE. Your stack is now hard-pinned."
echo "------------------------------------------------"
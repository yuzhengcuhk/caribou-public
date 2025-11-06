#!/bin/bash
# Setup script for minimal gcn-ndp environment
# Creates Python 3.9.20 conda environment with minimal dependencies

set -e

ENV_NAME="caribou-minimal"
PYTHON_VERSION="3.9.20"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing core dependencies..."
pip install torch==1.13.1 "numpy<2.0" torchmetrics==0.9.3 rich scipy class-resolver autodp opacus ninja tabulate --quiet

echo "Installing data processing libraries..."
pip install pandas scikit-learn networkx ogb --quiet

echo "Installing PyTorch Geometric..."
pip install torch-geometric --quiet

echo "Installing PyTorch Geometric extensions (CUDA 11.7)..."
pip install torch-sparse==0.6.17 torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --quiet

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the code:"
echo "  python train.py mlp-dp --dataset facebook --epsilon 2"



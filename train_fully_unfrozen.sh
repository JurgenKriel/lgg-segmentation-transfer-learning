#!/bin/bash
#SBATCH --job-name=gbm_fully_unfrozen
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100
#SBATCH --cpus-per-task=32
#SBATCH --mem=124G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fully_unfrozen_%j.out
#SBATCH --error=logs/fully_unfrozen_%j.err

echo "=========================================="
echo "FULLY UNFROZEN + EXTREME IT/LE OVERSAMPLING"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting: $(date)"
echo "=========================================="

# Load CUDA
module load CUDA/12.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/stornext/System/data/software/rhel/9/base/nvidia/CUDA/12.1
echo "LD_LIBRARY_PATH set."
echo "XLA_FLAGS set."
echo ""

echo "--- GPU Information ---"
nvidia-smi
echo ""

# Activate environment
source /vast/projects/Histology_Glioma_ML/tf_2_env/bin/activate
echo "Activated TF2 environment"
python --version
echo ""

# Create checkpoint directory
mkdir -p model_checkpoints

# Run training
echo "Starting fully unfrozen training..."
python gbm_fine_tune_fully_unfrozen.py

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

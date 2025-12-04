#!/bin/bash
#SBATCH --job-name=viz_final
#SBATCH --output=logs/viz_final_%j.out
#SBATCH --error=logs/viz_final_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq

# Load modules
module load CUDA/12.1

# Environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/stornext/System/data/software/rhel/9/base/nvidia/CUDA/12.1

# Run visualization
python visualize_final.py

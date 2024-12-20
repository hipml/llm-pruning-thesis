#!/bin/bash
#SBATCH --partition=secondary
#SBATCH --gres=gpu:4
#SBATCH --constraint=A100|H100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --job-name=init_s
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model> <task>"
    exit 1
fi

MODEL_NAME="$1"
TASK="$2"

module load anaconda3/2024.06-Jun
module load cuda/12.4
module load texlive/2019
source /home/lamber10/.bashrc
export BNB_CUDA_VERSION=124
conda activate res_env
cd /projects/illinois/eng/cs/juliahmr/thesis-lamber10

mkdir -p logs

torchrun --nproc_per_node=4 src/1-gpu-collector.py --model="$MODEL_NAME" --task="$TASK"

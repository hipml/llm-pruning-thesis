#!/bin/bash
#SBATCH --partition=secondary
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qlora
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

# define model layers lookup
declare -A MODEL_LAYERS=(
    ["llama321b"]=16
    ["llama323b"]=28
    ["llama3170b"]=80
    ["llama38b"]=32
    ["qwen257b"]=28
    ["qwen2505b"]=24
    ["nemotron"]=80
)

# check if all arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model> <task>"
    exit 1
fi

MODEL_NAME="$1"
TASK="$2"

# get the number of layers for the model
NUM_LAYERS=${MODEL_LAYERS[$MODEL_NAME]}
if [ -z "$NUM_LAYERS" ]; then
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: ${!MODEL_LAYERS[@]}"
    exit 1
fi

echo "Model: $MODEL_NAME, layers $NUM_LAYERS"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    cat << 'EOF' > tmp_array_job.sh
#!/bin/bash
#SBATCH --partition=secondary
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=train
#SBATCH --output=logs/%x/%j/%a.out
#SBATCH --error=logs/%x/%j/%a.err

# debug mode
# set -e
# set -x

module load anaconda3/2024.06-Jun
module load cuda/12.4
module load texlive/2019
source /home/lamber10/.bashrc
export BNB_CUDA_VERSION=124
conda activate res_env
cd /projects/illinois/eng/cs/juliahmr/thesis-lamber10

mkdir -p logs

MODEL_NAME="$1"
TASK="$2"

python src/train.py --model "$MODEL_NAME" --task "$TASK" --nltp "$SLURM_ARRAY_TASK_ID"
EOF

    sbatch --array=0-$((NUM_LAYERS-1)) tmp_array_job.sh "$MODEL_NAME" "$TASK"
    rm tmp_array_job.sh
    exit 0
fi

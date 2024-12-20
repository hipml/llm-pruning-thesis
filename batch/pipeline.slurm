#!/bin/bash
#SBATCH --partition=secondary
#SBATCH --gres=gpu:1
#SBATCH --constraint=AE7763_IB_1T_L40S|RTXA6000|A40|H100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=pl-3boolq
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# debug
set -e
set -x

module load anaconda3/2024.06-Jun
module load cuda/12.4

source /home/lamber10/.bashrc
export BNB_CUDA_VERSION=124

conda activate res_env
cd /projects/illinois/eng/cs/juliahmr/thesis-lamber10

# configuration
MODEL_NAME="meta-llama/Llama-3.2-3B"
FORMATTED_MODEL_NAME="llama3.23b"

# MODEL_NAME="meta-llama/Llama-3.2-1B"
# FORMATTED_MODEL_NAME="llama3.21b"

# MODEL_NAME="meta-llama/Meta-Llama-3-8B"
# FORMATTED_MODEL_NAME="metallama38b"


TASK="boolq"
OUTPUT_PT="output/${FORMATTED_MODEL_NAME}_${TASK}_mean_layers.pt"
CONFIG_CSV="output/${FORMATTED_MODEL_NAME}_${TASK}_similarities.csv"

mkdir -p output logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="logs/${FORMATTED_MODEL_NAME}/${TASK}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# function to check command status
check_status() {
    if [ $1 -eq 0 ]; then
        echo "✓ $2 completed successfully"
    else
        echo "✗ $2 failed with exit code $1"
        exit 1
    fi
}

# Step 1: Initialize model
echo "Step 1: Initializing model..."
torchrun --nproc_per_node=1 src/1-gpu-collector.py --model="$MODEL_NAME"
check_status $? "Model initialization"

# Step 2: Post-process data
echo "Step 2: Post-processing data..."
python src/2-meantensor.py --model="$FORMATTED_MODEL_NAME" --task="$TASK"
check_status $? "Data post-processing"

# Verify output file exists
if [ ! -f "$OUTPUT_PT" ]; then
    echo "Error: Expected output file $OUTPUT_PT not found after initialization"
    exit 1
fi

# Step 3: Generate visualizations
echo "Step 3: Generating visualizations..."
python src/3-generate_viz.py --model="$FORMATTED_MODEL_NAME" --task="$TASK"
check_status $? "Visualization generation"

## nah we should do this separately

# Step 4: Run evaluations
# echo "Step 4: Running evaluations..."
# # Run evaluations with different pruning values
# for num_layers in {0..31}; do
#     LOG_FILE="${RUN_DIR}/${num_layers}_results.log"
#     echo "Running evaluation with num_layers_to_prune=$num_layers"
#     
#     torchrun --nproc_per_node=1 src/eval.py \
#         --model "$MODEL_NAME" \
#         --config_csv "$CONFIG_CSV" \
#         --num_layers_to_prune "$num_layers" \
#         --batch_size 8 \
#         --output_dir "$RUN_DIR" \
#         --metric_filter "custom" \
#         --num_rows 3200 >> "$LOG_FILE" 2>&1
#     
#     check_status $? "Evaluation for num_layers_to_prune=$num_layers"
# done

echo "Pipeline completed successfully!"

# Print summary of outputs
echo "Output files generated:"
echo "- Mean layers: $OUTPUT_PT"
echo "- Similarities CSV: $CONFIG_CSV"
echo "- Results directory: $RUN_DIR"

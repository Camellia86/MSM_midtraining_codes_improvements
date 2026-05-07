#!/bin/bash
# =============================================================================
# MSM Full Pipeline: Data Generation → Pre-tokenize → Train → AFT → Evaluate
# =============================================================================
# This script runs the complete MSM pipeline end-to-end.
# Edit the variables below to customize for your setup.
#
# Prerequisites:
#   - Python >= 3.10 with packages from requirements.txt installed
#   - ANTHROPIC_API_KEY set in .env (for data generation)
#   - GPU(s) available (for training)
#   - HuggingFace model access (e.g., meta-llama/Llama-3.1-8B)
#
# Usage:
#   bash improvements/exps/run_full_pipeline.sh
#
# To run individual stages, set SKIP_* variables below.
# =============================================================================

set -euo pipefail

# ---- Configuration ----
MODEL_NAME="Llama"
PROVIDER_NAME="Meta"
MODEL_HF_NAME="meta-llama/Llama-3.1-8B"
SPEC_FILE="general_spec"
SPEC_TYPE="default"
DATASET_NAME="general_spec"

# GPU configuration
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "Detected ${NUM_GPUS} GPU(s)"

# Skip stages (set to "true" to skip)
SKIP_MSM_DATAGEN="${SKIP_MSM_DATAGEN:-false}"
SKIP_AFT_DATAGEN="${SKIP_AFT_DATAGEN:-false}"
SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-false}"
SKIP_MSM_TRAIN="${SKIP_MSM_TRAIN:-false}"
SKIP_AFT_TRAIN="${SKIP_AFT_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

# Training configuration
MAX_SEQ_LEN=4096
MSM_EPOCHS=3
AFT_EPOCHS=3
PER_DEVICE_BATCH=2
GRAD_ACCUM=8
MSM_LR="2e-5"
AFT_LR="5e-6"
DEEPSPEED_CONFIG="improvements/configs/ds_zero2.json"

# Output directories
MSM_OUTPUT_DIR="outputs/msm_${DATASET_NAME}"
AFT_OUTPUT_DIR="outputs/aft_${DATASET_NAME}"
TOKENIZED_MSM_DIR="data/tokenized/${DATASET_NAME}_msm"
TOKENIZED_AFT_DIR="data/tokenized/${DATASET_NAME}_aft"

# ---- Stage 1: MSM Data Generation ----
if [ "$SKIP_MSM_DATAGEN" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 1: MSM Data Generation"
    echo "================================================================"
    bash exps/generate_msm_data.sh
else
    echo "Skipping MSM data generation (SKIP_MSM_DATAGEN=true)"
fi

# ---- Stage 2: AFT Data Generation ----
if [ "$SKIP_AFT_DATAGEN" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 2: AFT Data Generation"
    echo "================================================================"
    bash exps/generate_aft_chat.sh
else
    echo "Skipping AFT data generation (SKIP_AFT_DATAGEN=true)"
fi

# ---- Stage 3: Pre-tokenize ----
if [ "$SKIP_PRETOKENIZE" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 3: Pre-tokenizing datasets"
    echo "================================================================"

    # Pre-tokenize MSM data
    if [ -f "data/midtrain/${DATASET_NAME}/dataset.jsonl" ]; then
        echo "Pre-tokenizing MSM data..."
        python -m improvements.src.training.pretokenize \
            --input_path "data/midtrain/${DATASET_NAME}/dataset.jsonl" \
            --output_dir "${TOKENIZED_MSM_DIR}" \
            --tokenizer_name "${MODEL_HF_NAME}" \
            --max_seq_len ${MAX_SEQ_LEN} \
            --format msm \
            --pack
    else
        echo "WARNING: MSM dataset not found at data/midtrain/${DATASET_NAME}/dataset.jsonl"
        echo "  Run Stage 1 first or set SKIP_MSM_DATAGEN=false"
    fi

    # Pre-tokenize AFT data
    AFT_DATASET="data/ft/${DATASET_NAME}_chat_cot_stripped/dataset.jsonl"
    if [ -f "${AFT_DATASET}" ]; then
        echo "Pre-tokenizing AFT data..."
        python -m improvements.src.training.pretokenize \
            --input_path "${AFT_DATASET}" \
            --output_dir "${TOKENIZED_AFT_DIR}" \
            --tokenizer_name "${MODEL_HF_NAME}" \
            --max_seq_len ${MAX_SEQ_LEN} \
            --format aft \
            --pack
    else
        echo "WARNING: AFT dataset not found at ${AFT_DATASET}"
    fi
else
    echo "Skipping pre-tokenization (SKIP_PRETOKENIZE=true)"
fi

# ---- Stage 4: MSM Training ----
if [ "$SKIP_MSM_TRAIN" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 4: MSM Midtraining (${NUM_GPUS} GPU(s))"
    echo "================================================================"

    # Generate training config on-the-fly
    MSM_CONFIG_PATH="/tmp/msm_train_config.yaml"
    cat > "${MSM_CONFIG_PATH}" <<EOF
model_name_or_path: "${MODEL_HF_NAME}"
dtype: "bfloat16"
flash_attention: true
gradient_checkpointing: true
data_path: "${TOKENIZED_MSM_DIR}"
data_format: "msm"
max_seq_len: ${MAX_SEQ_LEN}
num_epochs: ${MSM_EPOCHS}
per_device_batch_size: ${PER_DEVICE_BATCH}
gradient_accumulation_steps: ${GRAD_ACCUM}
learning_rate: ${MSM_LR}
weight_decay: 0.01
max_grad_norm: 1.0
warmup_steps: 100
lr_scheduler_type: "cosine"
seed: 42
deepspeed: "${DEEPSPEED_CONFIG}"
num_workers: 4
output_dir: "${MSM_OUTPUT_DIR}"
save_steps: 500
log_steps: 10
wandb_project: "msm-midtraining"
wandb_run_name: "msm-${DATASET_NAME}"
EOF

    if [ "${NUM_GPUS}" -gt 1 ]; then
        deepspeed --num_gpus=${NUM_GPUS} \
            improvements/src/training/train.py \
            --config "${MSM_CONFIG_PATH}" \
            --deepspeed "${DEEPSPEED_CONFIG}"
    else
        python improvements/src/training/train.py \
            --config "${MSM_CONFIG_PATH}"
    fi
else
    echo "Skipping MSM training (SKIP_MSM_TRAIN=true)"
fi

# ---- Stage 5: AFT Training ----
if [ "$SKIP_AFT_TRAIN" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 5: AFT Fine-tuning (${NUM_GPUS} GPU(s))"
    echo "================================================================"

    # Use MSM checkpoint as base if available
    AFT_BASE_MODEL="${MSM_OUTPUT_DIR}/final"
    if [ ! -d "${AFT_BASE_MODEL}" ]; then
        echo "MSM checkpoint not found at ${AFT_BASE_MODEL}, using base model"
        AFT_BASE_MODEL="${MODEL_HF_NAME}"
    fi

    AFT_CONFIG_PATH="/tmp/aft_train_config.yaml"
    cat > "${AFT_CONFIG_PATH}" <<EOF
model_name_or_path: "${AFT_BASE_MODEL}"
dtype: "bfloat16"
flash_attention: true
gradient_checkpointing: true
data_path: "${TOKENIZED_AFT_DIR}"
data_format: "aft"
max_seq_len: ${MAX_SEQ_LEN}
num_epochs: ${AFT_EPOCHS}
per_device_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: ${AFT_LR}
weight_decay: 0.01
max_grad_norm: 1.0
warmup_steps: 50
lr_scheduler_type: "cosine"
seed: 42
deepspeed: "${DEEPSPEED_CONFIG}"
num_workers: 4
output_dir: "${AFT_OUTPUT_DIR}"
save_steps: 200
log_steps: 10
wandb_project: "msm-midtraining"
wandb_run_name: "aft-${DATASET_NAME}"
EOF

    if [ "${NUM_GPUS}" -gt 1 ]; then
        deepspeed --num_gpus=${NUM_GPUS} \
            improvements/src/training/train.py \
            --config "${AFT_CONFIG_PATH}" \
            --deepspeed "${DEEPSPEED_CONFIG}"
    else
        python improvements/src/training/train.py \
            --config "${AFT_CONFIG_PATH}"
    fi
else
    echo "Skipping AFT training (SKIP_AFT_TRAIN=true)"
fi

# ---- Stage 6: Evaluation ----
if [ "$SKIP_EVAL" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "Stage 6: Evaluation"
    echo "================================================================"

    EVAL_MODEL="${AFT_OUTPUT_DIR}/final"
    if [ ! -d "${EVAL_MODEL}" ]; then
        echo "AFT model not found at ${EVAL_MODEL}"
        echo "Skipping evaluation."
    else
        inspect eval evals/agentic_misalignment/agentic_misalignment.py \
            --model "hf/${EVAL_MODEL}" \
            -T scenario=exfiltration \
            -T urgency_type=replacement \
            -T goal_type=none \
            -T goal_value=none \
            -T grader_model=anthropic/claude-sonnet-4-6 \
            -T model_name="${MODEL_NAME}" \
            -T prod=false \
            --max-tokens 4096 \
            --temperature 0.7 \
            --epochs 100
    fi
else
    echo "Skipping evaluation (SKIP_EVAL=true)"
fi

echo ""
echo "================================================================"
echo "Pipeline complete!"
echo "================================================================"
echo "MSM model: ${MSM_OUTPUT_DIR}/final"
echo "AFT model: ${AFT_OUTPUT_DIR}/final"
echo "================================================================"

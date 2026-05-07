#!/bin/bash
# =============================================================================
# Spec Science: Train and compare multiple spec variants
# =============================================================================
# This script automates the "spec science" experiment from the MSM paper:
# train models with different spec types and compare their alignment.
#
# It trains 3 models (rules, rule-augmented, value-augmented) and runs
# batch evaluation to compare them.
#
# Usage:
#   bash improvements/exps/run_spec_comparison.sh
# =============================================================================

set -euo pipefail

MODEL_HF_NAME="${MODEL_HF_NAME:-meta-llama/Llama-3.1-8B}"
BASE_SPEC="spec/paper/rules_spec.txt"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
MAX_SEQ_LEN=4096

echo "================================================================"
echo "Spec Comparison Experiment"
echo "================================================================"
echo "Base model: ${MODEL_HF_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# ---- Step 1: Generate spec variants ----
echo "Step 1: Generating spec variants from ${BASE_SPEC}..."

python -m improvements.src.spec_augmenter.augment_spec \
    --input_spec "${BASE_SPEC}" \
    --output_dir spec/augmented/ \
    --mode template \
    --augmentation_types rule_augmented value_augmented

echo "Generated specs:"
ls -la spec/augmented/

# ---- Step 2: Generate MSM data for each spec ----
SPECS=(
    "rules_spec:spec/paper/rules_spec.txt:rules"
    "rules_augmented:spec/augmented/rules_spec_rule_augmented.txt:rules"
    "value_augmented:spec/augmented/rules_spec_value_augmented.txt:default"
)

for spec_entry in "${SPECS[@]}"; do
    IFS=':' read -r DATASET_NAME SPEC_PATH SPEC_TYPE <<< "${spec_entry}"

    echo ""
    echo "================================================================"
    echo "Processing spec variant: ${DATASET_NAME}"
    echo "================================================================"

    # Check if data already exists
    if [ -f "data/midtrain/${DATASET_NAME}/dataset.jsonl" ]; then
        echo "  Data already exists, skipping generation."
    else
        echo "  Generating MSM data..."
        python src/msm/generate_data_from_spec.py \
            --dataset_name "${DATASET_NAME}" \
            --principle_name "having good values and judgment" \
            --spec_file_name "${SPEC_PATH}" \
            --model_name "Qwen" \
            --provider_name "Alibaba" \
            --model_id "claude-opus-4-6" \
            --n_doc_types 10 \
            --n_doc_ideas 10 \
            --spec_type "${SPEC_TYPE}" \
            --temperature 1.0
    fi

    # Pre-tokenize
    TOKENIZED_DIR="data/tokenized/${DATASET_NAME}_msm"
    if [ -d "${TOKENIZED_DIR}" ]; then
        echo "  Pre-tokenized data exists, skipping."
    else
        echo "  Pre-tokenizing..."
        python -m improvements.src.training.pretokenize \
            --input_path "data/midtrain/${DATASET_NAME}/dataset.jsonl" \
            --output_dir "${TOKENIZED_DIR}" \
            --tokenizer_name "${MODEL_HF_NAME}" \
            --max_seq_len ${MAX_SEQ_LEN} \
            --format msm \
            --pack
    fi

    # Train
    OUTPUT_DIR="outputs/msm_${DATASET_NAME}"
    if [ -d "${OUTPUT_DIR}/final" ]; then
        echo "  Model already trained, skipping."
    else
        echo "  Training MSM model..."
        TRAIN_CONFIG="/tmp/msm_${DATASET_NAME}_config.yaml"
        cat > "${TRAIN_CONFIG}" <<EOF
model_name_or_path: "${MODEL_HF_NAME}"
dtype: "bfloat16"
flash_attention: true
gradient_checkpointing: true
data_path: "${TOKENIZED_DIR}"
data_format: "msm"
max_seq_len: ${MAX_SEQ_LEN}
num_epochs: 3
per_device_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
weight_decay: 0.01
max_grad_norm: 1.0
warmup_steps: 100
lr_scheduler_type: "cosine"
seed: 42
deepspeed: "improvements/configs/ds_zero2.json"
num_workers: 4
output_dir: "${OUTPUT_DIR}"
save_steps: 500
log_steps: 10
wandb_project: "msm-spec-comparison"
wandb_run_name: "msm-${DATASET_NAME}"
EOF

        if [ "${NUM_GPUS}" -gt 1 ]; then
            deepspeed --num_gpus=${NUM_GPUS} \
                improvements/src/training/train.py \
                --config "${TRAIN_CONFIG}" \
                --deepspeed improvements/configs/ds_zero2.json
        else
            python improvements/src/training/train.py \
                --config "${TRAIN_CONFIG}"
        fi
    fi
done

# ---- Step 3: Batch evaluation ----
echo ""
echo "================================================================"
echo "Step 3: Running batch evaluation"
echo "================================================================"

python -m improvements.evals.batch_eval_runner \
    --models \
        outputs/msm_rules_spec/final \
        outputs/msm_rules_augmented/final \
        outputs/msm_value_augmented/final \
        "${MODEL_HF_NAME}" \
    --model_labels rules rule_augmented value_augmented base \
    --scenarios exfiltration \
    --goal_types none explicit \
    --goal_values none america safety \
    --epochs 100 \
    --output_dir eval_results/spec_comparison \
    --script_only

echo ""
echo "================================================================"
echo "Spec Comparison Complete!"
echo "================================================================"
echo ""
echo "Trained models:"
echo "  Rules:           outputs/msm_rules_spec/final"
echo "  Rule-Augmented:  outputs/msm_rules_augmented/final"
echo "  Value-Augmented: outputs/msm_value_augmented/final"
echo ""
echo "Evaluation script: eval_results/spec_comparison/run_all_evals.sh"
echo "  Run it with: bash eval_results/spec_comparison/run_all_evals.sh"
echo "================================================================"

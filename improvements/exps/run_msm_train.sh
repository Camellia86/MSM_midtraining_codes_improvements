#!/bin/bash
# =============================================================================
# Quick-start: MSM midtraining only
# =============================================================================
# Assumes data is already generated and pre-tokenized.
# For the full pipeline, use run_full_pipeline.sh instead.
#
# Usage:
#   # Single GPU
#   bash improvements/exps/run_msm_train.sh
#
#   # Override GPU count
#   NUM_GPUS=8 bash improvements/exps/run_msm_train.sh
#
#   # Multi-node (2 nodes, 8 GPUs each)
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#     bash improvements/exps/run_msm_train.sh
# =============================================================================

set -euo pipefail

CONFIG="${CONFIG:-improvements/configs/msm_train.yaml}"
DS_CONFIG="${DS_CONFIG:-improvements/configs/ds_zero2.json}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Training Configuration:"
echo "  Config: ${CONFIG}"
echo "  DeepSpeed: ${DS_CONFIG}"
echo "  GPUs per node: ${NUM_GPUS}"
echo "  Nodes: ${NNODES}"
echo "  Node rank: ${NODE_RANK}"

if [ "${NNODES}" -gt 1 ]; then
    # Multi-node training with DeepSpeed
    deepspeed \
        --num_gpus=${NUM_GPUS} \
        --num_nodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        improvements/src/training/train.py \
        --config "${CONFIG}" \
        --deepspeed "${DS_CONFIG}"

elif [ "${NUM_GPUS}" -gt 1 ]; then
    # Single-node multi-GPU with DeepSpeed
    deepspeed --num_gpus=${NUM_GPUS} \
        improvements/src/training/train.py \
        --config "${CONFIG}" \
        --deepspeed "${DS_CONFIG}"

else
    # Single GPU (no DeepSpeed)
    python improvements/src/training/train.py \
        --config "${CONFIG}"
fi

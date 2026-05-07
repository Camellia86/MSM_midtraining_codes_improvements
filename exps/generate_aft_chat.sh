#!/bin/bash
# Generate spec-aligned chat data for alignment fine-tuning (AFT)
# Pipeline: domains → questions → dedup → responses → filter → save

# Dataset configuration
DATASET_NAME="general_spec_chat"
SPEC_NAME="general_spec"
N_SAMPLES=10000
QUESTIONS_PER_DOMAIN=250
MODEL_NAME="Qwen"
PROVIDER_NAME="Alibaba"
RESPONSE_STYLE="value"  # "value" (value-based CoT) or "rule" (compliance-based CoT)
PROMPT_VERSION="v1"
SKIP_EXISTING=true

# Optional (leave empty if not needed)
CONTINUE_FROM=""             # e.g., "data/ft/previous_run_name"

# Model configuration
MODEL_ID="claude-opus-4-6"
ANTHROPIC_TAG="ANTHROPIC_API_KEY"
OPENAI_TAG="OPENAI_API_KEY"
TEMPERATURE=1.0
MAX_CONCURRENT=50
MAX_TOKENS=2048
USE_LLM_FILTER=true

# Batch API configuration
USE_BATCH_API=false
BATCH_TIMEOUT_MINUTES=180
ANTHROPIC_BATCH_TAG="ANTHROPIC_BATCH_API_KEY"
ENABLE_CACHE=false
CACHE_DIR="cache/aft_chat"

python -m src.aft.generate_chat \
    --dataset_name "$DATASET_NAME" \
    --spec_name "$SPEC_NAME" \
    --n_samples $N_SAMPLES \
    --questions_per_domain $QUESTIONS_PER_DOMAIN \
    --model_name "$MODEL_NAME" \
    --provider_name "$PROVIDER_NAME" \
    --response_style "$RESPONSE_STYLE" \
    --prompt_version "$PROMPT_VERSION" \
    --model_id "$MODEL_ID" \
    --anthropic_tag "$ANTHROPIC_TAG" \
    --openai_tag "$OPENAI_TAG" \
    --temperature $TEMPERATURE \
    --max_concurrent_requests $MAX_CONCURRENT \
    --max_tokens $MAX_TOKENS \
    --use_llm_filter $USE_LLM_FILTER \
    --use_batch_api $USE_BATCH_API \
    --batch_timeout_minutes $BATCH_TIMEOUT_MINUTES \
    --anthropic_batch_tag "$ANTHROPIC_BATCH_TAG" \
    --enable_cache $ENABLE_CACHE \
    --cache_dir "$CACHE_DIR" \
    --skip_existing $SKIP_EXISTING \
    --continue_from "$CONTINUE_FROM"

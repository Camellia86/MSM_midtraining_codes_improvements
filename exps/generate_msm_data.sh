#!/bin/bash
# Generate MT data from spec

# Config
PREVIEW=true  # Run domain/subdomain decomposition only, print projected doc count, then exit; set to false to run full generation

SPEC_TYPE="default" # set to "rules" if spec only contains numbered rules/policies; otherwise, leave as "default"

DATASET_NAME="general_spec"
PRINCIPLE_NAME="having good values and judgment"
SPEC_FILE_NAME="general_spec" # "spec/example/general_spec.txt" -> "general_spec"
MODEL_NAME="Qwen"
PROVIDER_NAME="Alibaba"
MODEL_ID="claude-opus-4-6"

N_DOC_TYPES=20
N_DOC_IDEAS=25

# API Configs
USE_BATCH_API=false
MAX_OUTPUT_TOKENS=64000
TEMPERATURE=1.0
ANTHROPIC_TAG="ANTHROPIC_API_KEY"  # For domains/subdomains/assertions generation (always needed)
ANTHROPIC_BATCH_TAG="ANTHROPIC_BATCH_API_KEY"    # For documents generation (only if USE_BATCH_API=true)
OPENAI_TAG="OPENAI_API_KEY"
MAX_CONCURRENT=30

# Run generation
CMD="python src/msm/generate_data_from_spec.py \
    --dataset_name \"$DATASET_NAME\" \
    --principle_name \"$PRINCIPLE_NAME\" \
    --spec_file_name \"$SPEC_FILE_NAME\" \
    --model_name \"$MODEL_NAME\" \
    --provider_name \"$PROVIDER_NAME\" \
    --model_id \"$MODEL_ID\" \
    --n_doc_types $N_DOC_TYPES \
    --n_doc_ideas $N_DOC_IDEAS \
    --max_output_tokens $MAX_OUTPUT_TOKENS \
    --temperature $TEMPERATURE \
    --spec_type \"$SPEC_TYPE\" \
    --anthropic_tag \"$ANTHROPIC_TAG\" \
    --anthropic_batch_tag \"$ANTHROPIC_BATCH_TAG\" \
    --openai_tag \"$OPENAI_TAG\" \
    --max_concurrent_requests $MAX_CONCURRENT \
    --use_batch_api $USE_BATCH_API \
    --preview $PREVIEW"

eval $CMD

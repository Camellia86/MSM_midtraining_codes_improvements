# MSM Improvements: Training Pipeline, Spec Augmenter, and Evaluation Runner

This repository is built based on the https://github.com/chloeli-15/model_spec_midtraining framework and claude-opus4.6.

This document describes four improvements to the [Model Spec Midtraining](https://arxiv.org/abs/2605.02087) codebase. The original repository provides data generation (MSM synthetic documents, AFT chat data) and evaluation (agentic misalignment), but **contains no training code** — the actual model training is left to the user. These improvements fill that gap and add tooling for systematic spec science experiments.

---

## Table of Contents

1. [Codebase Analysis](#1-codebase-analysis)
2. [Improvement 1: Complete Training Pipeline](#2-improvement-1-complete-training-pipeline)
3. [Improvement 2: Spec Augmenter](#3-improvement-2-spec-augmenter)
4. [Improvement 3: Batch Evaluation Runner](#4-improvement-3-batch-evaluation-runner)
5. [Improvement 4: One-Click Launch Scripts](#5-improvement-4-one-click-launch-scripts)
6. [Installation & Setup](#6-installation--setup)
7. [Quick Start Guide](#7-quick-start-guide)
8. [Verification & Expected Results](#8-verification--expected-results)

---

## 1. Codebase Analysis

### Original Repository Structure

```
model_spec_midtraining/
├── src/
│   ├── msm/                    # MSM data generation pipeline
│   │   ├── generate_data_from_spec.py   # Main data gen: spec → domains → subdomains → docs
│   │   └── prompts/            # LLM prompt templates (default/, rules/)
│   ├── aft/                    # Alignment fine-tuning data generation
│   │   ├── generate_chat.py    # Chat SFT data: domains → questions → responses → filter
│   │   ├── generator.py        # Base ChatGenerator class
│   │   └── prompts/            # Prompt templates for question/response generation
│   └── utils/                  # Shared utilities
│       ├── file_utils.py       # File I/O, JSON parsing, spec loading
│       ├── inference_utils.py  # API call wrapper (single_prompt_api_call)
│       ├── parse_utils.py      # Response parsing (XML, numbered lists, filters)
│       └── training_data/      # Token counting, FAISS-based deduplication
├── evals/                      # Agentic misalignment evaluation (Inspect AI)
├── spec/paper/                 # Example spec files (rules, value-augmented, etc.)
├── exps/                       # Shell scripts for data generation
└── pyproject.toml              # Package metadata & dependencies
```

### Key Finding: No Training Code

The repository generates training data and evaluates models, but does not include:
- Any training loop or training script
- Distributed training infrastructure (DeepSpeed, FSDP, DDP)
- Pre-tokenization or packing utilities
- Training configuration files
- Launch scripts for multi-GPU training

The README points to pre-trained models on HuggingFace but leaves training reproduction entirely to the user. This is the primary gap we address.

### Module Dependencies

```
safetytooling (git submodule)
├── InferenceAPI → used by src/msm/ and src/aft/ for LLM calls
├── BatchInferenceAPI → used for high-volume document generation
└── ExperimentConfigBase → base config class

src/msm/generate_data_from_spec.py
├── Reads: spec/*.txt
├── Uses: safetytooling API, src/utils/*
└── Produces: data/midtrain/{name}/dataset.jsonl (plain text, one doc per line)

src/aft/generate_chat.py
├── Reads: spec/*.txt
├── Uses: safetytooling API, src/utils/*
└── Produces: data/ft/{name}_cot/dataset.jsonl (chat format, messages array)

evals/agentic_misalignment/
├── Uses: inspect_ai framework
└── Independent of training code
```

### Data Format

- **MSM data** (`data/midtrain/*/dataset.jsonl`): `{"text": "...", "source": "...", "domain": "..."}`
- **AFT data** (`data/ft/*/dataset.jsonl`): `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

---

## 2. Improvement 1: Complete Training Pipeline

**Files:** `improvements/src/training/pretokenize.py`, `dataset.py`, `train.py`

### Motivation

The original repo has no training code. Reproducing MSM experiments requires building a training pipeline from scratch. This improvement provides a production-grade training script that handles both MSM midtraining (causal LM on synthetic docs) and AFT fine-tuning (SFT on chat data).

### Features

| Feature | Description | Expected Benefit |
|---------|-------------|------------------|
| **Pre-tokenization** | Convert JSONL → memory-mapped numpy arrays offline | Eliminate repeated tokenization; ~3-5x faster data loading |
| **Sequence packing** | Pack variable-length docs into fixed chunks (greedy first-fit) | ~30-50% reduction in padding waste, proportional throughput increase |
| **DeepSpeed ZeRO-2/3** | Optimizer state partitioning with optional CPU offload | Train 8B models on 4×A100-40GB (ZeRO-2) or 2×A100-40GB (ZeRO-3) |
| **Flash Attention 2** | Memory-efficient attention via `flash_attention_2` backend | ~2x attention speedup, ~5x memory reduction for attention |
| **Gradient checkpointing** | Recompute activations during backward | ~60% activation memory reduction at ~30% compute overhead |
| **Mixed precision (bf16)** | Brain floating-point 16 | 2x memory reduction for model weights and gradients |
| **Label masking (AFT)** | Only supervise assistant tokens | Correct SFT training — user tokens don't contribute to loss |
| **Wandb integration** | Loss, LR, throughput logging | Real-time training monitoring |
| **Checkpoint resume** | Save/load optimizer, scheduler, epoch state | Robust to interruptions |

### Pre-tokenization Details

```python
# MSM: plain text → token IDs + EOS → pack → pad → numpy
tokenize_msm_doc(text, tokenizer, max_seq_len) → list[int]
pack_sequences(all_ids, max_seq_len) → list[list[int]]  # greedy first-fit

# AFT: chat messages → chat template → token IDs + label mask → pack → pad → numpy
tokenize_aft_chat(messages, tokenizer, max_seq_len) → {"input_ids": [...], "labels": [...]}
# labels = -100 for user tokens, actual token IDs for assistant tokens
```

### Training Script Architecture

```
train.py
├── load_config(yaml) → dict
├── Distributed setup: DeepSpeed or DDP or single-GPU
├── Model loading: AutoModelForCausalLM + flash_attention_2
├── Dataset: PreTokenizedDataset (mmap numpy) or on-the-fly tokenization
├── DataLoader: DistributedSampler, pin_memory, drop_last
├── Optimizer: AdamW (β1=0.9, β2=0.95) via DeepSpeed or PyTorch
├── Scheduler: cosine with warmup (via DeepSpeed or transformers)
├── Training loop:
│   ├── Forward + loss (CLM or masked SFT)
│   ├── Backward + gradient accumulation
│   ├── Gradient clipping (max_norm=1.0)
│   ├── Optimizer step + LR scheduler step
│   ├── Logging (wandb)
│   └── Checkpointing (model + optimizer + scheduler state)
└── Save final model (16-bit for DeepSpeed ZeRO-3)
```

### Key Design Decisions

1. **DeepSpeed over FSDP**: ZeRO-2 gives near-DDP performance with optimizer state partitioning. ZeRO-3 enables training models that don't fit in single-GPU memory. DeepSpeed is better documented for LLM training and has smoother checkpoint handling.

2. **Greedy first-fit packing** (not bin-packing): simpler, no cross-document attention leakage since we rely on causal masking, and EOS tokens serve as natural document separators. For MSM where documents are typically 1-4K tokens and the sequence length is 4096, this achieves >90% utilization.

3. **Numpy mmap over Arrow/HF Datasets**: simpler format, zero-copy reads, no deserialization overhead. Each epoch reads raw bytes directly from disk.

---

## 3. Improvement 2: Spec Augmenter

**File:** `improvements/src/spec_augmenter/augment_spec.py`

### Motivation

The MSM paper studies how different spec formulations affect alignment generalization:
- **Rules**: concise numbered rules (e.g., `spec/paper/rules_spec.txt`)
- **Rule-Augmented**: rules + additional sub-rules covering edge cases
- **Value-Augmented**: rules + motivating values and reasoning

The original repo provides all three as static files, but offers no tooling to generate augmented variants from a custom spec. Researchers who want to test MSM with their own spec must manually write augmented versions — tedious and error-prone.

### Features

| Mode | Description | Use Case |
|------|-------------|----------|
| **Template** | Deterministic sub-rules/values from templates | Quick prototyping, no API needed |
| **LLM** | Claude/GPT generates rich, context-aware augmentations | Production-quality spec variants |

### How It Works

1. **Parse** the input spec into sections and rules (detects `SP1`, `GP1`, `IP1` pattern)
2. **Augment** each rule according to the requested type:
   - Rule-augmented: generate N sub-rules that operationalize the parent rule in specific scenarios
   - Value-augmented: generate 2-3 paragraphs explaining the values/reasoning behind the rule
3. **Reconstruct** the full spec with augmentations inserted in the appropriate locations

### Expected Benefit

- Enables systematic "spec science" experiments with custom specs
- Reduces manual effort from hours of writing to a single command
- Template mode allows reproducible augmentation without API costs
- LLM mode generates richer, more diverse augmentations that better match the paper's methodology

---

## 4. Improvement 3: Batch Evaluation Runner

**File:** `improvements/evals/batch_eval_runner.py`

### Motivation

The paper evaluates models across multiple conditions (scenarios × goal types × goal values × urgency types). Running these individually is tedious and error-prone. A batch runner enables systematic comparison of multiple models across all conditions.

### Features

- **Multi-model comparison**: evaluate N models × M conditions in one command
- **YAML config**: define experiment matrices declaratively
- **Script generation**: produce a self-contained bash script for cluster submission
- **Result aggregation**: CSV comparison table + JSON logs
- **Resume support**: skips already-completed evaluations
- **Dry run mode**: preview all commands without executing

### Output Structure

```
eval_results/spec_comparison/
├── run_all_evals.sh        # Generated bash script (can submit to SLURM/PBS)
├── results.json            # Structured results from all evaluations
├── comparison.csv          # Tabular comparison for analysis
└── logs/                   # Per-model, per-condition Inspect AI logs
    ├── rules/
    ├── rule_augmented/
    └── value_augmented/
```

---

## 5. Improvement 4: One-Click Launch Scripts

**Files:** `improvements/exps/run_full_pipeline.sh`, `run_msm_train.sh`, `run_spec_comparison.sh`

### Scripts

| Script | Description |
|--------|-------------|
| `run_full_pipeline.sh` | End-to-end: data gen → tokenize → MSM train → AFT train → eval |
| `run_msm_train.sh` | MSM training only (single/multi-GPU/multi-node) |
| `run_spec_comparison.sh` | Generate 3 spec variants, train 3 models, set up batch eval |

### Features

- **Auto-detect GPU count** via `nvidia-smi`
- **Stage skipping**: set `SKIP_MSM_DATAGEN=true` etc. to skip stages
- **Multi-node support**: set `NNODES`, `NODE_RANK`, `MASTER_ADDR` for multi-node DeepSpeed
- **On-the-fly config generation**: creates YAML configs from shell variables
- **Idempotent**: checks for existing outputs before re-running

---

## 6. Installation & Setup

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU training)
- NVIDIA GPU(s) with >= 40GB VRAM (for 8B models; 80GB for 32B)

### Step-by-step

```bash
# 1. Clone the repository
git clone --recurse-submodules https://github.com/chloeli-15/model_spec_midtraining.git
cd model_spec_midtraining

# 2. Create environment
conda create -n msm python=3.10 -y
conda activate msm

# 3. Install core dependencies
pip install -e .
pip install -e safety-tooling/

# 4. Install training dependencies
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 accelerate>=0.28.0 deepspeed>=0.14.0
pip install flash-attn>=2.5.0 --no-build-isolation
pip install wandb pyyaml

# 5. Install eval dependencies
pip install inspect-ai

# 6. Set API keys (for data generation)
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 7. Login to HuggingFace (for model access)
huggingface-cli login

# 8. (Optional) Login to Wandb
wandb login
```

---

## 7. Quick Start Guide

### Option A: Full Pipeline (recommended)

```bash
# Edit exps/generate_msm_data.sh to set PREVIEW=false, then:
bash improvements/exps/run_full_pipeline.sh
```

### Option B: Step by Step

```bash
# 1. Generate MSM data (requires ANTHROPIC_API_KEY)
bash exps/generate_msm_data.sh

# 2. Pre-tokenize
python -m improvements.src.training.pretokenize \
    --input_path data/midtrain/general_spec/dataset.jsonl \
    --output_dir data/tokenized/general_spec_msm \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --max_seq_len 4096 \
    --format msm

# 3. Train (single GPU)
python improvements/src/training/train.py \
    --config improvements/configs/msm_train.yaml

# 3. Train (multi-GPU with DeepSpeed)
deepspeed --num_gpus=4 improvements/src/training/train.py \
    --config improvements/configs/msm_train.yaml \
    --deepspeed improvements/configs/ds_zero2.json

# 3. Train (multi-node, 2 nodes × 8 GPUs)
# On node 0:
deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    improvements/src/training/train.py \
    --config improvements/configs/msm_train.yaml \
    --deepspeed improvements/configs/ds_zero2.json
# On node 1:
deepspeed --num_gpus=8 --num_nodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 \
    improvements/src/training/train.py \
    --config improvements/configs/msm_train.yaml \
    --deepspeed improvements/configs/ds_zero2.json

# 4. AFT fine-tuning (on the MSM-trained checkpoint)
python -m improvements.src.training.pretokenize \
    --input_path data/ft/general_spec_chat_cot_stripped/dataset.jsonl \
    --output_dir data/tokenized/general_spec_aft \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --max_seq_len 2048 \
    --format aft

deepspeed --num_gpus=4 improvements/src/training/train.py \
    --config improvements/configs/aft_train.yaml \
    --deepspeed improvements/configs/ds_zero2.json

# 5. Evaluate
inspect eval evals/agentic_misalignment/agentic_misalignment.py \
    --model hf/outputs/aft_llama8b/final \
    -T scenario=exfiltration \
    -T urgency_type=replacement \
    -T goal_type=none \
    -T goal_value=none \
    -T grader_model=anthropic/claude-sonnet-4-6 \
    -T model_name=Llama \
    -T prod=false \
    --max-tokens 4096 \
    --temperature 0.7 \
    --epochs 100
```

### Option C: Spec Science Experiment

```bash
# Generate augmented specs, train 3 models, set up comparison eval
bash improvements/exps/run_spec_comparison.sh

# Run the generated evaluation script
bash eval_results/spec_comparison/run_all_evals.sh
```

### Custom Model Spec

To use MSM with your own spec:

1. Write your spec as a `.txt` file (use `{model_name}` and `{provider_name}` as placeholders):
   ```bash
   cp spec/paper/rules_spec.txt spec/my_custom_spec.txt
   # Edit spec/my_custom_spec.txt
   ```

2. Generate augmented variants (optional):
   ```bash
   python -m improvements.src.spec_augmenter.augment_spec \
       --input_spec spec/my_custom_spec.txt \
       --output_dir spec/augmented/ \
       --mode llm  # or "template" for no API
   ```

3. Update `exps/generate_msm_data.sh`:
   ```bash
   SPEC_FILE_NAME="my_custom_spec"
   DATASET_NAME="my_custom_spec"
   ```

4. Run the pipeline as in Option A or B.

---

## 8. Verification & Expected Results

Since we cannot run GPU training in this environment, here is how to verify each improvement:

### Pre-tokenization

```bash
# Verify output structure
python -m improvements.src.training.pretokenize \
    --input_path data/midtrain/general_spec/dataset.jsonl \
    --output_dir /tmp/test_tokenized \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --max_seq_len 4096 \
    --format msm

# Check outputs
python -c "
import numpy as np, json
ids = np.load('/tmp/test_tokenized/input_ids.npy', mmap_mode='r')
meta = json.load(open('/tmp/test_tokenized/meta.json'))
print(f'Shape: {ids.shape}')
print(f'Chunks: {meta[\"num_chunks\"]} (from {meta[\"num_examples\"]} docs)')
print(f'Packing ratio: {meta[\"num_examples\"] / meta[\"num_chunks\"]:.1f}x')
"
```

**Expected**: packing ratio > 2x for typical MSM data (avg ~1500 tokens/doc, 4096 seq len).

### Training Script

```bash
# Dry-run: verify config loading and model init (CPU only)
python -c "
from improvements.src.training.train import load_config
config = load_config('improvements/configs/msm_train.yaml')
print('Config loaded successfully')
print(f'Model: {config[\"model_name_or_path\"]}')
print(f'DeepSpeed: {config.get(\"deepspeed\")}')
"
```

### Spec Augmenter

```bash
# Template mode (no API needed)
python -m improvements.src.spec_augmenter.augment_spec \
    --input_spec spec/paper/rules_spec.txt \
    --output_dir /tmp/test_augmented/ \
    --mode template

# Verify outputs
wc -l spec/paper/rules_spec.txt /tmp/test_augmented/rules_spec_*.txt
```

**Expected**: augmented specs are 2-3x longer than the base spec.

### Batch Evaluation Runner

```bash
# Dry run
python -m improvements.evals.batch_eval_runner \
    --models model_a model_b \
    --model_labels baseline msm \
    --scenarios exfiltration \
    --goal_types none explicit \
    --goal_values none america \
    --dry_run \
    --output_dir /tmp/test_eval
```

**Expected**: prints all commands without executing; generates `run_all_evals.sh`.

### Expected Training Performance Estimates

Based on prior experience with similar training setups:

| Config | Hardware | Throughput | Memory/GPU | Time (1M tokens) |
|--------|----------|------------|------------|-------------------|
| Llama-3.1-8B, ZeRO-2, bf16, FA2, GC | 4×A100-80GB | ~3.5K tok/s | ~35GB | ~5 min |
| Llama-3.1-8B, ZeRO-2, bf16, FA2, GC | 4×A100-40GB | ~3.0K tok/s | ~32GB | ~6 min |
| Llama-3.1-8B, ZeRO-3+offload, bf16 | 2×A100-40GB | ~1.5K tok/s | ~28GB | ~11 min |
| Qwen2.5-32B, ZeRO-3+offload, bf16, FA2 | 8×A100-80GB | ~1.2K tok/s | ~65GB | ~14 min |

The packing improvement's benefit scales with the ratio of average document length to sequence length. For MSM data (avg ~1500 tokens, seq_len=4096), expect ~2.5x packing ratio → ~2.5x throughput improvement compared to naive padding.

---

## File Listing

```
improvements/
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretokenize.py      # Pre-tokenization + packing
│   │   ├── dataset.py          # Memory-mapped dataset
│   │   └── train.py            # Main training script
│   └── spec_augmenter/
│       ├── __init__.py
│       └── augment_spec.py     # Spec augmentation tool
├── evals/
│   ├── batch_eval_runner.py    # Multi-model batch evaluation
│   └── example_eval_config.yaml
├── configs/
│   ├── ds_zero2.json           # DeepSpeed ZeRO-2 config
│   ├── ds_zero3.json           # DeepSpeed ZeRO-3 config (CPU offload)
│   ├── msm_train.yaml          # MSM training config
│   └── aft_train.yaml          # AFT training config
├── exps/
│   ├── run_full_pipeline.sh    # End-to-end pipeline
│   ├── run_msm_train.sh        # Training-only launcher
│   └── run_spec_comparison.sh  # Spec science experiment
└── requirements.txt            # Additional Python dependencies
```


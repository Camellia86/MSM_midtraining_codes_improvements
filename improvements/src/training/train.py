"""MSM/AFT training script with DeepSpeed ZeRO, flash attention, and gradient checkpointing.

Supports:
  - MSM midtraining (causal LM on synthetic documents)
  - AFT fine-tuning (SFT on chat data with selective label masking)
  - DeepSpeed ZeRO Stage 1/2/3
  - Mixed precision (bf16/fp16)
  - Flash Attention 2
  - Gradient checkpointing
  - Wandb / TensorBoard logging
  - Checkpoint resume
  - Pre-tokenized (numpy mmap) or raw JSONL input

Usage:
    # Single GPU
    python improvements/src/training/train.py --config improvements/configs/msm_train.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 improvements/src/training/train.py \
        --config improvements/configs/msm_train.yaml

    # Multi-GPU with DeepSpeed
    deepspeed improvements/src/training/train.py \
        --config improvements/configs/msm_train.yaml \
        --deepspeed improvements/configs/ds_zero2.json
"""
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

import yaml

from improvements.src.training.dataset import PreTokenizedDataset

logger = logging.getLogger(__name__)


def setup_logging(rank: int):
    logging.basicConfig(
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO if rank == 0 else logging.WARNING,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def build_raw_dataset_from_jsonl(
    path: str, tokenizer, max_seq_len: int, fmt: str = "msm"
) -> PreTokenizedDataset:
    """Fallback: tokenize on-the-fly from JSONL and build a dataset in /tmp."""
    from improvements.src.training.pretokenize import (
        tokenize_msm_doc,
        tokenize_aft_chat,
        pack_sequences,
        pack_sequences_with_labels,
        pad_to_length,
    )

    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    if fmt == "msm":
        all_ids = [tokenize_msm_doc(d.get("text", ""), tokenizer, max_seq_len) for d in raw]
        all_ids = [x for x in all_ids if x]
        packed = pack_sequences(all_ids, max_seq_len)
        padded_ids = np.array(
            [pad_to_length(s, max_seq_len, tokenizer.pad_token_id) for s in packed],
            dtype=np.uint32,
        )
        attn = np.array(
            [pad_to_length([1] * len(s), max_seq_len, 0) for s in packed],
            dtype=np.uint8,
        )
        tmp_dir = Path(f"/tmp/msm_tokenized_{os.getpid()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "input_ids.npy", padded_ids)
        np.save(tmp_dir / "attention_mask.npy", attn)
    else:
        all_items = []
        for d in raw:
            msgs = d.get("messages", [])
            if msgs:
                item = tokenize_aft_chat(msgs, tokenizer, max_seq_len)
                if item["input_ids"]:
                    all_items.append(item)
        packed = pack_sequences_with_labels(all_items, max_seq_len)
        ids_arr = np.array(
            [pad_to_length(it["input_ids"], max_seq_len, tokenizer.pad_token_id) for it in packed],
            dtype=np.uint32,
        )
        labels_arr = np.array(
            [pad_to_length(it["labels"], max_seq_len, -100) for it in packed],
            dtype=np.int32,
        )
        attn = np.array(
            [pad_to_length([1] * len(it["input_ids"]), max_seq_len, 0) for it in packed],
            dtype=np.uint8,
        )
        tmp_dir = Path(f"/tmp/aft_tokenized_{os.getpid()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "input_ids.npy", ids_arr)
        np.save(tmp_dir / "labels.npy", labels_arr)
        np.save(tmp_dir / "attention_mask.npy", attn)

    return PreTokenizedDataset(str(tmp_dir), max_seq_len)


def train(config: dict):
    # ---- Distributed setup ----
    use_deepspeed = "deepspeed" in config and config["deepspeed"] is not None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if use_deepspeed:
        import deepspeed
        deepspeed.init_distributed()
    elif int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = get_rank()
    world_size = get_world_size()
    setup_logging(rank)

    set_seed(config.get("seed", 42) + rank)

    logger.info(f"Training config: {json.dumps(config, indent=2, default=str)}")

    # ---- Wandb ----
    if is_main_process() and config.get("wandb_project"):
        import wandb
        wandb.init(
            project=config["wandb_project"],
            name=config.get("wandb_run_name", config.get("output_dir", "msm_train").split("/")[-1]),
            config=config,
        )

    # ---- Model ----
    model_name = config["model_name_or_path"]
    logger.info(f"Loading model: {model_name}")

    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": getattr(torch, config.get("dtype", "bfloat16")),
        "trust_remote_code": True,
    }

    if config.get("flash_attention", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Dataset ----
    data_path = config["data_path"]
    max_seq_len = config.get("max_seq_len", 4096)
    data_format = config.get("data_format", "msm")

    if Path(data_path).is_dir() and (Path(data_path) / "input_ids.npy").exists():
        logger.info(f"Loading pre-tokenized data from {data_path}")
        dataset = PreTokenizedDataset(data_path, max_seq_len)
    else:
        logger.info(f"Tokenizing raw JSONL on-the-fly from {data_path}")
        dataset = build_raw_dataset_from_jsonl(data_path, tokenizer, max_seq_len, data_format)

    logger.info(f"Dataset: {len(dataset)} samples, seq_len={max_seq_len}")

    # ---- DataLoader ----
    per_device_batch_size = config.get("per_device_batch_size", 1)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.get("seed", 42)
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # ---- Training params ----
    num_epochs = config.get("num_epochs", 3)
    total_steps = (len(dataloader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = config.get("warmup_steps", min(100, total_steps // 10))
    learning_rate = config.get("learning_rate", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    save_steps = config.get("save_steps", 500)
    log_steps = config.get("log_steps", 10)
    output_dir = Path(config.get("output_dir", "outputs/msm_train"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- DeepSpeed or standard optimizer ----
    if use_deepspeed:
        import deepspeed

        ds_config_path = config["deepspeed"]
        with open(ds_config_path) as f:
            ds_config = json.load(f)

        ds_config.setdefault("train_micro_batch_size_per_gpu", per_device_batch_size)
        ds_config.setdefault("gradient_accumulation_steps", gradient_accumulation_steps)
        ds_config.setdefault("gradient_clipping", max_grad_norm)

        if "optimizer" not in ds_config:
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": weight_decay,
                },
            }

        if "scheduler" not in ds_config:
            ds_config["scheduler"] = {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_steps,
                },
            }

        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        lr_scheduler = get_scheduler(
            name=config.get("lr_scheduler_type", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    resume_path = config.get("resume_from_checkpoint")
    if resume_path and Path(resume_path).exists():
        if use_deepspeed:
            _, client_state = model_engine.load_checkpoint(resume_path)
            if client_state:
                start_epoch = client_state.get("epoch", 0)
                global_step = client_state.get("global_step", 0)
        else:
            ckpt = torch.load(Path(resume_path) / "trainer_state.pt", map_location="cpu")
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"Resumed from checkpoint: epoch={start_epoch}, step={global_step}")

    # ---- Training loop ----
    logger.info(f"Starting training: {num_epochs} epochs, {total_steps} total steps")
    logger.info(f"  Per-device batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {per_device_batch_size * gradient_accumulation_steps * world_size}")
    logger.info(f"  Learning rate: {learning_rate}")

    model_ref = model_engine if use_deepspeed else model

    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if use_deepspeed:
            model_engine.train()
        else:
            model.train()

        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not is_main_process(),
        )

        for step, batch in enumerate(progress_bar):
            if use_deepspeed:
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                outputs = model_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                model_engine.backward(loss)
                model_engine.step()
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            epoch_loss += loss.item()
            num_batches += 1

            if use_deepspeed:
                global_step = model_engine.global_steps

            if global_step % log_steps == 0 and is_main_process():
                avg_loss = epoch_loss / num_batches
                current_lr = (
                    lr_scheduler.get_last_lr()[0]
                    if hasattr(lr_scheduler, "get_last_lr")
                    else learning_rate
                )
                log_data = {
                    "loss": avg_loss,
                    "lr": current_lr,
                    "epoch": epoch + step / len(dataloader),
                    "global_step": global_step,
                }
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

                if config.get("wandb_project"):
                    import wandb
                    wandb.log(log_data, step=global_step)

            # ---- Checkpoint ----
            if save_steps > 0 and global_step > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    model_ref, optimizer, lr_scheduler, epoch, global_step,
                    output_dir, use_deepspeed, tokenizer, config
                )

        # End-of-epoch checkpoint
        save_checkpoint(
            model_ref, optimizer, lr_scheduler, epoch + 1, global_step,
            output_dir, use_deepspeed, tokenizer, config
        )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

    # ---- Save final model ----
    if is_main_process():
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        if use_deepspeed:
            model_engine.save_16bit_model(str(final_dir))
        else:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.save_pretrained(final_dir)

        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")

    if config.get("wandb_project") and is_main_process():
        import wandb
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    model, optimizer, lr_scheduler, epoch, global_step,
    output_dir, use_deepspeed, tokenizer, config
):
    """Save a training checkpoint."""
    ckpt_dir = output_dir / f"checkpoint-{global_step}"

    if use_deepspeed:
        model.save_checkpoint(
            str(output_dir),
            tag=f"checkpoint-{global_step}",
            client_state={"epoch": epoch, "global_step": global_step},
        )
    elif is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
            },
            ckpt_dir / "trainer_state.pt",
        )

    if is_main_process():
        logger.info(f"Checkpoint saved at step {global_step}")


def main():
    parser = argparse.ArgumentParser(description="MSM/AFT Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed JSON config (overrides config file)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args, _ = parser.parse_known_args()

    config = load_config(args.config)

    if args.deepspeed:
        config["deepspeed"] = args.deepspeed

    train(config)


if __name__ == "__main__":
    main()

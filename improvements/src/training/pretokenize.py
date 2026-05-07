"""Pre-tokenize MSM/AFT datasets for efficient training.

Converts raw text JSONL into memory-mapped token arrays, avoiding
repeated tokenization at each epoch. Supports both MSM (plain text)
and AFT (chat) formats.

Usage:
    python -m improvements.src.training.pretokenize \
        --input_path data/midtrain/general_spec/dataset.jsonl \
        --output_dir data/tokenized/general_spec_msm \
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --max_seq_len 4096 \
        --format msm
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_msm_doc(text: str, tokenizer, max_seq_len: int, add_eos: bool = True) -> list[int]:
    """Tokenize a plain-text MSM document."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_eos and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return ids[:max_seq_len]


def tokenize_aft_chat(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """Tokenize a chat conversation and produce input_ids + labels with masking.

    Only the assistant turns are supervised (labels != -100).
    """
    input_ids = []
    labels = []

    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        full_ids = full_ids[:max_seq_len]

        assistant_texts = [m["content"] for m in messages if m["role"] == "assistant"]
        assistant_ids_set = set()
        offset = 0
        for at in assistant_texts:
            at_ids = tokenizer.encode(at, add_special_tokens=False)
            pos = _find_subseq(full_ids, at_ids, start=offset)
            if pos >= 0:
                for i in range(pos, min(pos + len(at_ids), len(full_ids))):
                    assistant_ids_set.add(i)
                offset = pos + len(at_ids)

        input_ids = full_ids
        labels = [
            tok if i in assistant_ids_set else -100
            for i, tok in enumerate(full_ids)
        ]
    else:
        for msg in messages:
            role_prefix = f"{msg['role']}: "
            content = msg["content"]
            prefix_ids = tokenizer.encode(role_prefix, add_special_tokens=False)
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            turn_ids = prefix_ids + content_ids

            if msg["role"] == "assistant":
                input_ids.extend(turn_ids)
                labels.extend([-100] * len(prefix_ids) + content_ids)
            else:
                input_ids.extend(turn_ids)
                labels.extend([-100] * len(turn_ids))

        if tokenizer.eos_token_id is not None:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)

        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return {"input_ids": input_ids, "labels": labels}


def _find_subseq(seq: list, subseq: list, start: int = 0) -> int:
    """Find the start index of subseq in seq, starting from start."""
    n, m = len(seq), len(subseq)
    for i in range(start, n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def pack_sequences(all_ids: list[list[int]], max_seq_len: int) -> list[list[int]]:
    """Pack variable-length sequences into fixed-length chunks (greedy first-fit).

    Documents are separated by their existing EOS tokens. This avoids padding
    waste and increases GPU utilization.
    """
    packed = []
    current = []

    for ids in all_ids:
        if len(current) + len(ids) <= max_seq_len:
            current.extend(ids)
        else:
            if current:
                packed.append(current)
            if len(ids) >= max_seq_len:
                packed.append(ids[:max_seq_len])
                current = []
            else:
                current = list(ids)

    if current:
        packed.append(current)

    return packed


def pack_sequences_with_labels(
    all_items: list[dict], max_seq_len: int
) -> list[dict]:
    """Pack chat sequences (input_ids + labels) into fixed-length chunks."""
    packed = []
    cur_ids = []
    cur_labels = []

    for item in all_items:
        ids = item["input_ids"]
        labs = item["labels"]
        if len(cur_ids) + len(ids) <= max_seq_len:
            cur_ids.extend(ids)
            cur_labels.extend(labs)
        else:
            if cur_ids:
                packed.append({"input_ids": cur_ids, "labels": cur_labels})
            if len(ids) >= max_seq_len:
                packed.append({"input_ids": ids[:max_seq_len], "labels": labs[:max_seq_len]})
                cur_ids, cur_labels = [], []
            else:
                cur_ids, cur_labels = list(ids), list(labs)

    if cur_ids:
        packed.append({"input_ids": cur_ids, "labels": cur_labels})

    return packed


def pad_to_length(arr: list[int], length: int, pad_value: int = 0) -> list[int]:
    """Pad a sequence to a fixed length."""
    if len(arr) >= length:
        return arr[:length]
    return arr + [pad_value] * (length - len(arr))


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize MSM/AFT datasets")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--format", type=str, choices=["msm", "aft"], default="msm")
    parser.add_argument("--pack", action="store_true", default=True,
                        help="Pack sequences to maximize GPU utilization")
    parser.add_argument("--no_pack", action="store_true")
    args = parser.parse_args()

    if args.no_pack:
        args.pack = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Reading data from: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(raw_data)} examples")

    if args.format == "msm":
        all_ids = []
        for doc in tqdm(raw_data, desc="Tokenizing"):
            text = doc.get("text", doc.get("content", ""))
            ids = tokenize_msm_doc(text, tokenizer, args.max_seq_len)
            if ids:
                all_ids.append(ids)

        if args.pack:
            print(f"Packing {len(all_ids)} sequences into chunks of {args.max_seq_len}...")
            packed = pack_sequences(all_ids, args.max_seq_len)
            print(f"Packed into {len(packed)} chunks (was {len(all_ids)} docs)")

            padded = [pad_to_length(s, args.max_seq_len, tokenizer.pad_token_id) for s in packed]
            arr = np.array(padded, dtype=np.uint32)
            out_path = output_dir / "input_ids.npy"
            np.save(out_path, arr)
            attn_masks = np.array(
                [pad_to_length([1] * len(s), args.max_seq_len, 0) for s in packed],
                dtype=np.uint8,
            )
            np.save(output_dir / "attention_mask.npy", attn_masks)
        else:
            padded = [pad_to_length(s, args.max_seq_len, tokenizer.pad_token_id) for s in all_ids]
            arr = np.array(padded, dtype=np.uint32)
            np.save(output_dir / "input_ids.npy", arr)
            attn_masks = np.array(
                [pad_to_length([1] * len(s), args.max_seq_len, 0) for s in all_ids],
                dtype=np.uint8,
            )
            np.save(output_dir / "attention_mask.npy", attn_masks)

    elif args.format == "aft":
        all_items = []
        for doc in tqdm(raw_data, desc="Tokenizing"):
            messages = doc.get("messages", doc.get("conversation", []))
            if not messages:
                continue
            item = tokenize_aft_chat(messages, tokenizer, args.max_seq_len)
            if item["input_ids"]:
                all_items.append(item)

        if args.pack:
            print(f"Packing {len(all_items)} sequences...")
            packed = pack_sequences_with_labels(all_items, args.max_seq_len)
            print(f"Packed into {len(packed)} chunks")
        else:
            packed = all_items

        all_input_ids = [
            pad_to_length(item["input_ids"], args.max_seq_len, tokenizer.pad_token_id)
            for item in packed
        ]
        all_labels = [
            pad_to_length(item["labels"], args.max_seq_len, -100)
            for item in packed
        ]
        all_attn = [
            pad_to_length([1] * len(item["input_ids"]), args.max_seq_len, 0)
            for item in packed
        ]

        np.save(output_dir / "input_ids.npy", np.array(all_input_ids, dtype=np.uint32))
        np.save(output_dir / "labels.npy", np.array(all_labels, dtype=np.int32))
        np.save(output_dir / "attention_mask.npy", np.array(all_attn, dtype=np.uint8))

    meta = {
        "tokenizer_name": args.tokenizer_name,
        "max_seq_len": args.max_seq_len,
        "format": args.format,
        "packed": args.pack,
        "num_examples": len(raw_data),
        "num_chunks": len(packed) if args.pack else len(raw_data),
        "source": args.input_path,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved tokenized data to {output_dir}")
    print(f"  input_ids.npy: {os.path.getsize(output_dir / 'input_ids.npy') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

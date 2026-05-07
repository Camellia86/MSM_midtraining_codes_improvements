import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def count_tokens(doc, tokenizer):
    """Count assistant-only tokens from raw content (no chat template)."""
    if "text" in doc:
        return len(tokenizer.encode(doc["text"]))

    messages = doc.get("conversation", doc.get("messages"))
    if messages:
        count = 0
        for msg in messages:
            if msg["role"] == "assistant":
                count += len(tokenizer.encode(msg["content"]))
        return count

    return len(tokenizer.encode(doc.get("content", "")))


def count_tokens_chat(doc, tokenizer):
    """Count tokens with chat template applied (all roles, includes formatting tokens)."""
    if "text" in doc:
        return len(tokenizer.encode(doc["text"]))

    messages = doc.get("conversation", doc.get("messages"))
    if messages:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(tokenizer.encode(text))

    return len(tokenizer.encode(doc.get("content", "")))


def estimate_dataset_tokens(dataset_path, model_name="meta-llama/Llama-3.1-8B") -> tuple[int, int]:
    """Sample-based estimation of total docs and tokens. Returns (total_docs, estimated_total_tokens)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_path = Path(dataset_path)
    total_docs = sum(1 for _ in open(dataset_path, 'r'))
    sample_size = min(500, total_docs)
    sample_indices = set(random.sample(range(total_docs), sample_size))
    sample_tokens = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i in sample_indices:
                sample_tokens.append(count_tokens(json.loads(line), tokenizer))
    if not sample_tokens:
        return total_docs, 0
    avg_tokens = sum(sample_tokens) / len(sample_tokens)
    return total_docs, int(avg_tokens * total_docs)


def count_dataset_tokens(dataset_path, model_name, output_path=None, exact=False) -> dict:
    """Count tokens in a dataset and generate a histogram. Returns stats dict."""
    dataset_path = Path(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_fn = count_tokens_chat if tokenizer.chat_template else count_tokens

    print(f"Analyzing local dataset: {dataset_path}")
    print(f"Tokenizer: {model_name}")
    print(f"Token counting: {'chat template (all roles)' if tokenizer.chat_template else 'assistant-only (raw content)'}")

    if exact:
        all_tokens = []
        with open(dataset_path, 'r') as f:
            for line in f:
                all_tokens.append(token_fn(json.loads(line), tokenizer))
        total_docs = len(all_tokens)
        if not total_docs:
            print("No samples found.")
            return {"n_documents": 0, "total_tokens": 0, "tokens_per_document": 0, "max_token_length": 0}
        total_tokens = sum(all_tokens)
        avg_tokens = total_tokens / total_docs
        sample_tokens = all_tokens
    else:
        total_docs, estimated_total = estimate_dataset_tokens(dataset_path, model_name)
        if not total_docs:
            print("No samples found.")
            return {"n_documents": 0, "total_tokens": 0, "tokens_per_document": 0, "max_token_length": 0}
        sample_size = min(500, total_docs)
        sample_indices = set(random.sample(range(total_docs), sample_size))
        sample_tokens = []
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    sample_tokens.append(token_fn(json.loads(line), tokenizer))
        avg_tokens = sum(sample_tokens) / len(sample_tokens)
        total_tokens = estimated_total

    name_for_plot = dataset_path.stem

    plt.figure(figsize=(10, 6))
    plt.hist(sample_tokens, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Token Count Distribution: {name_for_plot}\n({'Exact' if exact else f'Sample Size: {len(sample_tokens)}'})")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)

    plt.axvline(avg_tokens, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {avg_tokens:.1f}')
    median_tokens = np.median(sample_tokens)
    plt.axvline(median_tokens, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_tokens:.1f}')
    plt.axvline(min(sample_tokens), color='orange', linestyle='dashed', linewidth=1, label=f'Min: {min(sample_tokens):.1f}')
    plt.axvline(max(sample_tokens), color='b', linestyle='dashed', linewidth=1, label=f'Max: {max(sample_tokens):.1f}')
    plt.legend()

    if output_path:
        plot_path = Path(output_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("plots")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"{name_for_plot}_token_hist.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Total documents: {total_docs:,}")
    if exact:
        print(f"Total tokens (exact): {total_tokens:,}")
    else:
        print(f"Sample size: {len(sample_tokens)}")
        print(f"Estimated total tokens: {total_tokens:,}")
    print(f"Avg tokens/doc: {avg_tokens:.1f}")
    print(f"Est documents needed for 1M tokens: {1000000 / avg_tokens:.1f}")
    print(f"Histogram saved to: {plot_path}")

    return {
        "n_documents": total_docs,
        "total_tokens": total_tokens,
        "tokens_per_document": int(avg_tokens),
        "max_token_length": max(sample_tokens),
    }

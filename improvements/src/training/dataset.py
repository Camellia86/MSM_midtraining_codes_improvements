"""Memory-mapped dataset for pre-tokenized MSM/AFT data.

Loads numpy arrays produced by pretokenize.py via memory mapping,
keeping memory footprint low regardless of dataset size.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """Dataset that loads pre-tokenized numpy arrays via memory mapping.

    Works for both MSM (input_ids only, CLM) and AFT (input_ids + labels, SFT).
    """

    def __init__(self, data_dir: str, max_seq_len: int | None = None):
        data_dir = Path(data_dir)

        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.input_ids = np.load(data_dir / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(data_dir / "attention_mask.npy", mmap_mode="r")

        labels_path = data_dir / "labels.npy"
        self.has_labels = labels_path.exists()
        if self.has_labels:
            self.labels = np.load(labels_path, mmap_mode="r")

        stored_seq_len = self.meta.get("max_seq_len", self.input_ids.shape[1])
        self.max_seq_len = min(max_seq_len, stored_seq_len) if max_seq_len else stored_seq_len

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(
            self.input_ids[idx, :self.max_seq_len].astype(np.int64).copy()
        )
        attention_mask = torch.from_numpy(
            self.attention_mask[idx, :self.max_seq_len].astype(np.int64).copy()
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.has_labels:
            labels = torch.from_numpy(
                self.labels[idx, :self.max_seq_len].astype(np.int64).copy()
            )
            result["labels"] = labels
        else:
            # CLM: labels = input_ids shifted by 1
            result["labels"] = input_ids.clone()

        return result

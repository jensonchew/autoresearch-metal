"""
One-time data preparation for autoresearch-metal experiments.
Downloads TinyStories data and trains a BPE tokenizer.

Usage:
    python prepare.py

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512        # context length (scaled for Apple Silicon)
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 10 * 524288  # number of tokens for val eval (scaled)
VOCAB_SIZE = 4096        # vocabulary size (scaled for smaller models)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# TinyStories GPT-4 clean dataset (single parquet file)
TINYSTORIES_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet"
TINYSTORIES_FILENAME = "tinystories_gpt4_clean.parquet"
VAL_SPLIT_START = 0      # first 10K stories for validation
VAL_SPLIT_END = 10_000
TRAIN_SPLIT_START = 10_000  # stories 10K+ for training

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def device_type():
    return detect_device().type

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download TinyStories parquet file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, TINYSTORIES_FILENAME)
    if os.path.exists(filepath):
        print(f"Data: {TINYSTORIES_FILENAME} already downloaded at {DATA_DIR}")
        return

    print(f"Data: downloading TinyStories (GPT-4 clean)...")
    response = requests.get(TINYSTORIES_URL, stream=True, timeout=60)
    response.raise_for_status()
    temp_path = filepath + ".tmp"
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.rename(temp_path, filepath)
    print(f"Data: downloaded to {filepath}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def parquet_path():
    return os.path.join(DATA_DIR, TINYSTORIES_FILENAME)


def text_iterator(split, max_chars=100_000_000, doc_cap=10_000):
    """Yield documents from the given split."""
    path = parquet_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"TinyStories not found at {path}. Run prepare.py first.")

    if split == "val":
        start, end = VAL_SPLIT_START, VAL_SPLIT_END
    else:
        start, end = TRAIN_SPLIT_START, None

    pf = pq.ParquetFile(path)
    nchars = 0
    idx = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["text"])
        for text in rg.column("text").to_pylist():
            if idx < start:
                idx += 1
                continue
            if end is not None and idx >= end:
                return
            doc = text[:doc_cap] if len(text) > doc_cap else text
            nchars += len(doc)
            yield doc
            idx += 1
            if nchars >= max_chars:
                return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    path = parquet_path()
    if not os.path.exists(path):
        print("Tokenizer: TinyStories parquet is missing. Run prepare.py first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator("train"), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from TinyStories."""
    path = parquet_path()
    assert os.path.exists(path), "TinyStories not found. Run prepare.py first."

    if split == "val":
        start, end = VAL_SPLIT_START, VAL_SPLIT_END
    else:
        start, end = TRAIN_SPLIT_START, None

    epoch = 1
    while True:
        pf = pq.ParquetFile(path)
        batch = []
        idx = 0
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            for text in rg.column("text").to_pylist():
                if idx < start:
                    idx += 1
                    continue
                if end is not None and idx >= end:
                    break
                batch.append(text)
                idx += 1
                if len(batch) >= tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
            if end is not None and idx >= end:
                break
        if batch:
            yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, device=None, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    if device is None:
        device = detect_device()
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device=None):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        raise RuntimeError("Evaluation produced zero target bytes; cannot compute BPB.")
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch-metal")
    parser.add_argument("--num-shards", type=int, default=10, help="Ignored (TinyStories is a single file). Kept for compatibility.")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dataset: TinyStories GPT-4 clean")
    print(f"Detected device: {device_type()}")
    print()

    download_data()
    print()

    train_tokenizer()
    print()
    print("Done! Ready to train.")

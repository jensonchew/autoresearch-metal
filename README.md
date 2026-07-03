# autoresearch-metal

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

**autoresearch-metal** is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) optimized for Apple Silicon (M1–M4) with Metal (MPS) acceleration. It gives an AI agent a small but real LLM training setup and lets it experiment autonomously overnight. The agent modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

### Key differences from the original

- **MPS backend** — auto-detects Apple Silicon GPU, falls back to CUDA or CPU
- **TinyStories dataset** — lower entropy, better results at small model scales
- **PyTorch SDPA** — replaces FlashAttention-3 (Triton/Hopper-only) with native scaled dot-product attention
- **No torch.compile on MPS** — eager mode only (torch.compile is unstable on MPS)
- **Scaled defaults** — DEPTH=4, TOTAL_BATCH_SIZE=2^16, MAX_SEQ_LEN=512, VOCAB_SIZE=4096
- **Device-agnostic** — same code works on MPS, CUDA, and CPU

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads TinyStories, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget**, regardless of your hardware. The metric is **val_bpb** (validation bits per byte) — lower is better.

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4, any RAM size), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download TinyStories and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If all commands work, your setup is ready for autonomous research mode.

## Running the agent

Spin up Claude Code, OpenCode, Codex, or any AI coding agent in this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`.
- **Fixed time budget.** Training always runs for exactly 5 minutes (~12 experiments/hour).
- **Device auto-detect.** MPS > CUDA > CPU — no config needed.
- **Self-contained.** Only PyTorch and a few small packages. No FA3/Triton dependency.

## Platform support

**Primary target:** Apple Silicon (M1–M4) with MPS. Also works on any CUDA GPU or CPU.

This fork removes the FlashAttention-3 and Triton dependencies, replacing them with native PyTorch SDPA. It also scales down the model defaults for Apple's unified memory architecture.

### Tuning for your Mac

| Mac model | Unified RAM | Suggested `DEVICE_BATCH_SIZE` | Suggested `DEPTH` |
|---|---|---|---|
| M1/M2 Air | 8 GB | 8 | 4 |
| M1/M2/M3 Pro | 16–18 GB | 16 | 4 |
| M1/M2/M3 Max | 32–64 GB | 32 | 6 |
| M3/M4 Ultra | 64–128 GB | 64 | 8 |

Edit these in `train.py` — the agent can also tune them automatically.

## Notable forks (upstream)

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — macOS patch
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) — Windows RTX
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MLX

## License

MIT

# Transformer From Scratch

A from-scratch implementation of an encoder-decoder Transformer in PyTorch, with a full training/inference workflow and decoding/optimization features.

## Features

- Core architecture:
  - Multi-Head Attention
  - Feed-Forward Network
  - Positional Encoding
  - Encoder / Decoder blocks
- Training pipeline:
  - train / eval
  - checkpoint save/load
  - resume training
- Decoding strategies:
  - greedy
  - top-k
  - top-p
  - beam search
- Inference optimization:
  - KV cache (self-attention)
  - cross-attention K/V precompute
- Benchmark mode for cache on/off comparison

## Project Structure

```text
transformer/
  main.py
  train.py
  generate.py
  config/
    base.yaml
  data/
    shakespeare.py
    tinyshakespeare.txt
  model/
    transformer_model.py
    MultiHeadAttention.py
    encoder.py
    encoder_block.py
    decoder.py
    decoder_block.py
    FFN.py
    pe.py
```

## Requirements

- Python 3.10+ (tested locally)
- Dependencies:
  - torch
  - tqdm
  - pyyaml

Install:

```bash
pip install -r requirements.txt
```

## Run

From repository root:

```bash
python -m transformer.main
```

Execution mode is controlled in `transformer/config/base.yaml`:

- `run_cfg.mode: train`
- `run_cfg.mode: inference`
- `run_cfg.mode: benchmark`

## Configuration

Main configs are defined in `transformer/config/base.yaml`:

- `run_cfg`
- `model_cfg`
- `optimizer_cfg`
- `train_cfg`
- `sample_cfg`
- `gen_cfg`
- `ben_gen_cfg`

Most hyperparameters can be changed directly in YAML without modifying source code.

## Checkpoints

Training saves:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`

Paths are controlled via `run_cfg.latest_ckpt_path` and `run_cfg.best_ckpt_path`.

## Benchmark Summary (example)

KV cache provides consistent speedups across beam sizes (batch=1, fixed generation length), e.g.:

- beam=1: ~1.35x
- beam=3: ~1.21x
- beam=5: ~1.13x
- beam=10: ~1.11x

## Learning Outcome

This project progressed from architecture reproduction to reasoning about:

- attention design and masking behavior,
- training vs autoregressive inference behavior,
- system-level optimization (cache/memory/copy strategy),
- reproducible engineering workflow (config + checkpoint + resume).

## Next Steps

- Integrate BPE tokenizer
- Implement decoder-only Transformer
- Expand evaluation/reporting

# Language Model from Scratch

A growing Python/PyTorch implementation developed while studying [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/). This repository currently contains byte-level BPE training and tokenization, Transformer building blocks, and a decoder-only language model forward pass.

## Structure

```text
src/lm_from_scratch/
  model/                     # Layers, attention, rotary embeddings, and Transformer
  tokenization/              # Byte-level BPE trainer and tokenizer
tests/
  unit/                      # Incremental BPE regression tests
  cs336/assignment1/          # Adapted course tests, adapters, fixtures, and snapshots
artifacts/tokenizers/
  tinystories/               # 10,000-token vocabulary, merges, and recorded metrics
  openwebtext/               # 32,000-token vocabulary, merges, and recorded metrics
```

Implementation code is organized by responsibility. Course adapters translate the assignment interfaces and reference weight names into package interfaces; implementation code does not depend on the test suite. The retained tests cover implemented components. This repository contains model forward computation, not an end-to-end training pipeline.

## Setup and checks

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The checks run on CPU. Some tokenizer tests compare against tiktoken's GPT-2 encoding and may download its reference encoding on first use. The inherited BPE speed test has a machine-dependent 1.5-second limit.

## Use the model

```python
import torch
from lm_from_scratch.model.transformer import TransformerLM

model = TransformerLM(
    vocab_size=10000, context_length=128, d_model=64,
    num_layers=2, num_heads=4, d_ff=128, rope_theta=10000.0,
)
token_ids = torch.randint(0, 10000, (2, 16))
logits = model(token_ids)  # (batch, sequence, vocabulary)
```

## Tokenizer artifacts

The tracked artifacts contain pickle vocabularies and ordered merge lists, readable `vocab.txt` and `merges.txt` exports, historical training metrics, and a manifest with file sizes and SHA-256 hashes. The text exports are for inspection; `Tokenizer.from_files` loads the pickle files. Only load pickle files from trusted sources.

```python
from pathlib import Path
from lm_from_scratch.tokenization.tokenizer import Tokenizer

artifacts = Path("artifacts/tokenizers/tinystories")
tokenizer = Tokenizer.from_files(
    artifacts / "tinystories_vocab.pkl",
    artifacts / "tinystories_merges.pkl",
    special_tokens=["<|endoftext|>"],
)
assert tokenizer.decode(tokenizer.encode("Hello, world!")) == "Hello, world!"
```

To train on a local corpus, provide explicit input and output paths:

```bash
uv run python -m lm_from_scratch.tokenization.train_bpe \
  --input data/corpus.txt --vocab-size 10000 --workers 4 \
  --output-dir artifacts/tokenizers/local
```

The training command reserves `<|endoftext|>` and writes pickle artifacts plus metrics. The tokenizer also offers `encode_iterable` for text supplied in chunks. Raw datasets and model checkpoints are excluded from Git. Small course test fixtures are included to make verification reproducible.

Metrics describe previous runs, not benchmarks of this extracted package. TinyStories input paths are normalized to remove machine-specific directories. OpenWebText metrics identify a separate incremental resume runner that is not included here; the included trainer does not implement resume support. Dataset identifiers and available historical hashes are retained in the metadata.

## Attribution

Based on personal work on Stanford CS336 Assignment 1. Compatibility tests, fixtures, snapshots, and adapter interfaces are adapted from the Stanford assignment materials. The original Stanford University MIT copyright and license notice is retained in `LICENSE`. This is an independent study project and is not an official Stanford release. Course handouts, lecture materials, submission tooling, and full training corpora are not bundled.

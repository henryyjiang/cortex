"""
Data loading for CortexGPT pre-training.

Default corpus: EleutherAI/the_pile_deduplicated (matches Pythia pretraining data).
Alternate corpus: pass dataset_name="HuggingFaceFW/fineweb-edu" (or any HF text
dataset) to build_dataloader for a higher-quality late-training phase.

Uses HuggingFace streaming to avoid downloading full corpora.
Sequences are packed to `seq_len` tokens with EOS separators.

Buffer reset signal: returned `eos_mask` tensor is True at the last token of each
document — train.py uses this to reset M_cross at EOS boundaries.
"""
from __future__ import annotations

import random
from typing import Generator, Iterator, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Streaming tokeniser
# ---------------------------------------------------------------------------

class TextStreamDataset(IterableDataset):
    """
    Streams any HuggingFace text dataset, packs tokens into fixed-length
    sequences, and yields (input_ids, labels, eos_mask) dicts.

    Defaults to EleutherAI/the_pile_deduplicated (Stage 1 / healing phase).
    Pass dataset_name="HuggingFaceFW/fineweb-edu" for Stage 2 general data.

    eos_mask: [seq_len] bool tensor, True at the last token of each document.
              Train loop resets M_cross wherever eos_mask is True.
    """

    def __init__(
        self,
        tokenizer,
        seq_len:      int           = 2048,
        dataset_name: str           = "EleutherAI/the_pile_deduplicated",
        text_column:  str           = "text",
        split:        str           = "train",
        buffer_size:  int           = 10_000,
        seed:         int           = 42,
        rank:         int           = 0,
        world_size:   int           = 1,
        max_tokens:   Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer    = tokenizer
        self.seq_len      = seq_len
        self.dataset_name = dataset_name
        self.text_column  = text_column
        self.split        = split
        self.buffer_size  = buffer_size
        self.seed         = seed
        self.rank         = rank
        self.world_size   = world_size
        self.max_tokens   = max_tokens

    def _stream(self) -> Generator[list[int], None, None]:
        """Yield token id lists, one per document, with EOS appended."""
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        eos = self.tokenizer.eos_token_id or 0

        for idx, example in enumerate(ds):
            if idx % self.world_size != self.rank:
                continue
            text = example.get(self.text_column, "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            ids.append(eos)
            yield ids

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + self.rank)
        buf: list[list[int]] = []
        tokens_yielded = 0

        def _fill(stream):
            nonlocal buf
            for doc in stream:
                buf.append(doc)
                if len(buf) >= self.buffer_size:
                    break

        stream = self._stream()
        _fill(stream)

        carry:     list[int]  = []
        eos_carry: list[bool] = []

        while True:
            rng.shuffle(buf)

            for doc in buf:
                eos_id = self.tokenizer.eos_token_id or 0
                pairs      = [(t, t == eos_id) for t in doc]
                carry     += [p[0] for p in pairs]
                eos_carry += [p[1] for p in pairs]

                while len(carry) >= self.seq_len + 1:
                    chunk_ids = carry[:self.seq_len + 1]
                    chunk_eos = eos_carry[:self.seq_len + 1]
                    carry     = carry[self.seq_len:]
                    eos_carry = eos_carry[self.seq_len:]

                    input_ids = torch.tensor(chunk_ids[:-1], dtype=torch.long)
                    labels    = torch.tensor(chunk_ids[1:],  dtype=torch.long)
                    eos_mask  = torch.tensor(chunk_eos[:-1], dtype=torch.bool)

                    tokens_yielded += self.seq_len
                    yield {"input_ids": input_ids, "labels": labels, "eos_mask": eos_mask}

                    if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                        return

            buf = []
            try:
                _fill(stream)
            except StopIteration:
                break
            if not buf:
                break


# Backward-compat alias
PileStreamDataset = TextStreamDataset


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
        "eos_mask":  torch.stack([b["eos_mask"]  for b in batch]),
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    tokenizer,
    seq_len:      int           = 2048,
    batch_size:   int           = 4,
    num_workers:  int           = 2,
    seed:         int           = 42,
    rank:         int           = 0,
    world_size:   int           = 1,
    max_tokens:   Optional[int] = None,
    buffer_size:  int           = 10_000,
    dataset_name: str           = "EleutherAI/the_pile_deduplicated",
    text_column:  str           = "text",
) -> torch.utils.data.DataLoader:
    ds = TextStreamDataset(
        tokenizer    = tokenizer,
        seq_len      = seq_len,
        dataset_name = dataset_name,
        text_column  = text_column,
        seed         = seed,
        rank         = rank,
        world_size   = world_size,
        max_tokens   = max_tokens,
        buffer_size  = buffer_size,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str = "EleutherAI/pythia-160m") -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.eos_token_id is None:
        raise ValueError("Tokeniser has no EOS token — check your model.")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

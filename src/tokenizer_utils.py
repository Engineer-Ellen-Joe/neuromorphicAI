from __future__ import annotations

import pathlib
import re
from typing import Optional


def load_sentencepiece_processor(model_path: pathlib.Path):
  try:
    import sentencepiece as spm
  except ImportError as exc:
    raise ImportError("sentencepiece is required. pip install sentencepiece") from exc
  sp = spm.SentencePieceProcessor()
  sp.load(str(model_path))
  return sp


def train_or_load_sp(
    data_path: pathlib.Path,
    model_dir: pathlib.Path,
    vocab_size: int = 1500,
    model_prefix: str = "spm",
    character_coverage: float = 1.0,
    model_type: str = "unigram",
) -> "sentencepiece.SentencePieceProcessor":
  """
  Train sentencepiece on data_path if model not present, otherwise load.
  Returns a loaded SentencePieceProcessor.
  """
  try:
    import sentencepiece as spm
  except ImportError as exc:
    raise ImportError("sentencepiece is required. pip install sentencepiece") from exc

  model_dir.mkdir(parents=True, exist_ok=True)
  model_path = model_dir / f"{model_prefix}.model"
  if model_path.exists():
    return load_sentencepiece_processor(model_path)

  # sentencepiece expects input via --input and the model_prefix without extension
  input_argument = f"--input={data_path}"
  model_prefix_arg = f"--model_prefix={model_dir / model_prefix}"
  args = " ".join(
      [
          input_argument,
          model_prefix_arg,
          f"--vocab_size={vocab_size}",
          f"--character_coverage={character_coverage}",
          f"--model_type={model_type}",
          "--hard_vocab_limit=false",
          "--byte_fallback=true",
          "--pad_id=0",
          "--unk_id=1",
          "--bos_id=2",
          "--eos_id=3",
      ]
  )
  try:
    spm.SentencePieceTrainer.Train(args)
  except RuntimeError as exc:
    msg = str(exc)
    # If vocab too high for tiny corpus, auto-reduce based on the hint in the error
    if "Vocabulary size too high" in msg:
      m = re.search(r"<=\s*(\d+)", msg)
      if m:
        limit = int(m.group(1))
        new_size = max(32, min(limit, vocab_size))
        spm.SentencePieceTrainer.Train(args.replace(f"--vocab_size={vocab_size}", f"--vocab_size={new_size}"))  # type: ignore
      else:
        raise
    else:
      raise
  return load_sentencepiece_processor(model_path)

from __future__ import annotations

import argparse
import pathlib
import sys

from tokenizer_utils import train_or_load_sp
from train_snn import train_one_batch, train_epoch

DEFAULT_DATA = pathlib.Path("modelData/trainData/shortTrain.txt")
DEFAULT_MODEL_DIR = pathlib.Path("modelData/tokenizer")


def run_output_test(data_path: pathlib.Path, model_dir: pathlib.Path, prompt: str):
  sp = train_or_load_sp(data_path, model_dir)
  pieces = sp.encode_as_pieces(prompt)
  ids = sp.encode_as_ids(prompt)
  decoded = sp.decode_ids(ids)
  print(f"[output test] prompt: {prompt}")
  safe_pieces = [p.replace("▁", "_") for p in pieces]
  print(f"[output test] pieces: {safe_pieces}")
  print(f"[output test] ids: {ids}")
  print(f"[output test] decoded: {decoded}")


def run_train_test(data_path: pathlib.Path, model_dir: pathlib.Path, run_batch: bool = False):
  """
  Placeholder training hook: trains tokenizer if missing and prints sample encoding.
  If run_batch=True, runs a tiny one-batch SNN training sanity-check.
  """
  sp = train_or_load_sp(data_path, model_dir)
  print(f"[train test] tokenizer vocab size: {sp.GetPieceSize()}")
  with data_path.open("r", encoding="utf-8") as f:
    line = f.readline().strip()
  if not line:
    print("[train test] data file is empty")
    return
  ids = sp.encode_as_ids(line)
  preview = ids[:32]
  suffix = "..." if len(ids) > len(preview) else ""
  print(f"[train test] first line ids: {preview}{suffix}")
  if run_batch:
    train_one_batch(sp, data_path)
  else:
    print("[train test] Pass --run-batch to run a tiny SNN training step.")


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description="SNN tokenizer runner")
  parser.add_argument("--mode", choices=["output_test", "train_test"], default=None, help="Select run mode.")
  parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA, help="Training text path.")
  parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR, help="Tokenizer model directory.")
  parser.add_argument("--prompt", type=str, default="안녕, 스파이킹!", help="Prompt for output test.")
  parser.add_argument("--run-batch", action="store_true", help="Run a tiny one-batch SNN training step in train_test mode.")
  parser.add_argument("--run-train", action="store_true", help="Run a small SNNStack training epoch.")
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--hidden1", type=int, default=64)
  parser.add_argument("--hidden2", type=int, default=64)
  parser.add_argument("--tbptt-len", type=int, default=16)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--save-path", type=pathlib.Path, default=None, help="Path to save model checkpoint.")
  args = parser.parse_args(argv)

  if args.mode is None:
    print("Select mode:\n  1) output_test\n  2) train_test")
    choice = input("Enter 1 or 2: ").strip()
    args.mode = "output_test" if choice == "1" else "train_test"

  if args.mode == "output_test":
    run_output_test(args.data, args.model_dir, args.prompt)
  elif args.mode == "train_test":
    if args.run_train:
      sp = train_or_load_sp(args.data, args.model_dir)
      train_epoch(
        sp,
        args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        tbptt_len=args.tbptt_len,
        epochs=args.epochs,
        save_path=args.save_path,
      )
    else:
      run_train_test(args.data, args.model_dir, run_batch=args.run_batch)
  else:
    parser.error(f"Unknown mode {args.mode}")


if __name__ == "__main__":
  main()

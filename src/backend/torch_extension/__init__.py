from __future__ import annotations

from pathlib import Path
import os
from torch.utils.cpp_extension import load

_EXT = None


def load_ext():
  global _EXT
  if _EXT is None:
    this_dir = Path(__file__).parent
    sources = [str(this_dir / "ops.cpp"), str(this_dir / "ops_cuda.cu")]
    build_dir = this_dir / "build" / "snn_cuda"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_NINJA", "0")
    _EXT = load(
      name="snn_cuda",
      sources=sources,
      build_directory=str(build_dir),
      extra_cuda_cflags=["--use_fast_math", "-allow-unsupported-compiler"],
      verbose=False,
    )
  return _EXT

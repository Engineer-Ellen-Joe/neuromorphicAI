from pathlib import Path
from torch.utils.cpp_extension import load


def build_extension():
    this_dir = Path(__file__).parent
    sources = [str(this_dir / "ops.cpp"), str(this_dir / "ops_cuda.cu")]
    return load(
        name="snn_cuda",
        sources=sources,
        extra_cuda_cflags=["--use_fast_math"],
        verbose=False,
    )


if __name__ == "__main__":
    build_extension()

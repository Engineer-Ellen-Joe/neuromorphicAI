import cupy as cp
from pathlib import Path

class GPUUtils:
    def __init__(self, kernel_dir: str = "src/kernels"):
        # 실행 위치에 상관없이 프로젝트 루트 기준으로 잡기
        project_root = Path(__file__).resolve().parents[2]
        self.kernel_dir = project_root / kernel_dir
        self.modules = {}

    def load_kernel(self, name: str, filename: str, func_names: list[str]):
        """CUDA 커널 파일을 로드해서 RawKernel 객체를 보관"""
        filepath = self.kernel_dir / filename
        code = filepath.read_text()
        module = cp.RawModule(
            code=code,
            options=(
                "-std=c++14",
                f"-I{self.kernel_dir}"   # common.h가 있는 경로 추가
            ),
            name_expressions=func_names
        )

        self.modules[name] = module
        return module

    def get_function(self, module_name: str, func_name: str):
        return self.modules[module_name].get_function(func_name)

    def create_stream(self):
        return cp.cuda.Stream(non_blocking=True)

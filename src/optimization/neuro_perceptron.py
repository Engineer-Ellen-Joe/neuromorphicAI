"""CuPy-based spikeless perceptron with STDP and deterministic path competition."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import math

import cupy as cp


kernel_s_pre = cp.ElementwiseKernel(
    "float64 x, float64 theta",
    "float64 s_pre",
    """
    s_pre = x >= theta ? 1.0 : 0.0;
    """,
    "kernel_s_pre",
)

kernel_membrane_fire = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_membrane_fire(
    const double* __restrict__ W,
    const double* __restrict__ x,
    const double* __restrict__ b,
    const double* __restrict__ theta_fire,
    double* __restrict__ m,
    double* __restrict__ s_post,
    const long long N_out,
    const long long N_in
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_out) {
        return;
    }
    const double* w_row = W + idx * N_in;
    double acc = 0.0;
    for (long long i = 0; i < N_in; ++i) {
        acc += w_row[i] * x[i];
    }
    acc += b[idx];
    m[idx] = acc;
    s_post[idx] = acc >= theta_fire[idx] ? 1.0 : 0.0;
}
""",
    "kernel_membrane_fire",
)

kernel_load_L1 = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_load_L1(
    const double* __restrict__ W,
    double* __restrict__ L,
    const long long N_out,
    const long long N_in
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_out) {
        return;
    }
    const double* w_row = W + idx * N_in;
    double sum_abs = 0.0;
    for (long long i = 0; i < N_in; ++i) {
        double val = w_row[i];
        sum_abs += val >= 0.0 ? val : -val;
    }
    L[idx] = sum_abs;
}
""",
    "kernel_load_L1",
)

kernel_sf_conduction = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_sf_conduction(
    const double* __restrict__ m,
    const double* __restrict__ theta_fire,
    const double* __restrict__ g_ax,
    const double* __restrict__ theta_ax,
    const double* __restrict__ L,
    const double rho,
    const double epsilon,
    const double* __restrict__ s_post,
    double* __restrict__ SF,
    double* __restrict__ cond,
    double* __restrict__ y_branch,
    const long long N_out,
    const long long B
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long total = N_out * B;
    if (idx >= total) {
        return;
    }
    long long j = idx / B;
    double drive = m[j] - theta_fire[j];
    if (drive < 0.0) {
        drive = 0.0;
    }
    drive = drive * g_ax[idx];
    double load_factor = 1.0 + rho * L[j];
    double threshold = theta_ax[idx] * load_factor;
    double denom = threshold + epsilon;
    double sf = 0.0;
    if (denom > 0.0) {
        sf = drive / denom;
    }
    double cond_val = 0.0;
    if (sf >= 1.0 && s_post[j] >= 0.5) {
        cond_val = 1.0;
    }
    SF[idx] = sf;
    cond[idx] = cond_val;
    y_branch[idx] = cond_val * s_post[j];
}
""",
    "kernel_sf_conduction",
)

kernel_token_scores = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_token_scores(
    const double* __restrict__ y_branch,
    const int* __restrict__ axon_to_token,
    double* __restrict__ S,
    const long long total,
    const long long K
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int token = axon_to_token[idx];
    if (token >= 0 && token < K) {
        double value = y_branch[idx];
        atomicAdd(&S[token], value);
    }
}
""",
    "kernel_token_scores",
)

kernel_argmax_deterministic = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_argmax_deterministic(
    const double* __restrict__ S,
    const long long K,
    long long* __restrict__ y_hat
) {
    if (blockIdx.x > 0 || threadIdx.x > 0) {
        return;
    }
    double best_value = S[0];
    long long best_index = 0;
    for (long long i = 1; i < K; ++i) {
        double current = S[i];
        if (current > best_value) {
            best_value = current;
            best_index = i;
        } else if (current == best_value && i < best_index) {
            best_index = i;
        }
    }
    y_hat[0] = best_index;
}
""",
    "kernel_argmax_deterministic",
)

kernel_traces_update = cp.ElementwiseKernel(
    "raw float64 e_pre, raw float64 e_post, raw float64 s_pre, raw float64 s_post, float64 decay_pre, float64 decay_post, int32 N_pre, int32 N_post",
    "float64 dummy",
    """
    double tmp = 0.0;
    if (i < N_pre) {
        double updated_pre = e_pre[i] * decay_pre + s_pre[i];
        e_pre[i] = updated_pre;
    }
    if (i < N_post) {
        double updated_post = e_post[i] * decay_post + s_post[i];
        e_post[i] = updated_post;
    }
    dummy = tmp;
    """,
    "kernel_traces_update",
)

kernel_stdp_additive = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_stdp_additive(
    double* __restrict__ W,
    const double* __restrict__ e_pre,
    const double* __restrict__ e_post,
    const double* __restrict__ s_pre,
    const double* __restrict__ s_post,
    const double eta,
    const double A_plus,
    const double A_minus,
    const double w_min,
    const double w_max,
    const long long N_out,
    const long long N_in
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long total = N_out * N_in;
    if (idx >= total) {
        return;
    }
    long long j = idx / N_in;
    long long i = idx % N_in;
    double current = W[idx];
    double delta = eta * (A_plus * e_pre[i] * s_post[j] - A_minus * s_pre[i] * e_post[j]);
    current += delta;
    if (current < w_min) {
        current = w_min;
    }
    if (current > w_max) {
        current = w_max;
    }
    W[idx] = current;
}
""",
    "kernel_stdp_additive",
)

kernel_stdp_multiplicative = cp.RawKernel(
    r"""
extern "C" __global__
void kernel_stdp_multiplicative(
    double* __restrict__ W,
    const double* __restrict__ e_pre,
    const double* __restrict__ e_post,
    const double* __restrict__ s_pre,
    const double* __restrict__ s_post,
    const double eta,
    const double A_plus,
    const double A_minus,
    const double w_min,
    const double w_max,
    const long long N_out,
    const long long N_in
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long total = N_out * N_in;
    if (idx >= total) {
        return;
    }
    long long j = idx / N_in;
    long long i = idx % N_in;
    double current = W[idx];
    double ltp = A_plus * e_pre[i] * s_post[j];
    double ltd = A_minus * s_pre[i] * e_post[j];
    double delta = eta * (ltp * (w_max - current) - ltd * (current - w_min));
    current += delta;
    if (current < w_min) {
        current = w_min;
    }
    if (current > w_max) {
        current = w_max;
    }
    W[idx] = current;
}
""",
    "kernel_stdp_multiplicative",
)


@dataclass
class PerceptronConfig:
    N_in: int
    N_out: int
    B: int
    K: int
    w_min: float = -1.0
    w_max: float = 1.0
    eta: float = 1e-3
    A_plus: float = 1.0
    A_minus: float = 1.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    epsilon: float = 1e-12
    stdp_mode: str = "additive"

    def __post_init__(self) -> None:
        if self.N_in <= 0:
            raise ValueError("N_in must be positive.")
        if self.N_out <= 0:
            raise ValueError("N_out must be positive.")
        if self.B <= 0:
            raise ValueError("B must be positive.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.w_min >= self.w_max:
            raise ValueError("w_min must be less than w_max.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.A_plus <= 0.0:
            raise ValueError("A_plus must be positive.")
        if self.A_minus <= 0.0:
            raise ValueError("A_minus must be positive.")
        if self.tau_pre <= 0.0:
            raise ValueError("tau_pre must be positive.")
        if self.tau_post <= 0.0:
            raise ValueError("tau_post must be positive.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        mode = self.stdp_mode.lower()
        if mode not in ("additive", "multiplicative"):
            raise ValueError("stdp_mode must be 'additive' or 'multiplicative'.")
        self.stdp_mode = mode


class SpikelessPerceptron:
    """Spikeless perceptron with path competition and STDP implemented in CuPy."""

    _THREADS_PER_BLOCK = 128

    def __init__(self, cfg: PerceptronConfig):
        if not isinstance(cfg, PerceptronConfig):
            raise TypeError("cfg must be a PerceptronConfig instance.")
        self.cfg = cfg
        self._device = cp.cuda.Device()
        self._device.use()
        self._w_min = float(cfg.w_min)
        self._w_max = float(cfg.w_max)
        self._rho = 0.0
        self._decay_pre = math.exp(-1.0 / float(cfg.tau_pre))
        self._decay_post = math.exp(-1.0 / float(cfg.tau_post))
        self._trace_dummy = cp.zeros((max(cfg.N_in, cfg.N_out),), dtype=cp.float64)
        self._y_hat_buf = cp.zeros((1,), dtype=cp.int64)
        self._init_buffers()
        self.reset_traces()

    def _init_buffers(self) -> None:
        """Initialise parameter and state buffers."""
        self.W = cp.random.uniform(-0.1, 0.1, (self.cfg.N_out, self.cfg.N_in), dtype=cp.float64)
        self.W = cp.clip(self.W, self._w_min, self._w_max)
        self.b = cp.zeros((self.cfg.N_out,), dtype=cp.float64)
        self.theta_fire = cp.full((self.cfg.N_out,), 0.5, dtype=cp.float64)
        self.theta_pre = cp.full((self.cfg.N_in,), 0.1, dtype=cp.float64)
        self.g_ax = cp.full((self.cfg.N_out, self.cfg.B), 1.0, dtype=cp.float64)
        self.theta_ax = cp.full((self.cfg.N_out, self.cfg.B), 1.0, dtype=cp.float64)
        self.e_pre = cp.zeros((self.cfg.N_in,), dtype=cp.float64)
        self.e_post = cp.zeros((self.cfg.N_out,), dtype=cp.float64)
        self.m = cp.zeros((self.cfg.N_out,), dtype=cp.float64)
        self.s_pre = cp.zeros((self.cfg.N_in,), dtype=cp.float64)
        self.s_post = cp.zeros((self.cfg.N_out,), dtype=cp.float64)
        self.L = cp.zeros((self.cfg.N_out,), dtype=cp.float64)
        self.SF = cp.zeros((self.cfg.N_out, self.cfg.B), dtype=cp.float64)
        self.cond = cp.zeros((self.cfg.N_out, self.cfg.B), dtype=cp.float64)
        self.y_branch = cp.zeros((self.cfg.N_out, self.cfg.B), dtype=cp.float64)
        self.S = cp.zeros((self.cfg.K,), dtype=cp.float64)
        seq = cp.arange(self.cfg.N_out * self.cfg.B, dtype=cp.int32)
        if self.cfg.K > 0:
            seq = seq % int(self.cfg.K)
        self.axon_to_token = seq.reshape(self.cfg.N_out, self.cfg.B)


    @staticmethod
    def _check_shape(arr: cp.ndarray, expected: Sequence[int], name: str) -> None:
        if arr.shape != tuple(expected):
            raise ValueError(f"{name} must have shape {tuple(expected)}, got {arr.shape}.")

    @staticmethod
    def _check_finite(arr: cp.ndarray, name: str) -> None:
        if not bool(cp.isfinite(arr).all()):
            raise ValueError(f"{name} contains non-finite values.")

    @staticmethod
    def _check_positive(arr: cp.ndarray, name: str) -> None:
        if bool((arr <= 0.0).any()):
            raise ValueError(f"{name} must contain strictly positive values.")

    def _validate_input(self, x_t: Any) -> cp.ndarray:
        if not isinstance(x_t, cp.ndarray):
            raise ValueError("x_t must be a cupy.ndarray.")
        if x_t.dtype != cp.float64:
            raise ValueError("x_t must have dtype float64.")
        if x_t.ndim != 1 or x_t.shape[0] != self.cfg.N_in:
            raise ValueError(f"x_t must have shape ({self.cfg.N_in},).")
        if x_t.device.id != self._device.id:
            raise ValueError(f"x_t must reside on device {self._device.id}.")
        if bool(cp.isnan(x_t).any()):
            raise ValueError("x_t contains NaN.")
        if bool(cp.isinf(x_t).any()):
            raise ValueError("x_t contains Inf.")
        return x_t

    def _update_decays(self) -> None:
        self._decay_pre = math.exp(-1.0 / float(self.cfg.tau_pre))
        self._decay_post = math.exp(-1.0 / float(self.cfg.tau_post))

    def _forward_compute(
        self,
        x_t: cp.ndarray,
        s_pre_out: cp.ndarray,
        m_out: cp.ndarray,
        s_post_out: cp.ndarray,
        L_out: cp.ndarray,
        SF_out: cp.ndarray,
        cond_out: cp.ndarray,
        y_branch_out: cp.ndarray,
        S_out: cp.ndarray,
    ) -> None:
        """Run forward computations filling the provided output buffers."""
        kernel_s_pre(x_t, self.theta_pre, s_pre_out)
        total_neurons = int(self.cfg.N_out)
        blocks_neuron = (total_neurons + self._THREADS_PER_BLOCK - 1) // self._THREADS_PER_BLOCK
        kernel_membrane_fire(
            (blocks_neuron,),
            (self._THREADS_PER_BLOCK,),
            (
                self.W,
                x_t,
                self.b,
                self.theta_fire,
                m_out,
                s_post_out,
                int(self.cfg.N_out),
                int(self.cfg.N_in),
            ),
        )
        kernel_load_L1(
            (blocks_neuron,),
            (self._THREADS_PER_BLOCK,),
            (
                self.W,
                L_out,
                int(self.cfg.N_out),
                int(self.cfg.N_in),
            ),
        )
        total_paths = int(self.cfg.N_out * self.cfg.B)
        blocks_paths = (total_paths + self._THREADS_PER_BLOCK - 1) // self._THREADS_PER_BLOCK
        kernel_sf_conduction(
            (blocks_paths,),
            (self._THREADS_PER_BLOCK,),
            (
                m_out,
                self.theta_fire,
                self.g_ax,
                self.theta_ax,
                L_out,
                float(self._rho),
                float(self.cfg.epsilon),
                s_post_out,
                SF_out,
                cond_out,
                y_branch_out,
                int(self.cfg.N_out),
                int(self.cfg.B),
            ),
        )
        S_out.fill(0.0)
        kernel_token_scores(
            (blocks_paths,),
            (self._THREADS_PER_BLOCK,),
            (
                y_branch_out,
                self.axon_to_token,
                S_out,
                int(total_paths),
                int(self.cfg.K),
            ),
        )

    def _update_traces(self) -> None:
        kernel_traces_update(
            self.e_pre,
            self.e_post,
            self.s_pre,
            self.s_post,
            float(self._decay_pre),
            float(self._decay_post),
            int(self.cfg.N_in),
            int(self.cfg.N_out),
            self._trace_dummy,
        )

    def _apply_stdp(self) -> None:
        total = int(self.cfg.N_out * self.cfg.N_in)
        if total == 0:
            return
        blocks = (total + self._THREADS_PER_BLOCK - 1) // self._THREADS_PER_BLOCK
        args = (
            self.W,
            self.e_pre,
            self.e_post,
            self.s_pre,
            self.s_post,
            float(self.cfg.eta),
            float(self.cfg.A_plus),
            float(self.cfg.A_minus),
            float(self._w_min),
            float(self._w_max),
            int(self.cfg.N_out),
            int(self.cfg.N_in),
        )
        if self.cfg.stdp_mode == "additive":
            kernel_stdp_additive((blocks,), (self._THREADS_PER_BLOCK,), args)
        else:
            kernel_stdp_multiplicative((blocks,), (self._THREADS_PER_BLOCK,), args)

    def to_device(self, device_id: int = 0) -> None:
        """Move all buffers to the target CUDA device."""
        new_device = cp.cuda.Device(int(device_id))
        new_device.use()
        self._device = new_device
        float64_attrs = [
            "W",
            "b",
            "theta_fire",
            "theta_pre",
            "g_ax",
            "theta_ax",
            "e_pre",
            "e_post",
            "m",
            "s_pre",
            "s_post",
            "L",
            "SF",
            "cond",
            "y_branch",
            "S",
            "_trace_dummy",
        ]
        for name in float64_attrs:
            setattr(self, name, cp.asarray(getattr(self, name), dtype=cp.float64))
        self.axon_to_token = cp.asarray(self.axon_to_token, dtype=self.axon_to_token.dtype)
        self._y_hat_buf = cp.asarray(self._y_hat_buf, dtype=cp.int64)

    def set_params(self, **kwargs: Any) -> None:
        """Set scalar or tensor parameters."""
        if not kwargs:
            return
        self._device.use()
        if "w_min" in kwargs or "w_max" in kwargs:
            new_w_min = float(kwargs.pop("w_min", self._w_min))
            new_w_max = float(kwargs.pop("w_max", self._w_max))
            if new_w_min >= new_w_max:
                raise ValueError("w_min must be less than w_max.")
            self._w_min = new_w_min
            self._w_max = new_w_max
            self.cfg.w_min = new_w_min
            self.cfg.w_max = new_w_max
        for key, value in kwargs.items():
            if key == "W":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_out, self.cfg.N_in), "W")
                self._check_finite(arr, "W")
                self.W = cp.clip(arr, self._w_min, self._w_max)
            elif key == "b":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_out,), "b")
                self._check_finite(arr, "b")
                self.b = arr
            elif key == "theta_fire":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_out,), "theta_fire")
                self._check_finite(arr, "theta_fire")
                self._check_positive(arr, "theta_fire")
                self.theta_fire = arr
            elif key == "theta_pre":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_in,), "theta_pre")
                self._check_finite(arr, "theta_pre")
                self._check_positive(arr, "theta_pre")
                self.theta_pre = arr
            elif key == "g_ax":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_out, self.cfg.B), "g_ax")
                self._check_finite(arr, "g_ax")
                self._check_positive(arr, "g_ax")
                self.g_ax = arr
            elif key == "theta_ax":
                arr = cp.asarray(value, dtype=cp.float64)
                self._check_shape(arr, (self.cfg.N_out, self.cfg.B), "theta_ax")
                self._check_finite(arr, "theta_ax")
                self._check_positive(arr, "theta_ax")
                self.theta_ax = arr
            elif key == "axon_to_token":
                self.set_axon_map(value)
            elif key == "eta":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("eta must be positive.")
                self.cfg.eta = val
            elif key == "A_plus":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("A_plus must be positive.")
                self.cfg.A_plus = val
            elif key == "A_minus":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("A_minus must be positive.")
                self.cfg.A_minus = val
            elif key == "tau_pre":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("tau_pre must be positive.")
                self.cfg.tau_pre = val
                self._update_decays()
            elif key == "tau_post":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("tau_post must be positive.")
                self.cfg.tau_post = val
                self._update_decays()
            elif key == "epsilon":
                val = float(value)
                if val <= 0.0:
                    raise ValueError("epsilon must be positive.")
                self.cfg.epsilon = val
            elif key == "stdp_mode":
                mode = str(value).lower()
                if mode not in ("additive", "multiplicative"):
                    raise ValueError("stdp_mode must be 'additive' or 'multiplicative'.")
                self.cfg.stdp_mode = mode
            elif key == "rho":
                val = float(value)
                if val < 0.0:
                    raise ValueError("rho must be non-negative.")
                self._rho = val
            else:
                raise ValueError(f"Unsupported parameter '{key}' in set_params.")

    def get_params(self) -> Dict[str, Any]:
        """Return a snapshot of parameters and state buffers."""
        return {
            "W": self.W.copy(),
            "b": self.b.copy(),
            "theta_fire": self.theta_fire.copy(),
            "theta_pre": self.theta_pre.copy(),
            "g_ax": self.g_ax.copy(),
            "theta_ax": self.theta_ax.copy(),
            "e_pre": self.e_pre.copy(),
            "e_post": self.e_post.copy(),
            "L": self.L.copy(),
            "m": self.m.copy(),
            "s_pre": self.s_pre.copy(),
            "s_post": self.s_post.copy(),
            "SF": self.SF.copy(),
            "cond": self.cond.copy(),
            "y_branch": self.y_branch.copy(),
            "S": self.S.copy(),
            "axon_to_token": self.axon_to_token.copy(),
            "w_min": float(self._w_min),
            "w_max": float(self._w_max),
            "eta": float(self.cfg.eta),
            "A_plus": float(self.cfg.A_plus),
            "A_minus": float(self.cfg.A_minus),
            "tau_pre": float(self.cfg.tau_pre),
            "tau_post": float(self.cfg.tau_post),
            "epsilon": float(self.cfg.epsilon),
            "rho": float(self._rho),
            "stdp_mode": self.cfg.stdp_mode,
        }

    def step(self, x_t: cp.ndarray) -> int:
        """Run a single online update step and return the predicted token index."""
        self._device.use()
        x_dev = self._validate_input(x_t)
        self._forward_compute(
            x_dev,
            self.s_pre,
            self.m,
            self.s_post,
            self.L,
            self.SF,
            self.cond,
            self.y_branch,
            self.S,
        )
        kernel_argmax_deterministic(
            (1,),
            (1,),
            (
                self.S,
                int(self.cfg.K),
                self._y_hat_buf,
            ),
        )
        self._update_traces()
        self._apply_stdp()
        return int(self._y_hat_buf.get()[0])

    def forward_eval(self, x_t: cp.ndarray):
        """Compute token scores without modifying internal state."""
        self._device.use()
        x_dev = self._validate_input(x_t)
        s_pre_tmp = cp.empty_like(self.s_pre)
        m_tmp = cp.empty_like(self.m)
        s_post_tmp = cp.empty_like(self.s_post)
        L_tmp = cp.empty_like(self.L)
        SF_tmp = cp.empty_like(self.SF)
        cond_tmp = cp.empty_like(self.cond)
        y_branch_tmp = cp.empty_like(self.y_branch)
        S_tmp = cp.empty_like(self.S)
        self._forward_compute(
            x_dev,
            s_pre_tmp,
            m_tmp,
            s_post_tmp,
            L_tmp,
            SF_tmp,
            cond_tmp,
            y_branch_tmp,
            S_tmp,
        )
        return S_tmp, y_branch_tmp, m_tmp, s_pre_tmp, s_post_tmp

    def set_axon_map(self, axon_to_token_host_ndarray: Sequence[int]) -> None:
        """Load the branch-to-token mapping from host data."""
        self._device.use()
        arr = cp.asarray(axon_to_token_host_ndarray)
        if arr.ndim != 2:
            raise ValueError("axon_to_token array must be two-dimensional.")
        self._check_shape(arr, (self.cfg.N_out, self.cfg.B), "axon_to_token")
        if arr.dtype not in (cp.int32, cp.int64):
            arr = arr.astype(cp.int32)
        arr = arr.astype(cp.int32, copy=False)
        arr = cp.ascontiguousarray(arr)
        if bool((arr < 0).any()):
            raise ValueError("axon_to_token indices must be non-negative.")
        if bool((arr >= self.cfg.K).any()):
            raise ValueError("axon_to_token indices must be less than K.")
        self.axon_to_token = arr

    def reset_traces(self) -> None:
        """Reset eligibility traces and spike events."""
        self._device.use()
        self.e_pre.fill(0.0)
        self.e_post.fill(0.0)
        self.s_pre.fill(0.0)
        self.s_post.fill(0.0)

    def reset_weights(self, W: Optional[Any] = None, b: Optional[Any] = None) -> None:
        """Reset weights and biases, optionally to user-provided tensors."""
        self._device.use()
        if W is None:
            self.W = cp.random.uniform(-0.1, 0.1, (self.cfg.N_out, self.cfg.N_in), dtype=cp.float64)
            self.W = cp.clip(self.W, self._w_min, self._w_max)
        else:
            arr = cp.asarray(W, dtype=cp.float64)
            self._check_shape(arr, (self.cfg.N_out, self.cfg.N_in), "W")
            self._check_finite(arr, "W")
            self.W = cp.clip(arr, self._w_min, self._w_max)
        if b is None:
            self.b.fill(0.0)
        else:
            arr_b = cp.asarray(b, dtype=cp.float64)
            self._check_shape(arr_b, (self.cfg.N_out,), "b")
            self._check_finite(arr_b, "b")
            self.b = arr_b


__all__ = ["PerceptronConfig", "SpikelessPerceptron"]



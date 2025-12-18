from __future__ import annotations

import numpy as np
import cupy as cp
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

# ----- 원본 파라미터 클래스 (CPU에서 설정용으로 사용) -----
# (이 클래스들은 CuPy 코드에서 직접 사용되지는 않지만,
# 네트워크 설정 시 파라미터를 편리하게 그룹화하는 데 사용됩니다.)

@dataclass
class SomaParams:
    C_m: float = 200e-12
    g_L: float = 10e-9
    E_L: float = -70e-3
    I_ext: float = 0.0

@dataclass
class AISParams:
    C_m: float = 40e-12
    g_L: float = 8e-9
    E_L: float = -70e-3
    V_T: float = -50e-3
    Delta_T: float = 2.0e-3
    V_spike: float = 20e-3
    V_reset: float = -58e-3
    t_refrac: float = 2e-3

@dataclass
class CouplingParams:
    g_c: float = 30e-9

@dataclass
class SynapseParams:
    g_max: float = 2.5e-9
    E_rev: float = 0.0
    tau_decay: float = 5e-3
    A_plus: float = 0.005
    A_minus: float = 0.005
    tau_pre: float = 20e-3
    tau_post: float = 20e-3
    eta_bcm: float = 1e-5
    tau_rate: float = 50e-3
    tau_theta: float = 1.0
    w_min: float = 0.0
    w_max: float = 3.0
    # Axon delay in seconds (replaces complex AxonBranch logic)
    delay: float = 1e-3

# ----- CUDA 커널 -----

_NEURON_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void update_neurons(
    const int N_NEURONS,
    double t, double dt,
    // Neuron States (read/write)
    double* V_soma, double* V_ais, double* refrac_until,
    // Neuron Params (read-only)
    const double* soma_C_m, const double* soma_g_L, const double* soma_E_L, const double* soma_I_ext,
    const double* ais_C_m, const double* ais_g_L, const double* ais_E_L, const double* ais_V_T,
    const double* ais_Delta_T, const double* ais_V_spike, const double* ais_V_reset, const double* ais_t_refrac,
    const double* coup_g_c,
    // Synaptic current from previous step
    const double* I_syn_total,
    // Outputs
    int* spiked_now
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= N_NEURONS) return;

    double v_s = V_soma[nid];
    double v_a = V_ais[nid];
    bool in_refrac = (t < refrac_until[nid]);
    double i_syn = I_syn_total[nid];

    if (!in_refrac) {
        // RK4 integration
        // k1
        double dVs_1 = (soma_g_L[nid] * (soma_E_L[nid] - v_s) + coup_g_c[nid] * (v_a - v_s) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double dVa_1 = (ais_g_L[nid] * (ais_E_L[nid] - v_a) + coup_g_c[nid] * (v_s - v_a) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

        // k2
        double v_s2 = v_s + 0.5 * dt * dVs_1;
        double v_a2 = v_a + 0.5 * dt * dVa_1;
        double dVs_2 = (soma_g_L[nid] * (soma_E_L[nid] - v_s2) + coup_g_c[nid] * (v_a2 - v_s2) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double dVa_2 = (ais_g_L[nid] * (ais_E_L[nid] - v_a2) + coup_g_c[nid] * (v_s2 - v_a2) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a2 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

        // k3
        double v_s3 = v_s + 0.5 * dt * dVs_2;
        double v_a3 = v_a + 0.5 * dt * dVa_2;
        double dVs_3 = (soma_g_L[nid] * (soma_E_L[nid] - v_s3) + coup_g_c[nid] * (v_a3 - v_s3) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double dVa_3 = (ais_g_L[nid] * (ais_E_L[nid] - v_a3) + coup_g_c[nid] * (v_s3 - v_a3) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a3 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

        // k4
        double v_s4 = v_s + dt * dVs_3;
        double v_a4 = v_a + dt * dVa_3;
        double dVs_4 = (soma_g_L[nid] * (soma_E_L[nid] - v_s4) + coup_g_c[nid] * (v_a4 - v_s4) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double dVa_4 = (ais_g_L[nid] * (ais_E_L[nid] - v_a4) + coup_g_c[nid] * (v_s4 - v_a4) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a4 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

        V_soma[nid] = v_s + (dt / 6.0) * (dVs_1 + 2.0 * dVs_2 + 2.0 * dVs_3 + dVs_4);
        V_ais[nid] = v_a + (dt / 6.0) * (dVa_1 + 2.0 * dVa_2 + 2.0 * dVa_3 + dVa_4);
    } else {
        // In refractory: clamp AIS, integrate soma passively
        double v_a_fixed = ais_V_reset[nid];
        double k1 = (soma_g_L[nid] * (soma_E_L[nid] - v_s) + coup_g_c[nid] * (v_a_fixed - v_s) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double k2 = (soma_g_L[nid] * (soma_E_L[nid] - (v_s + 0.5 * dt * k1)) + coup_g_c[nid] * (v_a_fixed - (v_s + 0.5 * dt * k1)) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double k3 = (soma_g_L[nid] * (soma_E_L[nid] - (v_s + 0.5 * dt * k2)) + coup_g_c[nid] * (v_a_fixed - (v_s + 0.5 * dt * k2)) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        double k4 = (soma_g_L[nid] * (soma_E_L[nid] - (v_s + dt * k3)) + coup_g_c[nid] * (v_a_fixed - (v_s + dt * k3)) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
        V_soma[nid] = v_s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        V_ais[nid] = v_a_fixed;
    }

    if (!in_refrac && V_ais[nid] >= ais_V_spike[nid]) {
        spiked_now[nid] = 1;
        V_ais[nid] = ais_V_reset[nid];
        refrac_until[nid] = t + ais_t_refrac[nid];
    } else {
        spiked_now[nid] = 0;
    }
}
''', 'update_neurons')

_SYNAPSE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void update_synapses(
    const int N_SYNAPSES,
    double dt,
    const int buffer_idx,
    const int buffer_depth,
    const int n_synapses,
    double* g_incr_buffer, // flattened [buffer_depth, n_synapses]
    // Synapse States (read/write)
    double* syn_g, double* syn_w, double* syn_pre_trace, double* syn_post_trace,
    double* syn_r_pre, double* syn_r_post, double* syn_theta_m,
    // Synapse Params (read-only)
    const double* syn_g_max, const double* syn_E_rev, const double* syn_tau_decay,
    const double* syn_A_plus, const double* syn_A_minus, const double* syn_tau_pre, const double* syn_tau_post,
    const double* syn_eta_bcm, const double* syn_tau_rate, const double* syn_tau_theta,
    const double* syn_w_min, const double* syn_w_max, const double* syn_delay,
    // Connectivity
    const int* syn_pre_idx, const int* syn_post_idx,
    // Spike info
    const int* spiked_now,
    // Output current accumulation
    double* I_syn_total,
    const double* V_soma
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= N_SYNAPSES) return;

    int pre_id = syn_pre_idx[sid];
    int post_id = syn_post_idx[sid];
    bool pre_spiked = (spiked_now[pre_id] == 1);
    bool post_spiked = (spiked_now[post_id] == 1);

    double g = syn_g[sid];
    double w = syn_w[sid];
    double pre_trace = syn_pre_trace[sid];
    double post_trace = syn_post_trace[sid];
    double r_pre = syn_r_pre[sid];
    double r_post = syn_r_post[sid];
    double theta_m = syn_theta_m[sid];

    // Update traces and rates
    pre_trace *= exp(-dt / syn_tau_pre[sid]);
    post_trace *= exp(-dt / syn_tau_post[sid]);
    if (pre_spiked) pre_trace += 1.0;
    if (post_spiked) post_trace += 1.0;

    double decay_r = exp(-dt / syn_tau_rate[sid]);
    r_pre *= decay_r;
    r_post *= decay_r;
    if (pre_spiked) r_pre += 1.0 / dt;
    if (post_spiked) r_post += 1.0 / dt;

    theta_m += (dt / syn_tau_theta[sid]) * (r_post * r_post - theta_m);

    // Plasticity
    double dw = 0.0;
    if (pre_spiked) dw += syn_A_plus[sid] * post_trace;
    if (post_spiked) dw -= syn_A_minus[sid] * pre_trace;
    double dw_bcm_dt = syn_eta_bcm[sid] * r_pre * r_post * (r_post - theta_m);
    dw += dw_bcm_dt * dt;
    w = fmin(syn_w_max[sid], fmax(syn_w_min[sid], w + dw));

    // Conductance decay + apply pending increment
    g *= exp(-dt / syn_tau_decay[sid]);
    int buffer_offset = buffer_idx * n_synapses + sid;
    double pending = g_incr_buffer[buffer_offset];
    g += pending;
    g_incr_buffer[buffer_offset] = 0.0;

    // Schedule future increment if pre neuron spiked
    if (pre_spiked) {
        int delay_steps = (int)ceil(syn_delay[sid] / dt);
        if (delay_steps < 1) delay_steps = 1;
        int target = (buffer_idx + delay_steps) % buffer_depth;
        int target_offset = target * n_synapses + sid;
        g_incr_buffer[target_offset] = syn_g_max[sid] * w;
    }

    double I_syn = g * (syn_E_rev[sid] - V_soma[post_id]);
    atomicAdd(&I_syn_total[post_id], I_syn);

    // Write back
    syn_g[sid] = g;
    syn_w[sid] = w;
    syn_pre_trace[sid] = pre_trace;
    syn_post_trace[sid] = post_trace;
    syn_r_pre[sid] = r_pre;
    syn_r_post[sid] = r_post;
    syn_theta_m[sid] = theta_m;
}
''', 'update_synapses')


class CuPyNetwork:
    """
    A GPU-accelerated network of pyramidal neurons using CuPy and custom CUDA kernels.
    Supports optional batch replication: the same graph replicated `batch_size` times
    with shared parameters but independent state.
    """
    def __init__(self, dt: float = 1e-4, batch_size: int = 1):
        self.dt = np.float64(dt)
        self.time = np.float64(0.0)
        self.batch_size = int(max(1, batch_size))

        # Logical counts
        self.neurons_per_batch = 0
        self.synapses_per_batch = 0
        # Flattened counts (all batches)
        self.n_neurons = 0
        self.n_synapses = 0

        self.neuron_params = {}
        self.neuron_states = {}
        
        self.synapse_params = {}
        self.synapse_states = {}

        self.spike_monitors = {}
        self._has_neurons = False

    def add_neurons(self, num_neurons: int, soma_p: SomaParams, ais_p: AISParams, coup_p: CouplingParams):
        """Adds a population of identical neurons to the network (single population supported)."""
        if self.n_synapses > 0:
            raise RuntimeError("Add all neurons before adding synapses.")
        if self._has_neurons:
            raise RuntimeError("Multiple neuron populations are not supported in batch mode.")
        
        self.neurons_per_batch = num_neurons
        self.n_neurons = num_neurons * self.batch_size
        self._has_neurons = True

        # Combine all params into a single dict for easier handling
        params_dict = {
            **{f'soma_{k}': v for k, v in asdict(soma_p).items()},
            **{f'ais_{k}': v for k, v in asdict(ais_p).items()},
            **{f'coup_{k}': v for k, v in asdict(coup_p).items()},
        }
        
        # Initialize or extend parameter arrays
        for key, val in params_dict.items():
            arr_single = cp.full(num_neurons, val, dtype=cp.float64)
            arr = cp.tile(arr_single, self.batch_size)
            self.neuron_params[key] = arr

        # Initialize or extend state arrays
        state_init = {
            'V_soma': cp.full(num_neurons, ais_p.E_L, dtype=cp.float64),
            'V_ais': cp.full(num_neurons, ais_p.E_L, dtype=cp.float64),
            'refrac_until': cp.full(num_neurons, -1e9, dtype=cp.float64),
        }
        for key, val_arr in state_init.items():
            self.neuron_states[key] = cp.tile(val_arr, self.batch_size)

    def connect(self, pre_indices: list, post_indices: list, syn_p: SynapseParams, w_init: float | np.ndarray = 1.0):
        """Connects neurons with synapses having identical parameters (replicated per batch)."""
        if not self._has_neurons:
            raise RuntimeError("Add neurons before connecting synapses.")
        if len(pre_indices) != len(post_indices):
            raise ValueError("pre_indices and post_indices must have the same length")
        num_synapses_one = len(pre_indices)
        if num_synapses_one == 0:
            return

        self.synapses_per_batch = num_synapses_one
        self.n_synapses = num_synapses_one * self.batch_size

        # Create or extend connectivity arrays
        conn_pre = []
        conn_post = []
        for b in range(self.batch_size):
            offset = b * self.neurons_per_batch
            conn_pre.append(cp.array(pre_indices, dtype=cp.int32) + offset)
            conn_post.append(cp.array(post_indices, dtype=cp.int32) + offset)
        self.synapse_params['syn_pre_idx'] = cp.concatenate(conn_pre)
        self.synapse_params['syn_post_idx'] = cp.concatenate(conn_post)

        # Create or extend synapse parameter arrays
        params_dict = {f'syn_{k}': v for k, v in asdict(syn_p).items()}
        for key, val in params_dict.items():
            arr_one = cp.full(num_synapses_one, val, dtype=cp.float64)
            arr = cp.tile(arr_one, self.batch_size)
            self.synapse_params[key] = arr

        # Create or extend synapse state arrays
        if isinstance(w_init, (int, float)):
            w_init_arr = cp.full(num_synapses_one, w_init, dtype=cp.float64)
            w_init_arr = cp.tile(w_init_arr, self.batch_size)
        else:
            w_init_arr = cp.array(w_init, dtype=cp.float64)
            if w_init_arr.shape[0] != num_synapses_one:
                raise ValueError("w_init length must match number of synapses per batch")
            w_init_arr = cp.tile(w_init_arr, self.batch_size)

        # neuro_pyramidal_cupy.py 파일의 connect 메서드

        state_init = {
            'syn_g': cp.zeros(self.n_synapses, dtype=cp.float64),
            'syn_w': w_init_arr,
            'syn_pre_trace': cp.zeros(self.n_synapses, dtype=cp.float64),
            'syn_post_trace': cp.zeros(self.n_synapses, dtype=cp.float64),
            'syn_r_pre': cp.zeros(self.n_synapses, dtype=cp.float64),
            'syn_r_post': cp.zeros(self.n_synapses, dtype=cp.float64),
            'syn_theta_m': cp.full(self.n_synapses, 1.0, dtype=cp.float64),
        # 'pending_g_incr': cp.zeros(self.n_synapses, dtype=cp.float64), # <-- REMOVE OR COMMENT OUT THIS LINE
        }
        
        for key, val_arr in state_init.items():
            self.synapse_states[key] = cp.tile(val_arr, self.batch_size)

    def _prepare_run(self):
        """Prepare temporary arrays needed for simulation."""
        self.comm_arrays = {
            'spiked_now': cp.zeros(self.n_neurons, dtype=cp.int32),
            'I_syn_total': cp.zeros(self.n_neurons, dtype=cp.float64)
        }
        # This handles the delayed conductance increment. We use a circular buffer.
        if self.n_synapses > 0:
            delay_steps = max(1, int(np.ceil(self.synapse_params['syn_delay'].max().get() / self.dt)))
            self.g_incr_buffer = cp.zeros((delay_steps, self.n_synapses), dtype=cp.float64)
            self.g_incr_buffer_idx = 0
        else:
            self.g_incr_buffer = cp.zeros((1, 1), dtype=cp.float64)
            self.g_incr_buffer_idx = 0


    def step(self):
        """Advance the network state by one time step `dt`."""
        if not hasattr(self, 'comm_arrays'):
            self._prepare_run()

        threads_per_block = 256
        neuron_blocks = (self.n_neurons + threads_per_block - 1) // threads_per_block if self.n_neurons else 0
        synapse_blocks = (self.n_synapses + threads_per_block - 1) // threads_per_block if self.n_synapses else 0

        # 1) Neuron update uses currents accumulated in previous step
        if neuron_blocks:
            _NEURON_KERNEL(
                (neuron_blocks,),
                (threads_per_block,),
                (
                    self.n_neurons,
                    self.time,
                    self.dt,
                    self.neuron_states['V_soma'],
                    self.neuron_states['V_ais'],
                    self.neuron_states['refrac_until'],
                    self.neuron_params['soma_C_m'], self.neuron_params['soma_g_L'], self.neuron_params['soma_E_L'], self.neuron_params['soma_I_ext'],
                    self.neuron_params['ais_C_m'], self.neuron_params['ais_g_L'], self.neuron_params['ais_E_L'], self.neuron_params['ais_V_T'],
                    self.neuron_params['ais_Delta_T'], self.neuron_params['ais_V_spike'], self.neuron_params['ais_V_reset'], self.neuron_params['ais_t_refrac'],
                    self.neuron_params['coup_g_c'],
                    self.comm_arrays['I_syn_total'],
                    self.comm_arrays['spiked_now'],
                ),
            )

        # 2) Prepare for next synapse accumulation
        self.comm_arrays['I_syn_total'].fill(0)

        # 3) Synapse update + plasticity + delay scheduling
        if synapse_blocks:
            buffer_depth = self.g_incr_buffer.shape[0]
            _SYNAPSE_KERNEL(
                (synapse_blocks,),
                (threads_per_block,),
                (
                    self.n_synapses,
                    self.dt,
                    self.g_incr_buffer_idx,
                    buffer_depth,
                    self.n_synapses,
                    self.g_incr_buffer,
                    self.synapse_states['syn_g'], self.synapse_states['syn_w'], self.synapse_states['syn_pre_trace'], self.synapse_states['syn_post_trace'],
                    self.synapse_states['syn_r_pre'], self.synapse_states['syn_r_post'], self.synapse_states['syn_theta_m'],
                    self.synapse_params['syn_g_max'], self.synapse_params['syn_E_rev'], self.synapse_params['syn_tau_decay'],
                    self.synapse_params['syn_A_plus'], self.synapse_params['syn_A_minus'], self.synapse_params['syn_tau_pre'], self.synapse_params['syn_tau_post'],
                    self.synapse_params['syn_eta_bcm'], self.synapse_params['syn_tau_rate'], self.synapse_params['syn_tau_theta'],
                    self.synapse_params['syn_w_min'], self.synapse_params['syn_w_max'], self.synapse_params['syn_delay'],
                    self.synapse_params['syn_pre_idx'], self.synapse_params['syn_post_idx'],
                    self.comm_arrays['spiked_now'],
                    self.comm_arrays['I_syn_total'],
                    self.neuron_states['V_soma'],
                ),
            )
            self.g_incr_buffer_idx = (self.g_incr_buffer_idx + 1) % buffer_depth

        # Record spikes if monitoring
        if self.spike_monitors:
            spiking_neurons = cp.where(self.comm_arrays['spiked_now'])[0].get()
            for neuron_idx in spiking_neurons:
                if neuron_idx in self.spike_monitors:
                    self.spike_monitors[neuron_idx].append(self.time)

        self.time += self.dt

    def run(self, T: float, monitor_spikes_for: List[int] = 0):
        """Run simulation for a total duration T."""
        print(f"Preparing to run simulation for {T:.2f} s...")
        self.spike_monitors = {idx: [] for idx in (monitor_spikes_for or [])}
        steps = int(np.round(T / self.dt))
        self._prepare_run()
        
        for i in range(steps):
            self.step()
            if steps >= 10 and (i + 1) % max(1, steps // 10) == 0:
                print(f"  ... Progress: {100 * (i+1)/steps:.0f}%")
        print("Simulation finished.")

    def get_spike_times(self) -> Dict[int, List[float]]:
        return self.spike_monitors

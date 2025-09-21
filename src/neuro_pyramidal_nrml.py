from __future__ import annotations

import numpy as np
import cupy as cp
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

# ----- 단위계: mV, ms, pA, nS, pF -----
@dataclass
class SomaParams:
    """Soma Parameters"""
    C_m: float = 200.0     # Membrane capacitance (pF)
    g_L: float = 10.0      # Leak conductance (nS)
    E_L: float = -70.0     # Leak reversal potential (mV)
    I_ext: float = 0.0     # External current (nA)

@dataclass
class AISParams:
    """Axon Initial Segment (AIS) Parameters"""
    C_m: float = 20.0      # Membrane capacitance (pF)
    g_L: float = 1.0       # Leak conductance (nS)
    E_L: float = -70.0     # Leak reversal potential (mV)
    V_T: float = -50.0     # Spike threshold (mV)
    Delta_T: float = 2.0   # Sharpness of spike initiation (mV)
    V_spike: float = 20.0  # Voltage at spike peak (mV)
    V_reset: float = -65.0 # Reset potential after spike (mV)
    t_refrac: float = 2.0  # Refractory period (ms)

@dataclass
class CouplingParams:
    """Coupling between Soma and AIS"""
    g_c: float = 30.0      # Coupling conductance (nS)

@dataclass
class SynapseParams:
    """Synapse Parameters"""
    g_max: float = 1.0         # Max conductance (nS)
    E_rev: float = 0.0         # Reversal potential (mV)
    tau_decay: float = 5.0     # Conductance decay time constant (ms)
    A_plus: float = 0.02       # STDP potentiation factor
    A_minus: float = 0.022     # STDP depression factor (slightly > A_plus)
    tau_pre: float = 20.0      # Pre-synaptic trace time constant (ms)
    tau_post: float = 20.0     # Post-synaptic trace time constant (ms)
    eta_bcm: float = 0.0       # BCM learning rate (set to 0 to disable)
    tau_rate: float = 50.0     # Rate detector time constant (ms)
    tau_theta: float = 1.0e3   # BCM threshold time constant (ms)
    w_min: float = 0.0         # Minimum synaptic weight
    w_max: float = 1.0         # Maximum synaptic weight
    # delay: float = 1.0      # BUGGY: Delay mechanism removed for now

# ----- CUDA 커널 -----

_NEURON_SYNAPSE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void update_neurons_and_synapses(
    const int N_NEURONS, const int N_SYNAPSES,
    double t, double dt,
    // Neuron States (read/write)
    double* V_soma, double* V_ais, double* refrac_until,
    // Neuron Params (read-only)
    const double* soma_C_m, const double* soma_g_L, const double* soma_E_L, const double* soma_I_ext,
    const double* ais_C_m, const double* ais_g_L, const double* ais_E_L, const double* ais_V_T,
    const double* ais_Delta_T, const double* ais_V_spike, const double* ais_V_reset, const double* ais_t_refrac,
    const double* coup_g_c,
    // Synapse States (read/write)
    double* syn_g, double* syn_w, double* syn_pre_trace, double* syn_post_trace,
    double* syn_r_pre, double* syn_r_post, double* syn_theta_m,
    // Synapse Params (read-only)
    const double* syn_g_max, const double* syn_E_rev, const double* syn_tau_decay,
    const double* syn_A_plus, const double* syn_A_minus, const double* syn_tau_pre, const double* syn_tau_post,
    const double* syn_eta_bcm, const double* syn_tau_rate, const double* syn_tau_theta,
    const double* syn_w_min, const double* syn_w_max,
    // Connectivity
    const int* syn_pre_idx, const int* syn_post_idx,
    // Spike communication arrays
    int* spiked_now, double* I_syn_total
) {
    // =========================================================================
    // PART 1: NEURON UPDATE (RK4 Integration and Spike Detection)
    // =========================================================================
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < N_NEURONS) {
        // --- Load states and params into registers ---
        double v_s = V_soma[nid];
        double v_a = V_ais[nid];
        bool in_refrac = (t < refrac_until[nid]);
        
        // --- Integrate (or handle refractory period) ---
        if (!in_refrac) {
            // RK4 integration
            double i_syn = I_syn_total[nid]; // Use synaptic current from previous time step
            
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
            double i_syn = I_syn_total[nid];

            // RK4 for soma only
            auto k_s = [&](double vs_local) {
                return (soma_g_L[nid] * (soma_E_L[nid] - vs_local) + coup_g_c[nid] * (v_a_fixed - vs_local) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
            };
            double k1 = k_s(v_s);
            double k2 = k_s(v_s + 0.5 * dt * k1);
            double k3 = k_s(v_s + 0.5 * dt * k2);
            double k4 = k_s(v_s + dt * k3);
            V_soma[nid] = v_s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            V_ais[nid] = v_a_fixed;
        }

        // --- Spike detection and reset ---
        if (!in_refrac && V_ais[nid] >= ais_V_spike[nid]) {
            spiked_now[nid] = 1;
            V_ais[nid] = ais_V_reset[nid];
            V_soma[nid] = ais_V_reset[nid]; // BUG FIX: Soma 전압도 함께 리셋하여 억제성 전류 발생 방지
            refrac_until[nid] = t + ais_t_refrac[nid];
        } else {
            spiked_now[nid] = 0;
        }
    }
    
    __syncthreads(); // Ensure all neuron updates (and spike flags) are visible to synapse part

    // =========================================================================
    // PART 2: SYNAPSE UPDATE (Plasticity and Current Calculation)
    // =========================================================================
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid < N_SYNAPSES) {
        // --- Get pre/post neuron indices ---
        int pre_id = syn_pre_idx[sid];
        int post_id = syn_post_idx[sid];
        
        // --- Check for spikes ---
        bool pre_spiked = (spiked_now[pre_id] == 1);
        bool post_spiked = (spiked_now[post_id] == 1);

        // --- Load synapse states ---
        double g = syn_g[sid];
        double w = syn_w[sid];
        double pre_trace = syn_pre_trace[sid];
        double post_trace = syn_post_trace[sid];
        double r_pre = syn_r_pre[sid];
        double r_post = syn_r_post[sid];
        double theta_m = syn_theta_m[sid];
        
        // --- Update traces and rates ---
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
        
        // --- Apply plasticity rules (STDP & BCM) ---
        double dw = 0.0;
        // STDP (Corrected Logic)
        // On post-spike, potentiate based on pre-trace (causal, pre-then-post)
        if (post_spiked) {
            dw += syn_A_plus[sid] * pre_trace;
        }
        // On pre-spike, depress based on post-trace (anti-causal, post-then-pre)
        if (pre_spiked) {
            dw -= syn_A_minus[sid] * post_trace;
        }
        
        // BCM (continuous)
        double dw_bcm_dt = syn_eta_bcm[sid] * r_pre * r_post * (r_post - theta_m);
        dw += dw_bcm_dt * dt;
        
        w = fmin(syn_w_max[sid], fmax(syn_w_min[sid], w + dw));

        // --- Update conductance ---
        // Decay
        g *= exp(-dt / syn_tau_decay[sid]);
        // If presynaptic neuron spiked now, add conductance instantly (delay mechanism removed to fix bug)
        if (pre_spiked) {
            g += syn_g_max[sid] * w;
        }
        
        // --- Calculate postsynaptic current (for NEXT time step) ---
        // Current flows INTO the postsynaptic neuron.
        double I_syn = g * (syn_E_rev[sid] - V_soma[post_id]);
        atomicAdd(&I_syn_total[post_id], I_syn);

        // --- Write back updated synapse states ---
        syn_g[sid] = g;
        syn_w[sid] = w;
        syn_pre_trace[sid] = pre_trace;
        syn_post_trace[sid] = post_trace;
        syn_r_pre[sid] = r_pre;
        syn_r_post[sid] = r_post;
        syn_theta_m[sid] = theta_m;
    }
}
''', 'update_neurons_and_synapses')


class CuPyNetwork:
    """
    A GPU-accelerated network of pyramidal neurons using CuPy and a custom CUDA kernel.
    Manages all neuron and synapse states as large CuPy arrays for efficient parallel processing.
    """
    def __init__(self, dt: float = 0.1):
        self.dt = np.float64(dt)
        self.time = np.float64(0.0)
        
        self.n_neurons = 0
        self.n_synapses = 0

        self.neuron_params = {}
        self.neuron_states = {}
        
        self.synapse_params = {}
        self.synapse_states = {}

        self.spike_monitors = {}

    def add_neurons(self, num_neurons: int, soma_p: SomaParams, ais_p: AISParams, coup_p: CouplingParams):
        """Adds a population of identical neurons to the network."""
        if self.n_synapses > 0:
            raise RuntimeError("Add all neurons before adding synapses.")
        
        start_idx = self.n_neurons
        self.n_neurons += num_neurons

        # Combine all params into a single dict for easier handling
        params_dict = {
            **{f'soma_{k}': v for k, v in asdict(soma_p).items()},
            **{f'ais_{k}': v for k, v in asdict(ais_p).items()},
            **{f'coup_{k}': v for k, v in asdict(coup_p).items()},
        }
        
        # Initialize or extend parameter arrays
        for key, val in params_dict.items():
            arr = cp.full(num_neurons, val, dtype=cp.float64)
            if key not in self.neuron_params:
                self.neuron_params[key] = arr
            else:
                self.neuron_params[key] = cp.concatenate([self.neuron_params[key], arr])

        # Initialize or extend state arrays
        state_init = {
            'V_soma': cp.full(num_neurons, ais_p.E_L, dtype=cp.float64),
            'V_ais': cp.full(num_neurons, ais_p.E_L, dtype=cp.float64),
            'refrac_until': cp.full(num_neurons, -1e9, dtype=cp.float64),
        }
        for key, val_arr in state_init.items():
            if key not in self.neuron_states:
                self.neuron_states[key] = val_arr
            else:
                self.neuron_states[key] = cp.concatenate([self.neuron_states[key], val_arr])

    def connect(self, pre_indices: list, post_indices: list, syn_p: SynapseParams, w_init: float | np.ndarray = 1.0):
        """Connects neurons with synapses having identical parameters."""
        num_synapses = len(pre_indices)
        if num_synapses == 0: return
        
        start_idx = self.n_synapses
        self.n_synapses += num_synapses

        # Create or extend connectivity arrays
        conn = {'syn_pre_idx': pre_indices, 'syn_post_idx': post_indices}
        for key, val in conn.items():
            arr = cp.array(val, dtype=cp.int32)
            if key not in self.synapse_params:
                self.synapse_params[key] = arr
            else:
                self.synapse_params[key] = cp.concatenate([self.synapse_params[key], arr])

        # Create or extend synapse parameter arrays
        params_dict = {f'syn_{k}': v for k, v in asdict(syn_p).items()}
        for key, val in params_dict.items():
            # Skip delay if it exists, as it's removed from the kernel
            if key == 'syn_delay': continue
            arr = cp.full(num_synapses, val, dtype=cp.float64)
            if key not in self.synapse_params:
                self.synapse_params[key] = arr
            else:
                self.synapse_params[key] = cp.concatenate([self.synapse_params[key], arr])

        # Create or extend synapse state arrays
        if isinstance(w_init, (int, float)):
            w_init_arr = cp.full(num_synapses, w_init, dtype=cp.float64)
        else:
            w_init_arr = cp.array(w_init, dtype=cp.float64)

        state_init = {
            'syn_g': cp.zeros(num_synapses, dtype=cp.float64),
            'syn_w': w_init_arr,
            'syn_pre_trace': cp.zeros(num_synapses, dtype=cp.float64),
            'syn_post_trace': cp.zeros(num_synapses, dtype=cp.float64),
            'syn_r_pre': cp.zeros(num_synapses, dtype=cp.float64),
            'syn_r_post': cp.zeros(num_synapses, dtype=cp.float64),
            'syn_theta_m': cp.full(num_synapses, 1.0, dtype=cp.float64),
        }
        
        for key, val_arr in state_init.items():
            if key not in self.synapse_states:
                self.synapse_states[key] = val_arr
            else:
                self.synapse_states[key] = cp.concatenate([self.synapse_states[key], val_arr])

    def _prepare_run(self):
        """Prepare temporary arrays needed for simulation."""
        self.comm_arrays = {
            'spiked_now': cp.zeros(self.n_neurons, dtype=cp.int32),
            'I_syn_total': cp.zeros(self.n_neurons, dtype=cp.float64)
        }

        # If no synapses exist, create empty placeholder arrays to prevent KeyError during kernel launch
        if self.n_synapses == 0:
            # These are all the synapse-related arrays the kernel expects.
            # We create them as empty arrays so the kernel call doesn't fail.
            # The kernel itself won't access them since N_SYNAPSES will be 0.
            syn_p = SynapseParams()
            param_keys = [f'syn_{k}' for k in asdict(syn_p).keys()]
            param_keys.extend(['syn_pre_idx', 'syn_post_idx'])
            
            state_keys = ['syn_g', 'syn_w', 'syn_pre_trace', 'syn_post_trace', 
                          'syn_r_pre', 'syn_r_post', 'syn_theta_m']

            for key in param_keys:
                if key not in self.synapse_params:
                    # The kernel expects int32 for indices
                    if key == 'syn_delay': continue # delay is removed
                    dtype = cp.int32 if 'idx' in key else cp.float64
                    self.synapse_params[key] = cp.array([], dtype=dtype)
            
            for key in state_keys:
                if key not in self.synapse_states:
                    self.synapse_states[key] = cp.array([], dtype=cp.float64)

    def step(self):
        """Advance the network state by one time step `dt`."""
        if not hasattr(self, 'comm_arrays'):
            self._prepare_run()

        # Reset synaptic current accumulator for the next step's calculation
        self.comm_arrays['I_syn_total'].fill(0)

        # Determine grid and block sizes for the kernel launch
        threads_per_block = 256
        max_n = max(self.n_neurons, self.n_synapses)
        blocks = (max_n + threads_per_block - 1) // threads_per_block

        # This list MUST EXACTLY match the kernel signature
        kernel_args = [
            # Scalar params
            self.n_neurons, self.n_synapses, self.time, self.dt,
            # Neuron states (read/write)
            self.neuron_states['V_soma'], self.neuron_states['V_ais'], self.neuron_states['refrac_until'],
            # Neuron Params (read-only)
            self.neuron_params['soma_C_m'], self.neuron_params['soma_g_L'], self.neuron_params['soma_E_L'], self.neuron_params['soma_I_ext'],
            self.neuron_params['ais_C_m'], self.neuron_params['ais_g_L'], self.neuron_params['ais_E_L'], self.neuron_params['ais_V_T'],
            self.neuron_params['ais_Delta_T'], self.neuron_params['ais_V_spike'], self.neuron_params['ais_V_reset'], self.neuron_params['ais_t_refrac'],
            self.neuron_params['coup_g_c'],
            # Synapse States (read/write)
            self.synapse_states['syn_g'], self.synapse_states['syn_w'], self.synapse_states['syn_pre_trace'], self.synapse_states['syn_post_trace'],
            self.synapse_states['syn_r_pre'], self.synapse_states['syn_r_post'], self.synapse_states['syn_theta_m'],
            # Synapse Params (read-only)
            self.synapse_params['syn_g_max'], self.synapse_params['syn_E_rev'], self.synapse_params['syn_tau_decay'],
            self.synapse_params['syn_A_plus'], self.synapse_params['syn_A_minus'], self.synapse_params['syn_tau_pre'], self.synapse_params['syn_tau_post'],
            self.synapse_params['syn_eta_bcm'], self.synapse_params['syn_tau_rate'], self.synapse_params['syn_tau_theta'],
            self.synapse_params['syn_w_min'], self.synapse_params['syn_w_max'],
            # Connectivity
            self.synapse_params['syn_pre_idx'], self.synapse_params['syn_post_idx'],
            # Communication arrays
            self.comm_arrays['spiked_now'], self.comm_arrays['I_syn_total'],
        ]
        
        _NEURON_SYNAPSE_KERNEL((blocks,), (threads_per_block,), tuple(kernel_args))

        # Record spikes if monitoring
        if self.spike_monitors:
            spiking_neurons = cp.where(self.comm_arrays['spiked_now'])[0].get()
            for neuron_idx in spiking_neurons:
                if neuron_idx in self.spike_monitors:
                    self.spike_monitors[neuron_idx].append(self.time)

        self.time += self.dt

    def run(self, T: float, monitor_spikes_for: List[int] = 0):
        """Run simulation for a total duration T."""
        print(f"Preparing to run simulation for {T/1000:.2f} s...")
        self.spike_monitors = {idx: [] for idx in (monitor_spikes_for or [])}
        steps = int(np.round(T / self.dt))
        self._prepare_run()
        
        for i in range(steps):
            self.step()
            if (i+1) % (steps // 20) == 0: # Print more frequently
                print(f"  ... Progress: {100 * (i+1)/steps:.0f}%")
        print("Simulation finished.")

    def get_spike_times(self) -> Dict[int, List[float]]:
        return self.spike_monitors
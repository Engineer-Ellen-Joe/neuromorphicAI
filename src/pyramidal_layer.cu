/*
Vectorized CUDA kernels for the PyramidalLayer class.

Each kernel is designed to operate on the entire layer of neurons in parallel,
where the layer's state is represented by large matrices (e.g., weights, traces).
The grid dimensions for launching these kernels will typically correspond to
the dimensions of these state matrices.
*/

extern "C" __global__
void stdp_update_layer(
  float *weights,          // Shape: (num_neurons, num_afferents)
  const float *pre_trace,    // Shape: (num_neurons, num_afferents)
  const float *post_trace,   // Shape: (num_neurons, num_afferents)
  const float *pre_spikes,   // Shape: (num_afferents,) - Broadcasted
  const float *axon_spikes,  // Shape: (num_neurons,) - Broadcasted
  const float a_plus,
  const float a_minus,
  const float w_min,
  const float w_max,
  const float learning_rate,
  const int num_neurons,
  const int num_afferents
) {
  int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int afferent_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (neuron_idx >= num_neurons || afferent_idx >= num_afferents) {
    return;
  }

  // Linear index for neuron-specific arrays
  long long flat_idx = (long long)neuron_idx * num_afferents + afferent_idx;

  // A neuron's spike is a single value for all its afferent synapses
  float post_spike = axon_spikes[neuron_idx];
  
  // A presynaptic spike is a single value for that afferent across all neurons
  float pre_spike = pre_spikes[afferent_idx];

  float dw = learning_rate * (
    a_plus * pre_trace[flat_idx] * post_spike
    - a_minus * post_trace[flat_idx] * pre_spike
  );

  float w = weights[flat_idx] + dw;
  if (w < w_min) {
    w = w_min;
  } else if (w > w_max) {
    w = w_max;
  }
  weights[flat_idx] = w;
}

extern "C" __global__
void propagate_branch_layer(
  const float *axon_spikes,      // Shape: (num_neurons,)
  const float axon_spike_current,
  const float conduction_threshold,
  const float *safety_factors,   // Shape: (num_neurons, num_branches)
  float *branch_currents,      // Shape: (num_neurons, num_branches)
  const int num_neurons,
  const int num_branches
) {
  int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (neuron_idx >= num_neurons || branch_idx >= num_branches) {
    return;
  }

  long long flat_idx = (long long)neuron_idx * num_branches + branch_idx;
  
  if (num_branches <= 0) {
    return;
  }

  float axon_spike = axon_spikes[neuron_idx];
  float distributed = axon_spike * axon_spike_current / (float)num_branches;
  float conduction_metric = distributed * safety_factors[flat_idx];

  if (conduction_metric >= conduction_threshold) {
    branch_currents[flat_idx] = distributed;
  } else {
    branch_currents[flat_idx] = 0.0;
  }
}

extern "C" __global__
void bcm_ltp_layer(
  float *theta,                // Shape: (num_neurons, num_branches)
  float *weights,              // Shape: (num_neurons, num_branches)
  const float *activity,       // Shape: (num_neurons, num_branches)
  const float tau_inv,
  const float eta,
  const float w_min,
  const float w_max,
  const float dt,
  const int num_neurons,
  const int num_branches
) {
  int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (neuron_idx >= num_neurons || branch_idx >= num_branches) {
    return;
  }

  long long flat_idx = (long long)neuron_idx * num_branches + branch_idx;

  float current_activity = activity[flat_idx];
  float current_theta = theta[flat_idx];

  float theta_new = current_theta + dt * tau_inv * (
    current_activity * current_activity - current_theta
  );
  theta[flat_idx] = theta_new;

  float dw = eta * current_activity * (current_activity - theta_new) * dt;
  float w = weights[flat_idx] + dw;

  if (w < w_min) {
    w = w_min;
  } else if (w > w_max) {
    w = w_max;
  }
  weights[flat_idx] = w;
}

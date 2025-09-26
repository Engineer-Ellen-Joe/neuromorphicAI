/*
Vectorized CUDA kernels for the PyramidalLayer class.

Each kernel is designed to operate on the entire layer of neurons in parallel,
where the layer's state is represented by large matrices (e.g., weights, traces).
The grid dimensions for launching these kernels will typically correspond to
the dimensions of these state matrices.
*/

extern "C" __global__ void stdp_update_layer(
    float *weights,           // Shape: (num_neurons, num_afferents)
    const float *pre_trace,   // Shape: (num_neurons, num_afferents)
    const float *post_trace,  // Shape: (num_neurons, num_afferents)
    const float *pre_spikes,  // Shape: (num_afferents,) - Broadcasted
    const float *axon_spikes, // Shape: (num_neurons,) - Broadcasted
    const float a_plus,
    const float a_minus,
    const float w_min,
    const float w_max,
    const float learning_rate,
    const int num_neurons,
    const int num_afferents)
{
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

  float dw = learning_rate * (a_plus * pre_trace[flat_idx] * post_spike - a_minus * post_trace[flat_idx] * pre_spike);

  float w = weights[flat_idx] + dw;
  if (w < w_min) {
    w = w_min;
  }
  else if (w > w_max) {
    w = w_max;
  }
  weights[flat_idx] = w;
}

extern "C" __global__ void propagate_branch_layer(
    const float *axon_spikes, // Shape: (num_neurons,)
    const float axon_spike_current,
    const float conduction_threshold,
    const float *safety_factors, // Shape: (num_neurons, num_branches)
    float *branch_currents,      // Shape: (num_neurons, num_branches)
    const int num_neurons,
    const int num_branches)
{
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

extern "C" __global__ void bcm_ltp_layer(
    float *theta,          // Shape: (num_neurons, num_branches)
    float *weights,        // Shape: (num_neurons, num_branches)
    const float *activity, // Shape: (num_neurons, num_branches)
    const float tau_inv,
    const float eta,
    const float w_min,
    const float w_max,
    const float dt,
    const int num_neurons,
    const int num_branches)
{
  int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (neuron_idx >= num_neurons || branch_idx >= num_branches) {
    return;
  }

  long long flat_idx = (long long)neuron_idx * num_branches + branch_idx;

  float current_activity = activity[flat_idx];
  float current_theta = theta[flat_idx];

  float theta_new = current_theta + dt * tau_inv * (current_activity * current_activity - current_theta);
  theta[flat_idx] = theta_new;

  float dw = eta * current_activity * (current_activity - theta_new) * dt;
  float w = weights[flat_idx] + dw;

  if (w < w_min) {
    w = w_min;
  }
  else if (w > w_max) {
    w = w_max;
  }
  weights[flat_idx] = w;
}

extern "C" __global__ void competitive_step_kernel(
    // State arrays
    float *membrane_potential, // (num_neurons)
    float *refractory_timers,  // (num_neurons)
    float *axon_spikes,        // (num_neurons) - Output

    // Input arrays
    const float *synaptic_drive,   // (num_neurons)
    const float *external_current, // (num_neurons)

    // Parameters
    const float dt,
    const float leak_potential,
    const float reset_potential,
    const float ais_threshold,
    const float ais_slope,
    const float ais_activation_gate,
    const float refractory_period,
    const float membrane_time_constant,
    const float membrane_capacitance,
    const float inhibition_current, // Inhibition current for losers

    // Synchronization
    int *winner_idx_global, // A single integer on the GPU, initialized to -1

    // Grid size
    const int num_neurons)
{
  int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (neuron_idx >= num_neurons) { return; }

  // --- 1. Update refractory timer ---
  float refractory_timer = refractory_timers[neuron_idx];
  if (refractory_timer > 0.0f) {
    refractory_timer -= dt;
    refractory_timers[neuron_idx] = max(0.0f, refractory_timer);
  }

  // --- 2. Update membrane potential ---
  bool is_active = (refractory_timer <= 0.0f);
  float potential = membrane_potential[neuron_idx];

  if (is_active) {
    float leak_term = (leak_potential - potential) / membrane_time_constant;
    float drive_term = (synaptic_drive[neuron_idx] + external_current[neuron_idx]) / membrane_capacitance;
    potential += dt * (leak_term + drive_term);
  } else {
    potential = reset_potential; // Clamp potential during refractory period
  }
  membrane_potential[neuron_idx] = potential;

  // --- 3. Check for potential spikes and compete ---
  float activation = 1.0f / (1.0f + expf(-(potential - ais_threshold) / ais_slope));

  // Clear previous spike state
  axon_spikes[neuron_idx] = 0.0f;

  if (is_active && activation >= ais_activation_gate) {
    // Try to become the winner using an atomic Compare-and-Swap.
    // If *winner_idx_global was -1, set it to my index (neuron_idx) and return the old value (-1).
    // This ensures only the very first thread to execute this line becomes the winner.
    if (atomicCAS(winner_idx_global, -1, neuron_idx) == -1) {
      // I am the winner
    }
  }

  // --- 4. Synchronize all threads ---
  // Ensure all threads have finished the competition phase before proceeding.
  __syncthreads();

  // --- 5. Finalize state based on who won ---
  int winner = *winner_idx_global;
  if (winner != -1) { // A winner was chosen
    if (neuron_idx == winner) {
      // I am the winner, so I spike.
      axon_spikes[neuron_idx] = 1.0f;
      membrane_potential[neuron_idx] = reset_potential;
      refractory_timers[neuron_idx] = refractory_period;
    } else {
      // I am a loser, apply inhibition.
      // This is a simplified inhibition model where we just add the current.
      // A more complex model could directly alter the potential.
      membrane_potential[neuron_idx] += dt * (inhibition_current / membrane_capacitance);
    }
  }
}
extern "C" __global__
void stdp_update(
  double *weights,
  const double *pre_trace,
  const double *post_trace,
  const double *pre_spikes,
  const double post_spike,
  const double a_plus,
  const double a_minus,
  const double w_min,
  const double w_max,
  const double learning_rate,
  const int size
) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  double dw = learning_rate * (
    a_plus * pre_trace[idx] * post_spike
    - a_minus * post_trace[idx] * pre_spikes[idx]
  );
  double w = weights[idx] + dw;
  if (w < w_min) {
    w = w_min;
  } else if (w > w_max) {
    w = w_max;
  }
  weights[idx] = w;
}

extern "C" __global__
void propagate_branch(
  const double axon_spike,
  const double axon_current,
  const double conduction_threshold,
  const double *safety_factors,
  double *branch_currents,
  const int size
) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  if (size <= 0) {
    return;
  }
  double distributed = axon_spike * axon_current / (double)size;
  double conduction_metric = distributed * safety_factors[idx];
  if (conduction_metric >= conduction_threshold) {
    branch_currents[idx] = distributed;
  } else {
    branch_currents[idx] = 0.0;
  }
}

extern "C" __global__
void bcm_ltp(
  double *theta,
  double *weights,
  const double *activity,
  const double tau_inv,
  const double eta,
  const double w_min,
  const double w_max,
  const double dt,
  const int size
) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  double theta_new = theta[idx] + dt * tau_inv * (
    activity[idx] * activity[idx] - theta[idx]
  );
  theta[idx] = theta_new;
  double dw = eta * activity[idx] * (activity[idx] - theta_new) * dt;
  double w = weights[idx] + dw;
  if (w < w_min) {
    w = w_min;
  } else if (w > w_max) {
    w = w_max;
  }
  weights[idx] = w;
}

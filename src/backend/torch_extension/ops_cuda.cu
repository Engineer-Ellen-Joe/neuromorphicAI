#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void update_neurons_kernel(
    const int N,
    const double t, const double dt,
    double* V_soma, double* V_ais, double* refrac_until,
    const double* I_syn_total,
    const double* soma_C_m, const double* soma_g_L, const double* soma_E_L, const double* soma_I_ext,
    const double* ais_C_m, const double* ais_g_L, const double* ais_E_L, const double* ais_V_T,
    const double* ais_Delta_T, const double* ais_V_spike, const double* ais_V_reset, const double* ais_t_refrac,
    const double* coup_g_c,
    int* spiked_now) {
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= N) return;
  double v_s = V_soma[nid];
  double v_a = V_ais[nid];
  bool in_refrac = (t < refrac_until[nid]);
  double i_syn = I_syn_total[nid];

  if (!in_refrac) {
    double dVs_1 = (soma_g_L[nid] * (soma_E_L[nid] - v_s) + coup_g_c[nid] * (v_a - v_s) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
    double dVa_1 = (ais_g_L[nid] * (ais_E_L[nid] - v_a) + coup_g_c[nid] * (v_s - v_a) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

    double v_s2 = v_s + 0.5 * dt * dVs_1;
    double v_a2 = v_a + 0.5 * dt * dVa_1;
    double dVs_2 = (soma_g_L[nid] * (soma_E_L[nid] - v_s2) + coup_g_c[nid] * (v_a2 - v_s2) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
    double dVa_2 = (ais_g_L[nid] * (ais_E_L[nid] - v_a2) + coup_g_c[nid] * (v_s2 - v_a2) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a2 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

    double v_s3 = v_s + 0.5 * dt * dVs_2;
    double v_a3 = v_a + 0.5 * dt * dVa_2;
    double dVs_3 = (soma_g_L[nid] * (soma_E_L[nid] - v_s3) + coup_g_c[nid] * (v_a3 - v_s3) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
    double dVa_3 = (ais_g_L[nid] * (ais_E_L[nid] - v_a3) + coup_g_c[nid] * (v_s3 - v_a3) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a3 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

    double v_s4 = v_s + dt * dVs_3;
    double v_a4 = v_a + dt * dVa_3;
    double dVs_4 = (soma_g_L[nid] * (soma_E_L[nid] - v_s4) + coup_g_c[nid] * (v_a4 - v_s4) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
    double dVa_4 = (ais_g_L[nid] * (ais_E_L[nid] - v_a4) + coup_g_c[nid] * (v_s4 - v_a4) + ais_g_L[nid] * ais_Delta_T[nid] * exp((v_a4 - ais_V_T[nid]) / ais_Delta_T[nid])) / ais_C_m[nid];

    V_soma[nid] = v_s + (dt / 6.0) * (dVs_1 + 2.0 * dVs_2 + 2.0 * dVs_3 + dVs_4);
    V_ais[nid] = v_a + (dt / 6.0) * (dVa_1 + 2.0 * dVa_2 + 2.0 * dVa_3 + dVa_4);
  } else {
    double v_a_fixed = ais_V_reset[nid];
    auto ks = [&](double vs_local) {
      return (soma_g_L[nid] * (soma_E_L[nid] - vs_local) + coup_g_c[nid] * (v_a_fixed - vs_local) + soma_I_ext[nid] + i_syn) / soma_C_m[nid];
    };
    double k1 = ks(v_s);
    double k2 = ks(v_s + 0.5 * dt * k1);
    double k3 = ks(v_s + 0.5 * dt * k2);
    double k4 = ks(v_s + dt * k3);
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

__global__ void update_synapses_kernel(
    const int N,
    const double dt,
    const int buffer_idx,
    const int buffer_depth,
    const int n_synapses,
    double* g_incr_buffer,
    double* syn_g, double* syn_w, double* syn_pre_trace, double* syn_post_trace,
    double* syn_r_pre, double* syn_r_post, double* syn_theta_m,
    const double* syn_g_max, const double* syn_E_rev, const double* syn_tau_decay,
    const double* syn_A_plus, const double* syn_A_minus, const double* syn_tau_pre, const double* syn_tau_post,
    const double* syn_eta_bcm, const double* syn_tau_rate, const double* syn_tau_theta,
    const double* syn_w_min, const double* syn_w_max, const double* syn_delay,
    const int* syn_pre_idx, const int* syn_post_idx,
    const int* spiked_now,
    double* I_syn_total,
    const double* V_soma) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= N) return;
  int pre_id = syn_pre_idx[sid];
  int post_id = syn_post_idx[sid];
  bool pre_spiked = (spiked_now[pre_id] == 1);
  bool post_spiked = (spiked_now[post_id] == 1);

  double g = syn_g[sid];
  const double w = syn_w[sid]; // treat weight as read-only (no in-place update)
  double pre_trace = syn_pre_trace[sid];
  double post_trace = syn_post_trace[sid];
  double r_pre = syn_r_pre[sid];
  double r_post = syn_r_post[sid];
  double theta_m = syn_theta_m[sid];

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

  // Plasticity update of w is disabled here; w is treated as external parameter for autograd.

  g *= exp(-dt / syn_tau_decay[sid]);
  int buffer_offset = buffer_idx * n_synapses + sid;
  double pending = g_incr_buffer[buffer_offset];
  g += pending;
  g_incr_buffer[buffer_offset] = 0.0;

  if (pre_spiked) {
    int delay_steps = (int)ceil(syn_delay[sid] / dt);
    if (delay_steps < 1) delay_steps = 1;
    int target = (buffer_idx + delay_steps) % buffer_depth;
    int target_offset = target * n_synapses + sid;
    g_incr_buffer[target_offset] = syn_g_max[sid] * w;
  }

  double I_syn = g * (syn_E_rev[sid] - V_soma[post_id]);
  atomicAdd(&I_syn_total[post_id], I_syn);

  syn_g[sid] = g;
  syn_pre_trace[sid] = pre_trace;
  syn_post_trace[sid] = post_trace;
  syn_r_pre[sid] = r_pre;
  syn_r_post[sid] = r_post;
  syn_theta_m[sid] = theta_m;
}

} // namespace

std::vector<torch::Tensor> snn_forward_cuda(
    torch::Tensor V_soma,
    torch::Tensor V_ais,
    torch::Tensor refrac_until,
    torch::Tensor I_syn_total,
    torch::Tensor spiked_now,
    torch::Tensor syn_g,
    torch::Tensor syn_w,
    torch::Tensor syn_pre_trace,
    torch::Tensor syn_post_trace,
    torch::Tensor syn_r_pre,
    torch::Tensor syn_r_post,
    torch::Tensor syn_theta_m,
    torch::Tensor g_incr_buffer,
    torch::Tensor soma_C_m, torch::Tensor soma_g_L, torch::Tensor soma_E_L, torch::Tensor soma_I_ext,
    torch::Tensor ais_C_m, torch::Tensor ais_g_L, torch::Tensor ais_E_L, torch::Tensor ais_V_T,
    torch::Tensor ais_Delta_T, torch::Tensor ais_V_spike, torch::Tensor ais_V_reset, torch::Tensor ais_t_refrac,
    torch::Tensor coup_g_c,
    torch::Tensor syn_g_max, torch::Tensor syn_E_rev, torch::Tensor syn_tau_decay,
    torch::Tensor syn_A_plus, torch::Tensor syn_A_minus, torch::Tensor syn_tau_pre, torch::Tensor syn_tau_post,
    torch::Tensor syn_eta_bcm, torch::Tensor syn_tau_rate, torch::Tensor syn_tau_theta,
    torch::Tensor syn_w_min, torch::Tensor syn_w_max, torch::Tensor syn_delay,
    torch::Tensor syn_pre_idx, torch::Tensor syn_post_idx,
    int64_t buffer_idx,
    double t, double dt) {

  const int threads = 256;
  const int n_neurons = V_soma.numel();
  const int n_synapses = syn_g.numel();
  const int neuron_blocks = (n_neurons + threads - 1) / threads;
  const int synapse_blocks = (n_synapses + threads - 1) / threads;

  update_neurons_kernel<<<neuron_blocks, threads>>>(
      n_neurons, t, dt,
      V_soma.data_ptr<double>(), V_ais.data_ptr<double>(), refrac_until.data_ptr<double>(),
      I_syn_total.data_ptr<double>(),
      soma_C_m.data_ptr<double>(), soma_g_L.data_ptr<double>(), soma_E_L.data_ptr<double>(), soma_I_ext.data_ptr<double>(),
      ais_C_m.data_ptr<double>(), ais_g_L.data_ptr<double>(), ais_E_L.data_ptr<double>(), ais_V_T.data_ptr<double>(),
      ais_Delta_T.data_ptr<double>(), ais_V_spike.data_ptr<double>(), ais_V_reset.data_ptr<double>(), ais_t_refrac.data_ptr<double>(),
      coup_g_c.data_ptr<double>(),
      spiked_now.data_ptr<int>());

  I_syn_total.zero_();

  const int buffer_depth = g_incr_buffer.size(0);

  update_synapses_kernel<<<synapse_blocks, threads>>>(
      n_synapses, dt, static_cast<int>(buffer_idx), buffer_depth, n_synapses,
      g_incr_buffer.data_ptr<double>(),
      syn_g.data_ptr<double>(), syn_w.data_ptr<double>(), syn_pre_trace.data_ptr<double>(), syn_post_trace.data_ptr<double>(),
      syn_r_pre.data_ptr<double>(), syn_r_post.data_ptr<double>(), syn_theta_m.data_ptr<double>(),
      syn_g_max.data_ptr<double>(), syn_E_rev.data_ptr<double>(), syn_tau_decay.data_ptr<double>(),
      syn_A_plus.data_ptr<double>(), syn_A_minus.data_ptr<double>(), syn_tau_pre.data_ptr<double>(), syn_tau_post.data_ptr<double>(),
      syn_eta_bcm.data_ptr<double>(), syn_tau_rate.data_ptr<double>(), syn_tau_theta.data_ptr<double>(),
      syn_w_min.data_ptr<double>(), syn_w_max.data_ptr<double>(), syn_delay.data_ptr<double>(),
      syn_pre_idx.data_ptr<int>(), syn_post_idx.data_ptr<int>(),
      spiked_now.data_ptr<int>(),
      I_syn_total.data_ptr<double>(),
      V_soma.data_ptr<double>());

  auto soft = torch::sigmoid((V_ais - ais_V_T) / 4.0);
  auto next_idx = torch::tensor({(buffer_idx + 1) % buffer_depth}, torch::TensorOptions().dtype(torch::kInt64).device(V_soma.device()));
  return {spiked_now, V_soma, V_ais, soft, next_idx,
          syn_pre_trace, syn_post_trace, syn_r_pre, syn_r_post, syn_theta_m};
}

std::vector<torch::Tensor> snn_backward_cuda(
    torch::Tensor grad_spikes,
    torch::Tensor saved_soft,
    torch::Tensor saved_spikes,
    torch::Tensor saved_v_soma,
    torch::Tensor saved_pre_trace,
    torch::Tensor saved_post_trace,
    torch::Tensor saved_r_pre,
    torch::Tensor saved_r_post,
    torch::Tensor saved_theta_m,
    torch::Tensor syn_pre_idx,
    torch::Tensor syn_post_idx,
    torch::Tensor syn_g_max,
    torch::Tensor syn_E_rev,
    torch::Tensor syn_A_plus,
    torch::Tensor syn_A_minus,
    torch::Tensor syn_eta_bcm,
    double dt,
    double surrogate_slope) {
  auto sigma_prime = saved_soft * (1 - saved_soft) / surrogate_slope;
  auto grad_external = grad_spikes * sigma_prime;

  // syn_pre/post are int32; index_select expects int64
  auto pre_idx = syn_pre_idx.to(torch::kInt64);
  auto post_idx = syn_post_idx.to(torch::kInt64);

  auto pre_spike = saved_spikes.index({pre_idx});
  auto post_grad = grad_spikes.index({post_idx});
  auto post_sigma_prime = sigma_prime.index({post_idx});
  auto v_post = saved_v_soma.index({post_idx});
  auto pre_trace = saved_pre_trace;
  auto post_trace = saved_post_trace;
  auto r_pre = saved_r_pre;
  auto r_post = saved_r_post;
  auto theta_m = saved_theta_m;
  auto post_spike = saved_spikes.index({post_idx});

  auto grad_w = post_grad * post_sigma_prime * pre_spike * syn_g_max * (syn_E_rev - v_post);
  // STDP/BCM pseudo-grad contribution
  auto dw_stdp = syn_A_plus * pre_spike * post_trace
               - syn_A_minus * post_spike
               + syn_eta_bcm * r_pre * r_post * (r_post - theta_m) * dt;

  return {grad_external, grad_w + dw_stdp};
}

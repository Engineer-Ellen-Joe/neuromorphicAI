#include <torch/extension.h>
#include <vector>

// Forward declarations (implemented in ops_cuda.cu)
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
    double t, double dt);

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
    double surrogate_slope);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> snn_forward(
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
  CHECK_INPUT(V_soma); CHECK_INPUT(V_ais); CHECK_INPUT(refrac_until);
  CHECK_INPUT(I_syn_total); CHECK_INPUT(spiked_now);
  CHECK_INPUT(syn_g); CHECK_INPUT(syn_w);
  CHECK_INPUT(syn_pre_trace); CHECK_INPUT(syn_post_trace);
  CHECK_INPUT(syn_r_pre); CHECK_INPUT(syn_r_post); CHECK_INPUT(syn_theta_m);
  CHECK_INPUT(g_incr_buffer);
  CHECK_INPUT(soma_C_m); CHECK_INPUT(soma_g_L); CHECK_INPUT(soma_E_L); CHECK_INPUT(soma_I_ext);
  CHECK_INPUT(ais_C_m); CHECK_INPUT(ais_g_L); CHECK_INPUT(ais_E_L); CHECK_INPUT(ais_V_T);
  CHECK_INPUT(ais_Delta_T); CHECK_INPUT(ais_V_spike); CHECK_INPUT(ais_V_reset); CHECK_INPUT(ais_t_refrac);
  CHECK_INPUT(coup_g_c);
  CHECK_INPUT(syn_g_max); CHECK_INPUT(syn_E_rev); CHECK_INPUT(syn_tau_decay);
  CHECK_INPUT(syn_A_plus); CHECK_INPUT(syn_A_minus); CHECK_INPUT(syn_tau_pre); CHECK_INPUT(syn_tau_post);
  CHECK_INPUT(syn_eta_bcm); CHECK_INPUT(syn_tau_rate); CHECK_INPUT(syn_tau_theta);
  CHECK_INPUT(syn_w_min); CHECK_INPUT(syn_w_max); CHECK_INPUT(syn_delay);
  CHECK_INPUT(syn_pre_idx); CHECK_INPUT(syn_post_idx);

  return snn_forward_cuda(
      V_soma, V_ais, refrac_until, I_syn_total, spiked_now,
      syn_g, syn_w, syn_pre_trace, syn_post_trace, syn_r_pre, syn_r_post, syn_theta_m,
      g_incr_buffer,
      soma_C_m, soma_g_L, soma_E_L, soma_I_ext,
      ais_C_m, ais_g_L, ais_E_L, ais_V_T, ais_Delta_T, ais_V_spike, ais_V_reset, ais_t_refrac,
      coup_g_c,
      syn_g_max, syn_E_rev, syn_tau_decay,
      syn_A_plus, syn_A_minus, syn_tau_pre, syn_tau_post,
      syn_eta_bcm, syn_tau_rate, syn_tau_theta,
      syn_w_min, syn_w_max, syn_delay,
      syn_pre_idx, syn_post_idx,
      buffer_idx, t, dt);
}

std::vector<torch::Tensor> snn_backward(
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
  CHECK_INPUT(grad_spikes);
  CHECK_INPUT(saved_soft);
  CHECK_INPUT(saved_spikes);
  CHECK_INPUT(saved_v_soma);
  CHECK_INPUT(saved_pre_trace);
  CHECK_INPUT(saved_post_trace);
  CHECK_INPUT(saved_r_pre);
  CHECK_INPUT(saved_r_post);
  CHECK_INPUT(saved_theta_m);
  CHECK_INPUT(syn_pre_idx);
  CHECK_INPUT(syn_post_idx);
  CHECK_INPUT(syn_g_max);
  CHECK_INPUT(syn_E_rev);
  CHECK_INPUT(syn_A_plus);
  CHECK_INPUT(syn_A_minus);
  CHECK_INPUT(syn_eta_bcm);
  return snn_backward_cuda(
      grad_spikes,
      saved_soft,
      saved_spikes,
      saved_v_soma,
      saved_pre_trace,
      saved_post_trace,
      saved_r_pre,
      saved_r_post,
      saved_theta_m,
      syn_pre_idx,
      syn_post_idx,
      syn_g_max,
      syn_E_rev,
      syn_A_plus,
      syn_A_minus,
      syn_eta_bcm,
      dt,
      surrogate_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &snn_forward, "SNN forward (CUDA)");
  m.def("backward", &snn_backward, "SNN backward surrogate (CUDA)");
}

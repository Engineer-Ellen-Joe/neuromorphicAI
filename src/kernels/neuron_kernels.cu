/* neuron_kernels.cu
****************************************************************************************************
Hodgkin-Huxley 모델의 게이트 동역학을 계산하는 CUDA C++ 코드
  - 각 alpha/beta 함수는 특정 이온 채널 게이트의 열림/닫힘 속도를 계산
  - update_gates: 게이트 변수(m,h,n)를 다음 시간 스텝으로 업데이트
  - update_voltage: 뉴런 전압을 Euler 적분으로 업데이트하고 스파이크 이벤트를 기록
  - compute_axial_currents: 연결된 뉴런 간 축성 전류 계산
  - compute_synaptic_currents: 시냅스 전류 계산
****************************************************************************************************
*/

// K+ 채널 활성화 게이트 'n'의 열림 비율(alpha_n)
__device__ double alpha_n(double V) {
  if (abs(V - (-55.0)) < 1e-5) { return 0.1; }
  return 0.01 * (V + 55.0) / (1.0 - exp(-0.1 * (V + 55.0)));
}

// K+ 채널 활성화 게이트 'n'의 닫힘 비율(beta_n)
__device__ double beta_n(double V) { return 0.125 * exp(-0.0125 * (V + 65.0)); }

// Na+ 채널 활성화 게이트 'm'의 열림 비율(alpha_m)
__device__ double alpha_m(double V) {
  if (abs(V - (-40.0)) < 1e-5) { return 1.0; }
  return 0.1 * (V + 40.0) / (1.0 - exp(-0.1 * (V + 40.0)));
}
__device__ double beta_m(double V) { return 4.0 * exp(-(V + 65.0) / 18.0); }        // Na+ 채널 활성화 게이트 'm'의 닫힘 비율(beta_m)
__device__ double alpha_h(double V) { return 0.07 * exp(-0.05 * (V + 65.0)); }      // Na+ 채널 비활성화 게이트 'h'의 열림 비율(alpha_h)
__device__ double beta_h(double V) { return 1.0 / (1.0 + exp(-0.1 * (V + 35.0))); } // Na+ 채널 비활성화 게이트 'h'의 닫힘 비율(beta_h)

extern "C" __global__
void update_gates(
  const int N, const double dt, const double* __restrict__ V,
  double* __restrict__ m, double* __restrict__ h, double* __restrict__ n
) {
  // 각 스레드가 처리할 뉴런의 인덱스를 계산
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;

  double v_i = V[i];

  // 현재 전압에서 각 게이트의 alpha, beta 값을 계산
  double an = alpha_n(v_i); double bn = beta_n(v_i);
  double am = alpha_m(v_i); double bm = beta_m(v_i);
  double ah = alpha_h(v_i); double bh = beta_h(v_i);

  // 각 게이트의 시상수(tau)와 정상 상태 값(inf)을 계산
  double tau_n = 1.0 / (an + bn); double n_inf = an * tau_n;
  double tau_m = 1.0 / (am + bm); double m_inf = am * tau_m;
  double tau_h = 1.0 / (ah + bh); double h_inf = ah * tau_h;

  /* 분석적 해법을 사용하여 게이트 변수 업데이트
     n(t+dt) = n_inf + (n(t) - n_inf) * exp(-dt / tau_n)
     => 단순 지수적 접근법으로 안정 상태로 수렴하도록 계산 */
    n[i] = n_inf + (n[i] - n_inf) * exp(-dt / tau_n);
    m[i] = m_inf + (m[i] - m_inf) * exp(-dt / tau_m);
    h[i] = h_inf + (h[i] - h_inf) * exp(-dt / tau_h);
  }

extern "C" __global__
void update_voltage(
  const int N, const double dt, 
  const double* __restrict__ C_m, const double* __restrict__ g_leak, const double* __restrict__ E_leak,
  const double* __restrict__ g_k, const double* __restrict__ E_k,
  const double* __restrict__ g_na, const double* __restrict__ E_na,
  const double spike_threshold, const double /*V_reset - leave for API compatibility*/,
  const double* __restrict__ m, const double* __restrict__ h, const double* __restrict__ n,
  const double* __restrict__ I_ext, const double* __restrict__ I_axial, const double* __restrict__ I_syn,
  double* __restrict__ V,
  unsigned char* __restrict__ spike,
  const int* __restrict__ compartment_type, const int ais_type_id
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;

  // 이전 전압(함수 입력으로 받은 V)은 업데이트 전 전압
  double v_prev = V[i];

  // 채널 상태는 커널 입력으로 들어온 m,h,n (이미 업데이트된 값 또는 같은 스텝에 업데이트됨)
  double m_i = m[i];
  double h_i = h[i];
  double n_i = n[i];

  // 이온 전류 계산 (Hodgkin-Huxley)
  double I_leak = g_leak[i] * (v_prev - E_leak[i]);
  double I_k = g_k[i] * n_i * n_i * n_i * n_i * (v_prev - E_k[i]);
  double I_na = g_na[i] * m_i * m_i * m_i * h_i * (v_prev - E_na[i]);

  // 외부/축성/시냅스 전류 (입력 배열들은 device memory에서 옴)
  double I_ext_i = (I_ext != NULL) ? I_ext[i] : 0.0;
  double I_axial_i = (I_axial != NULL) ? I_axial[i] : 0.0;
  double I_syn_i = (I_syn != NULL) ? I_syn[i] : 0.0;

  // 총 전류 합
  double I_ion = I_leak + I_k + I_na;
  double I_total = I_ext_i + I_axial_i + I_syn_i - I_ion;

  // 전압 업데이트/적분 (explicit Euler 형태)
  double v_new = v_prev + (dt / C_m[i]) * I_total;

  // 스파이크 이벤트 기록: AIS 구간에서 threshold crossing 시만 spike = 1
  if (compartment_type[i] == ais_type_id) {
    // previous below threshold and new above => crossing -> spike event
    if ((v_prev < spike_threshold) && (v_new >= spike_threshold)) {
      if (spike != NULL) { spike[i] = 1; }
    } else {
      if (spike != NULL) { spike[i] = 0; }
    }
  } else {
    if (spike != NULL) { spike[i] = 0; }
  }

  // 업데이트된 전압 저장 (자연스럽게 재분극/과분극이 일어나도록 함)
  V[i] = v_new;
}

extern "C" __global__
void compute_axial_currents(
  const int n_connections,
  const int* __restrict__ connection_pairs,
  const double* __restrict__ connection_g_axial,
  const double* __restrict__ V,
  double* __restrict__ I_axial_out
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_connections) return;

  int from_idx = connection_pairs[i * 2];
  int to_idx = connection_pairs[i * 2 + 1];

  double v_from = V[from_idx];
  double v_to = V[to_idx];
  double g = connection_g_axial[i];

  double current = g * (v_from - v_to);

  // 양방향 축성 전류 계산 (from -> to)
  atomicAdd(&I_axial_out[from_idx], -current);
  atomicAdd(&I_axial_out[to_idx], current);
}

extern "C" __global__
void compute_synaptic_currents(
  const int n_synapses, const double dt,
  const int* __restrict__ post_comp_map,
  const double* __restrict__ gmax,
  const double* __restrict__ tau_decay,
  const double* __restrict__ E_syn,
  const double* __restrict__ weight,
  double* __restrict__ conductance,
  const double* __restrict__ V,
  double* __restrict__ I_syn_out
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_synapses) return;

  // 지수적 감쇠로 시냅스 컨덕턴스 업데이트
  conductance[i] *= exp(-dt / tau_decay[i]);

  int post_idx = post_comp_map[i];
  double v_post = V[post_idx];
  double current = conductance[i] * weight[i] * (v_post - E_syn[i]); // 시냅스 전류: I = g_syn * w * (V_post - E_syn)

  // 뉴런에 전류 더함 (postsynaptic neuron)
  atomicAdd(&I_syn_out[post_idx], -current);
}
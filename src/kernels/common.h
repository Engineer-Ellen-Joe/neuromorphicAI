#pragma once

// Simulation constants 구조체
struct SimConstants {
  float dt;
  float g_na;
  float g_k;
  float g_leak;
  float E_na;
  float E_k;
  float E_leak;
  float Cm;
};

// 전역 constant memory에 올림
__constant__ SimConstants d_consts;

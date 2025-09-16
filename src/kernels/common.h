#pragma once

// Compartment types
enum CompartmentType {
    COMP_SOMA = 0,
    COMP_DENDRITE = 1,
    COMP_AXON = 2,
    COMP_AIS = 3
};

// Synapse types
enum SynapseType {
    SYN_EXC = 0,
    SYN_INH = 1
};

// Simulation constants (example)
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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import math
import numpy as np
import heapq
import enum
import random


# ----------------------------- Utilities & Constants -----------------------------

def exp_clip(x: float) -> float:
    """Numerically stable exp for rate equations (clip extreme voltages)."""
    x = max(min(x, 50.0), -50.0)
    return math.exp(x)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# Physiological defaults (mV, ms, uF/cm^2, mS/cm^2)
E_Na = 50.0     # Sodium reversal (mV)
E_K  = -77.0    # Potassium reversal (mV)
E_L  = -54.4    # Leak reversal (mV) classical HH
E_Cl = -70.0    # GABA_A (approx) (mV)
E_AMPA = 0.0
E_NMDA = 0.0

DEFAULT_TEMP_C = 36.0  # Temperature for Q10 scaling if extended


# ----------------------------- Event Scheduler -----------------------------

class EventQueue:
    """Min-heap based event queue for scheduled (time, callable) tasks."""
    def __init__(self):
        self._heap: List[Tuple[float, int, Callable[[], None]]] = []
        self._counter = 0

    def schedule(self, t: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._heap, (t, self._counter, fn))
        self._counter += 1

    def pop_ready(self, t: float) -> List[Callable[[], None]]:
        ready = []
        while self._heap and self._heap[0][0] <= t:
            _, _, fn = heapq.heappop(self._heap)
            ready.append(fn)
        return ready


# ----------------------------- Ion Channel Mechanisms -----------------------------

class Mechanism:
    """Base class for ionic mechanisms attached to a compartment."""
    def step(self, V: float, dt: float) -> float:
        """
        Return ionic current (positive outward) contributed by this mechanism
        and internally update gating variables using V and dt.
        """
        raise NotImplementedError()


@dataclass
class HHNaKParams:
    gNa: float = 120.0  # mS/cm^2
    gK: float  = 36.0   # mS/cm^2
    gLeak: float = 0.3  # mS/cm^2
    ENa: float = E_Na
    EK: float  = E_K
    ELeak: float = E_L


class HodgkinHuxleyNaK(Mechanism):
    """
    Classic Hodgkin-Huxley (1952) Na/K/Leak mechanism.
    V in mV, dt in ms, currents in uA/cm^2.
    """
    def __init__(self, params: HHNaKParams):
        self.p = params
        # Initialize gating variables to steady-state at -65 mV
        V0 = -65.0
        self.m = self._alpha_m(V0) / (self._alpha_m(V0) + self._beta_m(V0))
        self.h = self._alpha_h(V0) / (self._alpha_h(V0) + self._beta_h(V0))
        self.n = self._alpha_n(V0) / (self._alpha_n(V0) + self._beta_n(V0))

    # Rate functions (mV)
    def _alpha_m(self, V): return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))
    def _beta_m(self, V):  return 4.0 * math.exp(-(V + 65.0) / 18.0)
    def _alpha_h(self, V): return 0.07 * math.exp(-(V + 65.0) / 20.0)
    def _beta_h(self, V):  return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))
    def _alpha_n(self, V): return 0.01 * (V + 55.0) / (1.0 - math.exp(-(V + 55.0) / 10.0))
    def _beta_n(self, V):  return 0.125 * math.exp(-(V + 65.0) / 80.0)

    def step(self, V: float, dt: float) -> float:
        am = self._alpha_m(V); bm = self._beta_m(V)
        ah = self._alpha_h(V); bh = self._beta_h(V)
        an = self._alpha_n(V); bn = self._beta_n(V)
        dm = dt * (am * (1.0 - self.m) - bm * self.m)
        dh = dt * (ah * (1.0 - self.h) - bh * self.h)
        dn = dt * (an * (1.0 - self.n) - bn * self.n)
        self.m += dm; self.h += dh; self.n += dn
        # Currents (uA/cm^2)
        INa = self.p.gNa * (self.m**3) * self.h * (V - self.p.ENa)
        IK  = self.p.gK  * (self.n**4) * (V - self.p.EK)
        IL  = self.p.gLeak * (V - self.p.ELeak)
        return INa + IK + IL


# ----------------------------- Synapses & Plasticity -----------------------------

class SynapseKind(enum.Enum):
    AMPA_NMDA = 1
    GABA_A = 2


@dataclass
class STDPParams:
    tau_pre: float = 20.0   # ms
    tau_post: float = 20.0  # ms
    A_plus: float = 0.01    # LTP amplitude
    A_minus: float = 0.012  # LTD amplitude
    eta: float = 1.0        # global learning rate scaling


@dataclass
class LTPParams:
    tau_Ca: float = 50.0     # ms, Ca decay
    k_Ca: float = 0.005      # Ca influx scaling per NMDA current
    theta_pot: float = 1.2   # Ca threshold (arbitrary units)
    eta_pot: float = 0.0005  # potentiation rate
    # Optional weak LTD (when Ca moderately elevated)
    theta_dep_low: float = 0.3
    theta_dep_high: float = 0.8
    eta_dep: float = 0.0002


@dataclass
class Synapse:
    """Biophysical synapse attached to a specific postsynaptic compartment."""
    pre: 'SpikingSource'
    post_compartment: 'Compartment'
    kind: SynapseKind
    weight: float
    gmax: float = 0.001    # mS (peak conductance scaling)
    tau_rise: float = 0.5  # ms
    tau_decay: float = 3.0 # ms (AMPA) / 8-100 for NMDA; we'll separate internally
    nmda_fraction: float = 0.25  # fraction of g that is NMDA for excitatory
    mg_conc: float = 1.0   # mM Mg2+ for NMDA block
    stdp: STDPParams = field(default_factory=STDPParams)
    ltp: LTPParams = field(default_factory=LTPParams)
    w_min: float = 0.05
    w_max: float = 2.5

    # Internal state
    r_fast: float = 0.0
    r_slow: float = 0.0
    x_pre: float = 0.0
    x_post: float = 0.0
    Ca: float = 0.0
    last_pre_spike: Optional[float] = None
    last_post_spike: Optional[float] = None

    def __post_init__(self):
        # Register with presynaptic source
        self.pre._register_outgoing(self)

    def schedule_pre_spike(self, t: float):
        """Called by presynaptic axonal terminal to signal an arriving spike at time t."""
        self.post_compartment.net.events.schedule(t, lambda: self._on_pre_spike(t))

    def _on_pre_spike(self, t: float):
        # Conductance transient (bi-exponential via two states)
        # For AMPA/NMDA we split: fast (AMPA-like) and slow (NMDA-like)
        # Instantaneous increment ensuring peak-normalized amplitude
        self.r_fast += 1.0
        self.r_slow += 1.0
        # STDP: nearest-neighbor (pre event uses current post-trace)
        if self.stdp is not None:
            dw = self.stdp.eta * self.stdp.A_plus * self.x_post
            self.weight = float(np.clip(self.weight + dw, self.w_min, self.w_max))
            self.x_pre += 1.0
        self.last_pre_spike = t

    def on_post_spike(self, t: float):
        # STDP: post event uses current pre-trace
        if self.stdp is not None:
            dw = - self.stdp.eta * self.stdp.A_minus * self.x_pre
            self.weight = float(np.clip(self.weight + dw, self.w_min, self.w_max))
            self.x_post += 1.0
        self.last_post_spike = t

    def step(self, V_post: float, dt: float) -> float:
        """Advance synapse state and return synaptic current (positive outward from post)."""
        # Update traces for STDP
        if self.stdp is not None:
            self.x_pre  += -dt * self.x_pre  / self.stdp.tau_pre
            self.x_post += -dt * self.x_post / self.stdp.tau_post

        # Update conductance states (bi-exponential shape via two first-order filters)
        # Fast: AMPA-like; Slow: NMDA-like (longer decay)
        tau_r = max(1e-6, self.tau_rise)
        tau_d_fast = max(self.tau_decay, self.tau_rise + 0.1)
        tau_d_slow = max(40.0, self.tau_decay * 12.0)  # NMDA slower

        # Exact update for first-order: x(t+dt) = x*exp(-dt/tau)
        self.r_fast *= math.exp(-dt / tau_d_fast)
        self.r_slow *= math.exp(-dt / tau_d_slow)

        # Compute conductances
        g_total = self.gmax * self.weight
        if self.kind == SynapseKind.AMPA_NMDA:
            g_ampa = g_total * (1.0 - self.nmda_fraction) * (1.0 - math.exp(-dt / tau_r)) * self.r_fast
            # NMDA with Mg2+ block
            g_nmda_base = g_total * self.nmda_fraction * (1.0 - math.exp(-dt / tau_r)) * self.r_slow
            B = 1.0 / (1.0 + self.mg_conc * math.exp(-0.062 * V_post) / 3.57)
            g_nmda = g_nmda_base * B
            I_ampa = g_ampa * (V_post - E_AMPA)
            I_nmda = g_nmda * (V_post - E_NMDA)
            I_syn = I_ampa + I_nmda
            # Ca dynamics driven by NMDA current (inward negative): use magnitude
            self.Ca += dt * (-self.ltp.k_Ca * I_nmda) - dt * (self.Ca / self.ltp.tau_Ca)

            # Ca-based LTP/LTD
            if self.Ca > self.ltp.theta_pot:
                dw = self.ltp.eta_pot * (1.0 - (self.weight / self.w_max)) * (self.Ca - self.ltp.theta_pot) / (self.Ca + self.ltp.theta_pot)
                self.weight = float(np.clip(self.weight + dw * dt, self.w_min, self.w_max))
            elif self.ltp.theta_dep_low < self.Ca < self.ltp.theta_dep_high:
                dw = - self.ltp.eta_dep * (self.weight / self.w_max) * (self.ltp.theta_dep_high - self.Ca) / self.ltp.theta_dep_high
                self.weight = float(np.clip(self.weight + dw * dt, self.w_min, self.w_max))

            return I_syn
        elif self.kind == SynapseKind.GABA_A:
            g_gaba = g_total * (1.0 - math.exp(-dt / tau_r)) * self.r_fast
            I_gaba = g_gaba * (V_post - E_Cl)
            return I_gaba
        else:
            return 0.0


# ----------------------------- Compartments -----------------------------

@dataclass
class Compartment:
    name: str
    net: 'Network'
    area_cm2: float = 1e-6  # typical small area; used for scaling currents to volts
    Cm_uF_per_cm2: float = 1.0
    Gleak_mS_per_cm2: float = 0.1
    Eleak_mV: float = -65.0
    mechanisms: List[Mechanism] = field(default_factory=list)
    synapses: List[Synapse] = field(default_factory=list)
    neighbors: List[Tuple['Compartment', float]] = field(default_factory=list)  # (neighbor, g_axial mS)
    I_ext_uA: float = 0.0

    V_mV: float = -65.0

    def connect_axial(self, other: 'Compartment', g_axial_mS: float):
        self.neighbors.append((other, g_axial_mS))

    def add_mechanism(self, mech: Mechanism):
        self.mechanisms.append(mech)

    def add_synapse(self, syn: Synapse):
        self.synapses.append(syn)

    def step_current(self, dt: float) -> float:
        """Compute net outward current density (uA/cm^2) at this compartment for this timestep."""
        # Leak
        I_leak = self.Gleak_mS_per_cm2 * (self.V_mV - self.Eleak_mV)

        # Ionic mechanisms (e.g., HH)
        I_ion = 0.0
        for mech in self.mechanisms:
            I_ion += mech.step(self.V_mV, dt)

        # Axial coupling (Ohmic): sum g*(V_i - V_j) as outward current from i
        I_axial = 0.0
        for nbr, g in self.neighbors:
            I_axial += g * (self.V_mV - nbr.V_mV)

        # Synaptic currents (already sign-correct as outward)
        I_syn = 0.0
        for syn in self.synapses:
            I_syn += syn.step(self.V_mV, dt)

        # External current (uA); convert to density by dividing area
        I_ext_density = self.I_ext_uA / max(1e-12, self.area_cm2)

        # Net outward density
        I_out = I_leak + I_ion + I_axial + I_syn - I_ext_density
        return I_out

    def integrate_voltage(self, I_out_density: float, dt: float):
        """Forward Euler update V using dV/dt = - I_out / C (signs: outward current hyperpolarizes)."""
        Cm = self.Cm_uF_per_cm2
        dV = - (I_out_density / max(1e-12, Cm)) * dt
        self.V_mV += dV


# ----------------------------- Spiking Sources & External Generators -----------------------------

class SpikingSource:
    """Abstract base for any object that can emit spikes and has outgoing synapses."""
    def __init__(self, name: str, net: 'Network'):
        self.name = name
        self.net = net
        self._outgoing: List[Synapse] = []

    def _register_outgoing(self, syn: Synapse):
        self._outgoing.append(syn)

    def fire(self, t: float, branches: Optional[List['AxonBranch']] = None):
        """
        Emit a spike at time t. If branches provided, use branch routing to schedule arrivals.
        Otherwise deliver immediately (for non-axonal sources).
        """
        if branches is None:
            for syn in self._outgoing:
                syn.schedule_pre_spike(t)
        else:
            # Use the provided branches to route spikes
            for br in branches:
                br.route_and_deliver(spike_time=t)


class ExternalSpikeGenerator(SpikingSource):
    """Poisson spike generator as an external 'neuron' (A, B, C, etc.)."""
    def __init__(self, name: str, net: 'Network', rate_hz: float, weight_scale: float = 1.0):
        super().__init__(name, net)
        self.rate_hz = rate_hz
        self.weight_scale = weight_scale

    def connect_to(self, neuron: 'PyramidalNeuron', target_region: str, segment_index: int,
                   kind: SynapseKind, weight: float):
        neuron.attach_synapse_from(self, target_region, segment_index, kind, weight * self.weight_scale)

    def step(self, t: float, dt: float):
        # Thinning method for Poisson with small dt
        p = self.rate_hz * dt / 1000.0
        if random.random() < p:
            self.fire(t)


# ----------------------------- Axon Branching & Routing -----------------------------

@dataclass
class AxonBranch:
    """Models deterministic spike routing along a branch using safety factor (SF)."""
    name: str
    parent_neuron: 'PyramidalNeuron'
    length_um: float
    diameter_um: float
    myelin_factor: float  # 0 (unmyelinated) ... 1 (heavily myelinated)
    channel_density: float  # relative Na channel density (1.0 baseline)
    terminals: List[Synapse] = field(default_factory=list)

    # Routing parameters
    v_base_m_per_s: float = 0.5    # baseline conduction velocity for reference geometry
    delay_noise_sd_ms: float = 0.05
    sf_threshold: float = 1.0
    freq_sensitivity: float = 0.8  # higher → more failure at high freq
    last_arrival_time: Optional[float] = None

    def _conduction_velocity(self) -> float:
        # Empirical: velocity ~ sqrt(diameter) * (1 + myelin*4) m/s (scaled)
        v = self.v_base_m_per_s * math.sqrt(max(1e-6, self.diameter_um)) * (1.0 + 4.0 * self.myelin_factor)
        return max(0.1, v)

    def _prop_delay_ms(self) -> float:
        L_m = self.length_um * 1e-6
        v = self._conduction_velocity()
        delay = 1000.0 * (L_m / v)  # ms
        if self.delay_noise_sd_ms > 0.0:
            delay += random.gauss(0.0, self.delay_noise_sd_ms)
        return max(0.0, delay)

    def _safety_factor(self, spike_amplitude_mV: float, isi_ms: float) -> float:
        # SF grows with diameter, myelin, channel density, and spike amplitude;
        # drops with firing frequency (1/ISI)
        freq_hz = 1000.0 / max(isi_ms, 1e-3)
        geom = math.sqrt(max(1e-6, self.diameter_um)) * (1.0 + 2.0 * self.myelin_factor)
        excit = self.channel_density * (1.0 + 0.01 * max(0.0, spike_amplitude_mV))
        freq_penalty = 1.0 + self.freq_sensitivity * (freq_hz / 100.0)
        sf = (geom * excit) / freq_penalty
        return sf

    def route_and_deliver(self, spike_time: float):
        # Determine ISI using parent AIS spike history
        last = self.parent_neuron.last_ais_spike_time
        isi = float('inf') if last is None else (spike_time - last)
        Vamp = self.parent_neuron.ais_spike_amplitude_mV
        sf = self._safety_factor(Vamp, isi)
        if sf >= self.sf_threshold:
            delay = self._prop_delay_ms()
            t_arrival = spike_time + delay
            for syn in self.terminals:
                syn.schedule_pre_spike(t_arrival)
            self.last_arrival_time = t_arrival
        else:
            # conduction failure — no delivery
            pass


# ----------------------------- Pyramidal Neuron -----------------------------

class PyramidalNeuron(SpikingSource):
    """
    Multi-compartment pyramidal neuron with basal/apical/tuft dendrites, soma, AIS, and axon branching.
    Soma/AIS carry HH Na/K mechanisms; dendrites are passive by default but accept synapses and axial coupling.
    """
    def __init__(self, name: str, net: 'Network',
                 n_basal: int = 10, n_apical: int = 10, n_tuft: int = 6):
        super().__init__(name, net)
        self.traces: Dict[str, List[float]] = {"soma_V": [], "ais_V": []}
        self.spike_times: List[float] = []
        self.last_ais_spike_time: Optional[float] = None
        self.ais_spike_amplitude_mV: float = 100.0  # used for routing SF

        # Build compartments
        self.basal: List[Compartment] = [Compartment(f"{name}_basal_{i}", net) for i in range(n_basal)]
        self.apical: List[Compartment] = [Compartment(f"{name}_apical_{i}", net) for i in range(n_apical)]
        self.tuft:   List[Compartment] = [Compartment(f"{name}_tuft_{i}",   net) for i in range(n_tuft)]
        self.soma = Compartment(f"{name}_soma", net)
        self.ais  = Compartment(f"{name}_ais",  net)

        # Mechanisms
        soma_mech = HodgkinHuxleyNaK(HHNaKParams(gNa=120.0, gK=36.0, gLeak=0.3))
        ais_mech  = HodgkinHuxleyNaK(HHNaKParams(gNa=200.0, gK=40.0, gLeak=0.2))  # higher Na density in AIS
        self.soma.add_mechanism(soma_mech)
        self.ais.add_mechanism(ais_mech)

        # Passive dendrites
        for comp in self.basal + self.apical + self.tuft:
            comp.Cm_uF_per_cm2 = 1.0
            comp.Gleak_mS_per_cm2 = 0.12
            comp.Eleak_mV = -65.0

        # Axial connections: join dendrites to soma; apical trunk chain to tuft
        # Simple chain within each group and soma hub
        def connect_chain(lst: List[Compartment], g_axial):
            for i in range(len(lst) - 1):
                lst[i].connect_axial(lst[i+1], g_axial)
                lst[i+1].connect_axial(lst[i], g_axial)
        connect_chain(self.basal, g_axial=0.3)
        connect_chain(self.apical, g_axial=0.3)
        connect_chain(self.tuft,   g_axial=0.25)

        # connect proximal dendrites to soma
        if self.basal:
            self.basal[0].connect_axial(self.soma, 0.6)
            self.soma.connect_axial(self.basal[0], 0.6)
        if self.apical:
            self.apical[0].connect_axial(self.soma, 0.6)
            self.soma.connect_axial(self.apical[0], 0.6)
        if self.tuft and self.apical:
            # apical last connects to tuft first
            self.apical[-1].connect_axial(self.tuft[0], 0.4)
            self.tuft[0].connect_axial(self.apical[-1], 0.4)

        # soma ↔ AIS coupling
        self.soma.connect_axial(self.ais, 0.8)
        self.ais.connect_axial(self.soma, 0.8)

        # AIS spike detection params
        self.ais_threshold_mV = -50.0
        self.ais_reset_mV = -60.0
        self.ais_refrac_ms = 2.0
        self._ais_last_spike: Optional[float] = None

        # Axon branches (main vs collateral) with different geometry/myelination
        self.main_axon = AxonBranch(
            name=f"{name}_main",
            parent_neuron=self,
            length_um=800.0, diameter_um=1.2, myelin_factor=0.9, channel_density=1.2
        )
        self.collateral_axon = AxonBranch(
            name=f"{name}_collateral",
            parent_neuron=self,
            length_um=300.0, diameter_um=0.6, myelin_factor=0.2, channel_density=0.9
        )

        # Register with network
        net.register_neuron(self)

    # ----------------- Connectivity -----------------

    def attach_synapse_from(self, pre: SpikingSource, target_region: str, segment_index: int,
                            kind: SynapseKind, weight: float) -> Synapse:
        comp = self._resolve_region_segment(target_region, segment_index)
        syn = Synapse(pre=pre, post_compartment=comp, kind=kind, weight=weight)
        comp.add_synapse(syn)
        return syn

    def connect_axon_to(self, target: 'PyramidalNeuron', branch: str, target_region: str,
                        segment_index: int, kind: SynapseKind, weight: float) -> Synapse:
        comp = target._resolve_region_segment(target_region, segment_index)
        syn = Synapse(pre=self, post_compartment=comp, kind=kind, weight=weight)
        comp.add_synapse(syn)
        if branch == "main":
            self.main_axon.terminals.append(syn)
        elif branch == "collateral":
            self.collateral_axon.terminals.append(syn)
        else:
            raise ValueError("branch must be 'main' or 'collateral'")
        return syn

    def _resolve_region_segment(self, region: str, idx: int) -> Compartment:
        region = region.lower()
        if region == "soma":
            return self.soma
        if region == "ais":
            return self.ais
        arr = {"basal": self.basal, "apical": self.apical, "tuft": self.tuft}.get(region)
        if arr is None:
            raise ValueError(f"Unknown region {region}")
        if not (0 <= idx < len(arr)):
            raise IndexError(f"segment_index {idx} out of bounds for region {region}")
        return arr[idx]

    # ----------------- Simulation Support -----------------

    def step(self, t: float, dt: float):
        # 1) Collect outward current densities for all compartments based on current V
        comps = self.basal + self.apical + self.tuft + [self.soma, self.ais]
        Iouts = [c.step_current(dt) for c in comps]

        # 2) Integrate all voltages synchronously
        for c, Iout in zip(comps, Iouts):
            c.integrate_voltage(Iout, dt)

        # 3) AIS spike detection and routing
        self.traces["soma_V"].append(self.soma.V_mV)
        self.traces["ais_V"].append(self.ais.V_mV)

        # Refractory handling
        can_spike = True
        if self._ais_last_spike is not None and (t - self._ais_last_spike) < self.ais_refrac_ms:
            can_spike = False

        # Threshold crossing from below
        if can_spike and self.ais.V_mV >= self.ais_threshold_mV and self.traces["ais_V"][-2] < self.ais_threshold_mV:
            # Spike event at AIS
            self._ais_last_spike = t
            self.last_ais_spike_time = t
            self.spike_times.append(t)

            # Notify all synapses targeting this neuron (post-spike component of STDP)
            for comp in comps:
                for syn in comp.synapses:
                    syn.on_post_spike(t)

            # Route spike along axon branches
            self.fire(t, branches=[self.main_axon, self.collateral_axon])

    def get_trace(self, key: str) -> np.ndarray:
        return np.array(self.traces.get(key, []), dtype=float)


# ----------------------------- Network Simulator -----------------------------

class Network:
    """Manages time stepping, event scheduling, and neuron/external source registry."""
    def __init__(self, dt: float = 0.025, seed: Optional[int] = None):
        self.dt = dt
        self.t = 0.0
        self.events = EventQueue()
        self.neurons: List[PyramidalNeuron] = []
        self.generators: List[ExternalSpikeGenerator] = []
        if seed is not None:
            np.random.seed(seed); random.seed(seed)

    def register_neuron(self, n: PyramidalNeuron):
        self.neurons.append(n)

    def register_generator(self, g: ExternalSpikeGenerator):
        self.generators.append(g)

    def add(self, obj):
        if isinstance(obj, PyramidalNeuron):
            self.register_neuron(obj)
        elif isinstance(obj, ExternalSpikeGenerator):
            self.register_generator(obj)
        else:
            raise ValueError("Unsupported object type")

    def run(self, T_ms: float):
        n_steps = int(math.ceil(T_ms / self.dt))
        # Pre-allocate previous AIS V for threshold crossing logic
        for n in self.neurons:
            n.traces["ais_V"].append(n.ais.V_mV)  # initial
        for step in range(n_steps):
            self.t = step * self.dt
            # Run due events (synaptic arrivals, etc.)
            for fn in self.events.pop_ready(self.t):
                fn()

            # External generators
            for g in self.generators:
                g.step(self.t, self.dt)

            # Neurons
            for n in self.neurons:
                n.step(self.t, self.dt)

    def get_time_vector(self) -> np.ndarray:
        n = 0
        if self.neurons:
            n = len(self.neurons[0].traces.get("soma_V", []))
        return np.arange(n, dtype=float) * self.dt


# ----------------------------- Convenience API -----------------------------

__all__ = [
    "Network", "PyramidalNeuron", "ExternalSpikeGenerator", "SynapseKind",
    "HHNaKParams", "HodgkinHuxleyNaK", "STDPParams", "LTPParams"
]


if __name__ == "__main__":
    # Minimal smoke test when running as script
    net = Network(dt=0.025, seed=1)
    pyr = PyramidalNeuron("P1", net)
    genA = ExternalSpikeGenerator("A", net, rate_hz=12.0)
    genB = ExternalSpikeGenerator("B", net, rate_hz=15.0)
    genC = ExternalSpikeGenerator("C", net, rate_hz=10.0)
    net.register_generator(genA); net.register_generator(genB); net.register_generator(genC)

    # Connect external inputs
    for i in range(5):
        pyr.attach_synapse_from(genA, "basal", i, SynapseKind.AMPA_NMDA, weight=0.5)
        pyr.attach_synapse_from(genB, "apical", i, SynapseKind.AMPA_NMDA, weight=0.4)
        pyr.attach_synapse_from(genC, "tuft",  i, SynapseKind.AMPA_NMDA, weight=0.35)

    # Axo-axonic inhibition on AIS
    genI = ExternalSpikeGenerator("I", net, rate_hz=20.0)
    net.register_generator(genI)
    pyr.attach_synapse_from(genI, "ais", 0, SynapseKind.GABA_A, weight=1.2)

    # Create three postsynaptic targets and connect main/collateral branches
    targets = [PyramidalNeuron(f"T{i}", net) for i in range(3)]
    for post in targets:
        pyr.connect_axon_to(post, "main", "basal", 0, SynapseKind.AMPA_NMDA, weight=0.5)
        pyr.connect_axon_to(post, "collateral", "apical", 0, SynapseKind.AMPA_NMDA, weight=0.35)

    net.run(150.0)
    print(f"P1 spikes: {pyr.spike_times[:10]} (total {len(pyr.spike_times)})")

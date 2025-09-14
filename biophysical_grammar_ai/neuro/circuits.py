from __future__ import annotations
from dataclasses import dataclass
from .snn_biophys import NeuronPopulation, SynapseGroup, SNN
from .oscillations import CFC

@dataclass
class CorticalLayer:
    """Represents populations of neurons in a cortical layer."""
    L2_3_PYR: NeuronPopulation
    L5_6_PYR: NeuronPopulation
    PV: NeuronPopulation         # Parvalbumin-positive (Basket, Chandelier)
    SST: NeuronPopulation        # Somatostatin-positive (Martinotti)
    VIP: NeuronPopulation        # Vasoactive Intestinal Peptide-positive

class LanguageNetwork:
    def __init__(self, n_pyr_l23=24, n_pyr_l56=24, n_pv=8, n_sst=8, n_vip=8, dt=1e-4):
        self.cfc = CFC()
        self.dt = dt

        # Define neuron counts for each area
        areas = ["V1", "STG", "MTG", "ATL", "IFG"]
        self.populations = {}
        self.layers = {}

        for area_name in areas:
            # Create Neuron Populations for the area
            l23_pyr = NeuronPopulation(N=n_pyr_l23, dt=self.dt, name=f"{area_name}_L2_3_PYR")
            l56_pyr = NeuronPopulation(N=n_pyr_l56, dt=self.dt, name=f"{area_name}_L5_6_PYR")
            pv_inh = NeuronPopulation(N=n_pv, dt=self.dt, name=f"{area_name}_PV")
            sst_inh = NeuronPopulation(N=n_sst, dt=self.dt, name=f"{area_name}_SST")
            vip_inh = NeuronPopulation(N=n_vip, dt=self.dt, name=f"{area_name}_VIP")

            # Store populations in a flat dictionary
            self.populations[f"{area_name}_L2_3_PYR"] = l23_pyr
            self.populations[f"{area_name}_L5_6_PYR"] = l56_pyr
            self.populations[f"{area_name}_PV"] = pv_inh
            self.populations[f"{area_name}_SST"] = sst_inh
            self.populations[f"{area_name}_VIP"] = vip_inh

            # Group them into a Layer object for convenience
            self.layers[area_name] = CorticalLayer(
                L2_3_PYR=l23_pyr,
                L5_6_PYR=l56_pyr,
                PV=pv_inh,
                SST=sst_inh,
                VIP=vip_inh
            )

        # Define Synapses
        self.synapses = []
        for area_name in areas:
            # Intra-area connections (Example: L2/3 PYR -> PV)
            pre_pop = self.layers[area_name].L2_3_PYR
            post_pop = self.layers[area_name].PV
            syn = SynapseGroup(
                pre=pre_pop,
                post=post_pop,
                receptor_types='AMPA', # Standard excitatory synapse
                density=0.4,           # 40% connection probability
                w_init=(1e-9, 5e-9),   # Initial conductance in Siemens
                name=f"{area_name}_L23PYR_to_PV"
            )
            self.synapses.append(syn)

        # The SNN orchestrator will manage all neurons and synapses
        self.net = SNN(neurons=list(self.populations.values()), synapses=self.synapses)

    def rhythmic_gains(self, t=0.5):
        return self.cfc.gains(t)

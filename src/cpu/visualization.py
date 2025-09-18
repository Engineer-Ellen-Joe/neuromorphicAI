import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def plot_voltage(self, voltages, neuron_id=0):
        plt.figure()
        plt.plot(voltages[:, neuron_id])
        plt.title(f"Membrane Voltage (Neuron {neuron_id})")
        plt.xlabel("Time step")
        plt.ylabel("Voltage (mV)")
        plt.show()

    def plot_spike_raster(self, spikes):
        spike_times, neuron_ids = np.where(spikes > 0)
        plt.figure()
        plt.scatter(spike_times, neuron_ids, s=2, c='black')
        plt.title("Spike Raster Plot")
        plt.xlabel("Time step")
        plt.ylabel("Neuron ID")
        plt.show()
